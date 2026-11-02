import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime import add_prediction_arguments, prediction_inputs, predict, dates_for
from runtime import load_returns
from evaluate import nw_ols


def choose_holdings(scores, previous, size=50, max_replacements=10):
    if size < 1 or size > len(scores) or max_replacements < 0:
        raise ValueError("Invalid portfolio size or replacement cap")

    order = np.argsort(-np.asarray(scores), kind="stable")
    desired = set(order[:size].tolist())
    if not previous:
        return sorted(desired)
    held = set(previous)
    sells = [j for j in order[::-1] if j in held and j not in desired]
    buys = [j for j in order if j in desired and j not in held]
    count = min(max_replacements, len(sells), len(buys))
    return sorted((held - set(sells[:count])) | set(buys[:count]))


def backtest(scores, returns, dates, size=50, max_replacements=10, cost=0.003):
    scores, returns = np.asarray(scores), np.asarray(returns)
    if (
        scores.ndim != 2
        or not all(scores.shape)
        or scores.shape != returns.shape
        or len(dates) != len(scores)
    ):
        raise ValueError("Misaligned scores, realized returns, or dates")
    if (
        not 0 <= cost < 1
        or not np.isfinite(scores).all()
        or not np.isfinite(returns).all()
    ):
        raise ValueError("Invalid costs or nonfinite inputs")
    if (returns < -1).any():
        raise ValueError("Portfolio returns must be raw decimal simple returns")
    held = []
    previous_weights = np.zeros(scores.shape[1])
    nav = 1.0
    rows = []
    for date, signal, realized in zip(dates, scores, returns):
        selected = choose_holdings(signal, held, size, max_replacements)
        target = np.zeros(len(signal))
        target[selected] = 1 / size
        traded = float(np.abs(target - previous_weights).sum())
        transaction = cost * traded
        gross = float(target @ realized)
        net = (1 - transaction) * (1 + gross) - 1
        if 1 + gross <= 0 or transaction >= 1:
            raise ValueError(
                "Portfolio is bankrupt or transaction costs exhaust capital"
            )
        nav *= 1 + net
        rows.append(
            {
                "date": str(date),
                "gross_return": gross,
                "transaction_cost_fraction": transaction,
                "traded_notional": traded,
                "net_return": net,
                "portfolio_value": nav,
                "replacements": len(set(selected) - set(held)) if held else 0,
                "holdings": ";".join(map(str, selected)),
            }
        )

        previous_weights = target * (1 + realized) / (1 + gross)
        held = selected
    return pd.DataFrame(rows)


def portfolio_summary(history, periods_per_year=252, factors=None, lags=5):
    returns = history["net_return"].to_numpy()
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    out = {
        "mean_return_percent_per_period": float(returns.mean() * 100),
        "annualized_arithmetic_mean_percent": float(
            returns.mean() * periods_per_year * 100
        ),
        "sharpe_annualized": (
            float(np.sqrt(periods_per_year) * returns.mean() / std) if std > 0 else None
        ),
        "total_return_percent": float((history["portfolio_value"].iloc[-1] - 1) * 100),
        "periods": len(returns),
        "periods_per_year": periods_per_year,
    }
    if factors is not None:
        factors = factors.copy()
        required = ["date", "RF", "MKT", "SMB", "HML"]
        if any(c not in factors for c in required):
            raise ValueError("Factor file must contain date, RF, MKT, SMB and HML")
        extra = [c in factors for c in ["RMW", "CMA"]]
        if any(extra) and not all(extra):
            raise ValueError("FF5 requires both RMW and CMA")
        factors["date"] = pd.to_datetime(factors["date"])
        if factors["date"].isna().any() or factors["date"].duplicated().any():
            raise ValueError("Factor dates must be unique")
        frame = history.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.merge(factors, on="date", how="inner", validate="one_to_one")
        if len(frame) != len(history):
            raise ValueError(
                "Factor dates must cover every portfolio return; frequency must match"
            )
        if "RF" not in frame:
            raise ValueError("Factor file must include RF in decimal return units")
        numeric = required[1:] + (["RMW", "CMA"] if all(extra) else [])
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Factor returns must be finite for every portfolio date")
        for name, cols in [
            ("FF3", ["MKT", "SMB", "HML"]),
            ("FF5", ["MKT", "SMB", "HML", "RMW", "CMA"]),
        ]:
            if all(c in frame for c in cols):
                fit = nw_ols(
                    frame["net_return"].to_numpy() - frame["RF"].to_numpy(),
                    frame[cols].to_numpy(),
                    lags,
                )
                out[name] = {
                    "alpha_percent_per_period": (
                        None if fit["intercept"] is None else 100 * fit["intercept"]
                    ),
                    "nw_t": fit["intercept_nw_t"],
                    "lags": fit["lags"],
                }
    return out


def main():
    p = argparse.ArgumentParser(description="CASH")
    add_prediction_arguments(p, timestamps=True)
    p.add_argument("--portfolio-size", type=int, default=50)
    p.add_argument(
        "--max-turnover",
        type=int,
        default=10,
        help="Maximum number of holdings replaced per rebalance",
    )
    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.003,
        help="One-way cost on traded notional",
    )
    p.add_argument("--frequency", choices=["daily", "monthly"], default="daily")
    p.add_argument(
        "--realized-returns-path",
        help="Raw decimal simple returns for portfolio accounting",
    )
    p.add_argument(
        "--factors-path",
        help="Optional same-frequency date,MKT,SMB,HML,RMW,CMA,RF CSV, all returns decimal",
    )
    args = p.parse_args()
    model, x, r, indices, device = prediction_inputs(args)
    if args.realized_returns_path:
        realized = load_returns(args.realized_returns_path, args.layout, "simple")
        if realized.shape != r.shape:
            raise ValueError(
                "Realized portfolio returns must align with the model target panel"
            )
    elif args.return_type == "simple":
        realized = r
    else:
        raise ValueError(
            "Excess-return targets require --realized-returns-path for portfolio accounting"
        )
    dates = dates_for(args.timestamps_path, len(x))
    result = predict(model, x, r, indices["test"], args.fbar_lookback, device)
    history = backtest(
        result["mu_pred"],
        realized[result["indices"]].numpy(),
        dates[result["indices"]],
        args.portfolio_size,
        args.max_turnover,
        args.transaction_cost,
    )
    factors = pd.read_csv(args.factors_path) if args.factors_path else None
    out = portfolio_summary(
        history,
        252 if args.frequency == "daily" else 12,
        factors,
        5 if args.frequency == "daily" else 4,
    )
    out["strategy"] = {
        "selection": "deterministic_top_predictions",
        "size": args.portfolio_size,
        "max_replacements": args.max_turnover,
        "one_way_cost": args.transaction_cost,
        "realized_return_clipping": False,
        "model_target": args.return_type,
        "portfolio_return_type": "raw_decimal_simple_return",
    }
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    history.to_csv(folder / "portfolio_history.csv", index=False)
    (folder / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
