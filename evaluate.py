import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime import add_prediction_arguments, prediction_inputs, predict, dates_for


def cross_section_correlations(pred, target, rank=False):
    a, b = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or not all(a.shape):
        raise ValueError("Expected aligned nonempty prediction and target matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Predictions and targets must be finite")
    if rank:
        a = pd.DataFrame(a).rank(axis=1, method="average").to_numpy()
        b = pd.DataFrame(b).rank(axis=1, method="average").to_numpy()
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    scale_a, scale_b = np.abs(a).max(axis=1, keepdims=True), np.abs(b).max(
        axis=1, keepdims=True
    )
    a = np.divide(a, scale_a, out=np.zeros_like(a), where=scale_a > 0)
    b = np.divide(b, scale_b, out=np.zeros_like(b), where=scale_b > 0)
    norm = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return np.divide(
        (a * b).sum(axis=1), norm, out=np.full(len(a), np.nan), where=norm > 0
    )


def nw_ols(y, regressors=None, lags=5):

    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not isinstance(lags, int) or lags < 0:
        raise ValueError("Expected a return vector and nonnegative integer lags")
    x = (
        np.ones((len(y), 1))
        if regressors is None
        else np.column_stack([np.ones(len(y)), regressors])
    )
    good = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[good], x[good]
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) < k:
        return {
            "intercept": None,
            "intercept_nw_t": None,
            "n": n,
            "lags": min(lags, max(0, n - 1)),
        }
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    score = x * (y - x @ beta)[:, None]
    meat = score.T @ score
    lag = min(lags, n - 1)
    for j in range(1, lag + 1):
        covariance = score[j:].T @ score[:-j]
        meat += (1 - j / (lag + 1)) * (covariance + covariance.T)
    bread = np.linalg.inv(x.T @ x)
    variance = (bread @ meat @ bread)[0, 0]
    se = np.sqrt(max(0.0, variance))
    return {
        "intercept": float(beta[0]),
        "intercept_nw_t": float(beta[0] / se) if se > 0 else None,
        "n": n,
        "lags": lag,
    }


def summary(result, lags=5):
    pred, target = np.asarray(result["mu_pred"], dtype=float), np.asarray(
        result["returns"], dtype=float
    )
    ic = cross_section_correlations(pred, target)
    rank = cross_section_correlations(pred, target, True)
    finite = ic[np.isfinite(ic)]
    std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    finite_rank = rank[np.isfinite(rank)]
    rank_std = float(finite_rank.std(ddof=1)) if len(finite_rank) > 1 else 0.0

    denominator = float((target**2).sum())
    r2 = lambda p: (
        float(1 - ((target - p) ** 2).sum() / denominator) if denominator > 0 else None
    )
    return (
        {
            "IC": float(np.nanmean(ic)) if len(finite) else None,
            "ICIR": float(finite.mean() / std) if std > 0 else None,
            "RankIC": float(np.nanmean(rank)) if np.isfinite(rank).any() else None,
            "RankICIR": float(finite_rank.mean() / rank_std) if rank_std > 0 else None,
            "R2_total": r2(result["mu_total"]),
            "R2_predict": r2(pred),
            "IC_Newey_West": nw_ols(ic, lags=lags),
            "periods": len(pred),
        },
        ic,
        rank,
    )


def main():
    p = argparse.ArgumentParser(description="CASH")
    add_prediction_arguments(p, timestamps=True)
    p.add_argument("--frequency", choices=["daily", "monthly"], default="daily")
    args = p.parse_args()
    model, x, r, indices, device = prediction_inputs(args)
    dates = dates_for(args.timestamps_path, len(x))
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    output = {}
    for split, ts in indices.items():
        ts = [t for t in ts if t >= args.fbar_lookback]
        result = predict(model, x, r, ts, args.fbar_lookback, device)
        output[split], ic, rank = summary(result, 5 if args.frequency == "daily" else 4)
        pd.DataFrame({"date": dates[ts], "IC": ic, "RankIC": rank}).to_csv(
            folder / f"{split}_per_period.csv", index=False
        )
    for metric in ["IC", "RankIC"]:
        a, b = output["train"][metric], output["test"][metric]
        output[metric + "_Gap"] = (
            abs(a - b) / abs(a)
            if a is not None and b is not None and abs(a) > 1e-12
            else None
        )
    output["conventions"] = {
        "R2": "uncentered fraction, not percent",
        "ICIR": "period mean / sample standard deviation; not annualized",
        "target": args.return_type + " decimal return",
    }
    (folder / "metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
