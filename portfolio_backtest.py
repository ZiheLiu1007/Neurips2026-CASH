import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import json
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import gc
import random


from itransformer_encoder import TemporalCrossSectionEncoder, NoITransformerEncoder
from factor_network import VAE_Network
from beta_network import BetaNetwork, compute_factor_pricing_returns
import train
from runtime import validate_input_contract, load_dates, nw_ols, load_realized_returns


def get_safe_device(cuda_index: int = 0) -> torch.device:

    force_cpu = os.environ.get("FORCE_CPU", "0").lower() in {"1", "true", "yes"}
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_index}")
        try:
            probe = torch.randn(8, 8, device=device)
            _ = (probe @ probe).sum().item()
            bn = nn.BatchNorm1d(8).to(device)
            _ = bn(torch.randn(4, 8, device=device)).sum().item()
            return device
        except Exception as exc:
            print(f"CUDA 不可用，自动回退到 CPU: {exc}")

    return torch.device("cpu")


def load_time_axis(timestamps_path: str) -> List[str]:
    return load_dates(timestamps_path).strftime("%Y-%m-%d").tolist()


def _self_financing_trade(previous_weights, target_weights, transaction_cost):
    lower, upper = 0.0, 1.0
    for _ in range(80):
        invested = (lower + upper) / 2
        traded = float(np.abs(invested * target_weights - previous_weights).sum())
        if invested + transaction_cost * traded > 1:
            upper = invested
        else:
            lower = invested
    invested = (lower + upper) / 2
    traded = float(np.abs(invested * target_weights - previous_weights).sum())
    return invested, traded, transaction_cost * traded


class EnhancedPortfolioManager:


    def __init__(self, portfolio_size: int = 50, max_turnover: int = 10,
                 transaction_cost: float = 0.003, random_seed: int = None,
                 date_lookup: Optional[List[str]] = None, periods_per_year: int = 252):
        if portfolio_size < 1 or max_turnover < 0 or not 0 <= transaction_cost < 1:
            raise ValueError("Invalid portfolio size, replacement limit, or transaction cost")
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be positive")
        self.portfolio_size = portfolio_size
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        self.date_lookup = [] if date_lookup is None else list(date_lookup)
        self.periods_per_year = periods_per_year
        self.reset()

    def initialize_portfolio_random(self, date_index: int, predictions: np.ndarray,
                                   available_stocks: List[int] = None) -> List[int]:
        raise ValueError("Random portfolio selection is outside the specified strategy")

    def initialize_portfolio_top_predictions(self, date_index: int, predictions: np.ndarray) -> List[int]:
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim != 1 or not np.isfinite(predictions).all():
            raise ValueError("Predictions must be a finite cross-sectional vector")
        if len(predictions) < self.portfolio_size:
            raise ValueError("Not enough assets for the requested portfolio size")
        self.current_portfolio = np.argsort(-predictions, kind="stable")[:self.portfolio_size].tolist()
        self.initial_selection_method = "top_predictions"
        return self.current_portfolio

    def rebalance_portfolio(self, date_index: int, predictions: np.ndarray,
                            actual_returns: np.ndarray) -> List[int]:
        predictions = np.asarray(predictions, dtype=float)
        actual_returns = np.asarray(actual_returns, dtype=float)
        if predictions.ndim != 1 or actual_returns.shape != predictions.shape:
            raise ValueError("Predictions and realized returns must be aligned vectors")
        if not np.isfinite(predictions).all() or not np.isfinite(actual_returns).all():
            raise ValueError("Predictions and realized returns must be finite")
        if np.any(actual_returns < -1) or len(predictions) < self.portfolio_size:
            raise ValueError("Use raw decimal simple returns and enough assets")
        if self.portfolio_history and date_index <= self.portfolio_history[-1]["date_index"]:
            raise ValueError("Portfolio periods must be processed once in chronological order")
        order = np.argsort(-predictions, kind="stable")
        desired = set(order[:self.portfolio_size].tolist())
        old_set = set(self.current_portfolio or [])
        first_period = self.previous_weights is None
        if first_period:
            selected = sorted(desired)
            replacements = 0
            self.previous_weights = np.zeros(len(predictions))
        else:
            if len(predictions) != len(self.previous_weights):
                raise ValueError("Asset universe changed during the backtest")
            sells = [j for j in order[::-1] if j in old_set and j not in desired]
            buys = [j for j in order if j in desired and j not in old_set]
            replacements = min(self.max_turnover, len(sells), len(buys))
            selected = sorted((old_set - set(sells[:replacements])) | set(buys[:replacements]))
        self.current_portfolio = selected
        weights = np.zeros(len(predictions))
        weights[selected] = 1.0 / self.portfolio_size
        invested, traded, cost_fraction = _self_financing_trade(
            self.previous_weights, weights, self.transaction_cost)
        transaction_cost = self.portfolio_value * cost_fraction
        daily_gross_return = self._calculate_daily_return(actual_returns)
        if cost_fraction >= 1 or 1 + daily_gross_return <= 0:
            raise ValueError("Portfolio is bankrupt or costs exhaust capital")
        daily_return = invested * (1 + daily_gross_return) - 1
        self.portfolio_value *= 1 + daily_return
        self.daily_returns.append(daily_return)
        self.transaction_cost_history.append(transaction_cost)
        turnover = self.portfolio_size if first_period else 2 * replacements
        self.turnover_history.append(turnover)
        self.last_gross_return = daily_gross_return
        self.last_traded_notional = traded
        self.last_cost_fraction = cost_fraction
        self.last_replacements = replacements
        self._record_portfolio_state(date_index, "TOP_PREDICTIONS_INIT" if first_period else "REBALANCE",
                                     daily_return=daily_return, transaction_cost=transaction_cost,
                                     turnover=turnover)
        self.previous_weights = weights * (1 + actual_returns) / (1 + daily_gross_return)
        return self.current_portfolio

    def _calculate_daily_return(self, actual_returns: np.ndarray) -> float:
        if self.current_portfolio is None:
            return 0.0
        return float(np.mean(np.asarray(actual_returns)[self.current_portfolio]))

    def _calculate_transaction_cost(self, old_portfolio: set, new_portfolio: set) -> float:
        target = np.zeros(len(self.previous_weights))
        target[list(new_portfolio)] = 1.0 / self.portfolio_size
        _, _, cost_fraction = _self_financing_trade(
            self.previous_weights, target, self.transaction_cost)
        return self.portfolio_value * cost_fraction

    def _record_portfolio_state(self, date_index: int, action: str,
                                daily_return: Optional[float] = None,
                                transaction_cost: Optional[float] = None,
                                turnover: Optional[int] = None):

        trade_date = None
        if 0 <= date_index < len(self.date_lookup):
            trade_date = self.date_lookup[date_index]

        self.portfolio_history.append({
            'date_index': date_index,
            'trade_date': trade_date,
            'portfolio_value': self.portfolio_value,
            'daily_return': daily_return,
            'transaction_cost': transaction_cost,
            'turnover': turnover,
            'num_stocks': len(self.current_portfolio) if self.current_portfolio else 0,
            'action': action,
            'cash': self.cash,
            'initial_selection_method': self.initial_selection_method,
            'gross_return': self.last_gross_return,
            'net_return': daily_return,
            'traded_notional': self.last_traded_notional,
            'transaction_cost_fraction': self.last_cost_fraction,
            'replacements': self.last_replacements,
            'holdings': ';'.join(map(str, self.current_portfolio)),
        })

    def get_portfolio_history_df(self) -> pd.DataFrame:

        return pd.DataFrame(self.portfolio_history)

    def get_performance_metrics(self) -> Dict[str, float]:

        if len(self.daily_returns) == 0:
            return {}

        daily_returns_array = np.array(self.daily_returns)


        total_return = self.portfolio_value - 1.0


        if len(daily_returns_array) > 0:
            annual_return = (1 + total_return) ** (self.periods_per_year / len(daily_returns_array)) - 1
        else:
            annual_return = 0.0


        if len(daily_returns_array) > 1:
            annual_volatility = np.std(daily_returns_array, ddof=1) * np.sqrt(self.periods_per_year)
        else:
            annual_volatility = 0.0


        if annual_volatility > 0:
            sharpe_ratio = np.mean(daily_returns_array) * self.periods_per_year / annual_volatility
        else:
            sharpe_ratio = 0.0


        portfolio_values = [1.0]
        current_value = 1.0
        for ret in self.daily_returns:
            current_value *= (1 + ret)
            portfolio_values.append(current_value)

        portfolio_values_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values_array)
        drawdown = (portfolio_values_array - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0


        avg_turnover = np.mean(self.turnover_history) if self.turnover_history else 0.0


        total_transaction_cost = np.sum(self.transaction_cost_history)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': avg_turnover,
            'total_transaction_cost': total_transaction_cost,
            'final_portfolio_value': self.portfolio_value,
            'initial_selection_method': self.initial_selection_method,
            'mean_return_percent': float(np.mean(daily_returns_array) * 100),
            'annualized_arithmetic_mean_percent': float(np.mean(daily_returns_array) * self.periods_per_year * 100),
            'periods_per_year': self.periods_per_year,
            'periods': len(daily_returns_array),
            'avg_traded_notional': float(np.mean([row['traded_notional'] for row in self.portfolio_history])),
        }

    def reset(self):
        self.current_portfolio = None
        self.previous_weights = None
        self.portfolio_value = 1.0
        self.cash = 0.0
        self.portfolio_history = []
        self.daily_returns = []
        self.turnover_history = []
        self.transaction_cost_history = []
        self.initial_selection_method = "top_predictions"
        self.last_gross_return = 0.0
        self.last_traded_notional = 0.0
        self.last_cost_fraction = 0.0
        self.last_replacements = 0


class MultiPortfolioTester:


    def __init__(
        self,
        num_tests: int = 1,
        portfolio_size: int = 50,
        max_turnover: int = 10,
        transaction_cost: float = 0.003,
        init_method: str = "top_predictions",
        fbar_lookback: int = 3,
        filter_returns: bool = False,
    ):
        if num_tests != 1 or init_method != "top_predictions" or filter_returns:
            raise ValueError("Use one deterministic top-prediction backtest without return filtering")
        self.num_tests = num_tests
        self.portfolio_size = portfolio_size
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        self.init_method = init_method
        self.fbar_lookback = max(1, fbar_lookback)
        self.filter_returns = filter_returns
        self.best_portfolio = None
        self.best_final_value = 0.0
        self.all_results = []

    def run_multiple_tests(self, encoder: TemporalCrossSectionEncoder,
                          vae: VAE_Network, beta_network: BetaNetwork,
                          dataset: train.StockDataset, test_indices: list,
                          device: torch.device,
                          test_dates: Optional[List[str]] = None,
                          realized_returns: Optional[np.ndarray] = None,
                          periods_per_year: int = 252) -> Tuple[EnhancedPortfolioManager, pd.DataFrame]:


        mu_pred, returns_true, returns_filtered = compute_predictions_for_indices(
            encoder,
            vae,
            beta_network,
            dataset,
            test_indices,
            device,
            fbar_lookback=self.fbar_lookback,
            filter_returns=self.filter_returns,
        )

        if realized_returns is None:
            raise ValueError("Provide aligned raw decimal simple realized returns")
        realized_returns = np.asarray(realized_returns, dtype=np.float64)
        if realized_returns.shape != tuple(dataset.returns.shape):
            raise ValueError("Realized return panel must align with the model target panel")
        used_indices = [t for t in test_indices if t in dataset.valid_indices and t >= self.fbar_lookback]
        returns_filtered = realized_returns[used_indices]
        if not np.isfinite(returns_filtered).all() or np.any(returns_filtered < -1):
            raise ValueError("Use finite raw decimal simple realized returns")
        self.prediction_cache = (mu_pred, returns_filtered, np.asarray(used_indices))
        T_test = len(mu_pred)


        actual_num_tests = self.num_tests if self.init_method == "random" else 1

        for test_id in range(actual_num_tests):


            portfolio_manager = EnhancedPortfolioManager(
                portfolio_size=self.portfolio_size,
                max_turnover=self.max_turnover,
                transaction_cost=self.transaction_cost,
                random_seed=test_id,
                date_lookup=test_dates,
                periods_per_year=periods_per_year,
            )


            for t in range(T_test):
                selected_stocks = portfolio_manager.rebalance_portfolio(
                    t, mu_pred[t], returns_filtered[t]
                )


            final_value = portfolio_manager.portfolio_value
            self.all_results.append({
                'test_id': test_id,
                'final_portfolio_value': final_value,
                'total_return': final_value - 1.0,
                'num_stocks': len(portfolio_manager.current_portfolio) if portfolio_manager.current_portfolio else 0,
                'initial_selection_method': portfolio_manager.initial_selection_method,
            })

            self.best_final_value = final_value
            self.best_portfolio = portfolio_manager


        all_results_df = pd.DataFrame(self.all_results)

        return self.best_portfolio, all_results_df


@torch.no_grad()
def compute_predictions_for_indices(encoder: TemporalCrossSectionEncoder,
                                   vae: VAE_Network,
                                   beta_network: BetaNetwork,
                                   dataset: train.StockDataset,
                                   indices: list,
                                   device: torch.device,
                                   fbar_lookback: int = 3,
                                   filter_returns: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:


    if fbar_lookback < 1:
        raise ValueError("fbar_lookback must be positive")
    if filter_returns:
        raise ValueError("Return filtering must be performed externally, never inside a backtest")
    encoder.eval()
    vae.eval()
    beta_network.eval()

    T_dataset = len(dataset)
    N = dataset.N


    all_f = []
    all_beta = []
    all_returns = []

    for idx in range(T_dataset):
        batch = dataset[idx]
        hist_features = batch["hist_features"].unsqueeze(0).to(device)
        returns = batch["returns"].unsqueeze(0).to(device)

        F_t = vae.get_factor_returns(returns, use_mean=True)


        Z = encoder(hist_features)


        beta_Z = beta_network(Z)

        all_f.append(F_t.cpu().numpy())
        all_beta.append(beta_Z.cpu().numpy())
        all_returns.append(returns.cpu().numpy())


    all_f = np.concatenate(all_f, axis=0)
    all_beta = np.concatenate(all_beta, axis=0)
    if not np.isfinite(all_f).all() or not np.isfinite(all_beta).all():
        raise FloatingPointError("Nonfinite factor or loading estimates")
    all_returns = np.concatenate(all_returns, axis=0)


    mu_pred_list = []
    returns_list = []
    returns_filtered_list = []


    original_to_dataset_idx = {orig_idx: idx for idx, orig_idx in enumerate(dataset.valid_indices)}

    for t_original in indices:
        if t_original not in original_to_dataset_idx:
            continue

        dataset_idx = original_to_dataset_idx[t_original]
        if t_original < fbar_lookback:
            continue


        beta_prev = all_beta[dataset_idx]


        history_returns = dataset.returns[t_original - fbar_lookback:t_original].to(device)
        f_bar = vae.get_factor_returns(history_returns, use_mean=True).mean(dim=0).cpu().numpy()


        mu_pred = np.einsum("nk,k->n", beta_prev, f_bar)


        returns_true = all_returns[dataset_idx]


        returns_filtered = returns_true.copy()

        mu_pred_list.append(mu_pred)
        returns_list.append(returns_true)
        returns_filtered_list.append(returns_filtered)

    if not mu_pred_list:
        raise ValueError("No forecast periods have the required observable history")

    return (
        np.array(mu_pred_list),
        np.array(returns_list),
        np.array(returns_filtered_list)
    )


def factor_adjusted_metrics(history: pd.DataFrame, factors: pd.DataFrame, lags: int = 5) -> dict:
    required = ["date", "RF", "MKT", "SMB", "HML"]
    if any(column not in factors for column in required):
        raise ValueError("Factors require date, RF, MKT, SMB and HML in decimal return units")
    extra = [column in factors for column in ["RMW", "CMA"]]
    if any(extra) and not all(extra):
        raise ValueError("FF5 requires both RMW and CMA")
    factors = factors.copy()
    factors["date"] = pd.to_datetime(factors["date"], errors="raise")
    if factors["date"].hasnans or factors["date"].duplicated().any():
        raise ValueError("Factor dates must be unique")
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="raise")
    frame = frame.merge(factors, on="date", how="inner", validate="one_to_one")
    if len(frame) != len(history):
        raise ValueError("Same-frequency factor dates must cover every portfolio return")
    numeric = required[1:] + (["RMW", "CMA"] if all(extra) else [])
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Factor returns must be finite")
    output = {}
    for label, columns in [("ff3", ["MKT", "SMB", "HML"]),
                           ("ff5", ["MKT", "SMB", "HML", "RMW", "CMA"])]:
        if all(column in frame for column in columns):
            fit = nw_ols(frame["net_return"].to_numpy() - frame["RF"].to_numpy(),
                         frame[columns].to_numpy(), lags)
            output[label + "_alpha_percent"] = None if fit["intercept"] is None else fit["intercept"] * 100
            output[label + "_nw_t"] = fit["intercept_nw_t"]
    return output


def run_enhanced_portfolio_experiment(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description="CASH模型增强投资组合实验")
        parser.add_argument("--model-dir", type=str, default="outputs/cash", help="模型目录")
        parser.add_argument("--layout", choices=["NTP", "TNP"], default="NTP")
        parser.add_argument("--return-type", choices=["simple", "excess"], default="excess")
        parser.add_argument("--features-path", type=str, default="price_and_factor.npy", help="特征文件路径")
        parser.add_argument("--returns-path", type=str, default="return.npy", help="收益文件路径")
        parser.add_argument("--timestamps-path", type=str, default="time_stamp.npy", help="时间戳文件路径")
        parser.add_argument("--output-dir", type=str, default=".", help="结果输出目录")
        parser.add_argument("--cuda", type=int, default=0, help="GPU编号")
        parser.add_argument("--num-random-tests", type=int, default=1, help="兼容参数，确定性策略仅支持1次")
        parser.add_argument("--portfolio-size", type=int, default=50, help="持仓股票数量")
        parser.add_argument("--max-turnover", type=int, default=10, help="单次调仓最多更换股票数")
        parser.add_argument("--transaction-cost", type=float, default=0.003, help="单边交易成本")
        parser.add_argument("--init-method", type=str, default="top_predictions", choices=["top_predictions"], help="初始持仓方式")
        parser.add_argument("--fbar-lookback", type=int, default=3, help="计算 f_bar 时使用的历史期数")
        parser.add_argument("--realized-returns-path", default=None)
        parser.add_argument("--factors-path", default=None)
        parser.add_argument("--frequency", choices=["daily", "monthly"], default="daily")
        args = parser.parse_args()


    model_dir = args.model_dir
    features_path = args.features_path
    returns_path = args.returns_path
    timestamps_path = args.timestamps_path
    output_dir = args.output_dir
    output_csv = os.path.join(output_dir, "投资组合绩效汇总.csv")
    portfolio_history_csv = os.path.join(output_dir, "投资组合每日历史.csv")
    all_tests_csv = os.path.join(output_dir, "全部随机组合结果.csv")
    config_json = os.path.join(output_dir, "投资组合实验配置.json")
    cuda_device = args.cuda
    num_random_tests = args.num_random_tests
    portfolio_size = args.portfolio_size
    max_turnover = args.max_turnover
    transaction_cost = args.transaction_cost
    init_method = args.init_method
    fbar_lookback = int(args.fbar_lookback)
    if fbar_lookback < 1:
        raise ValueError("fbar_lookback must be positive")
    filter_returns = False
    if num_random_tests != 1 or init_method != "top_predictions":
        raise ValueError("Only one deterministic top-prediction portfolio is supported")

    os.makedirs(output_dir, exist_ok=True)

    device = get_safe_device(cuda_device)


    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)


    window = config_dict.get('window', 20)
    factor_dim = config_dict.get('factor_dim', 8)
    z_dim = config_dict.get('z_dim', 64)
    transformer_nhead = config_dict.get('transformer_nhead', 4)
    transformer_layers_time = config_dict.get('transformer_layers_time', 2)
    transformer_layers_cross = config_dict.get('transformer_layers_cross', 2)
    transformer_dropout = config_dict.get('transformer_dropout', 0.1)
    vae_hidden_dim = config_dict.get('vae_hidden_dim', 32)
    vae_beta = config_dict.get('vae_beta', 1.0)
    disable_itransformer = bool(config_dict.get("disable_itransformer", False))
    beta_hidden_dim = config_dict.get('beta_hidden_dim', None)
    train_ratio = config_dict.get('train_ratio', 0.7)
    val_ratio = config_dict.get('val_ratio', 0.1)


    portfolio_config = {
        "model_dir": model_dir,
        "features_path": features_path,
        "returns_path": returns_path,
        "timestamps_path": timestamps_path,
        "output_dir": output_dir,
        "cuda": cuda_device,
        "num_random_tests": num_random_tests,
        "portfolio_size": portfolio_size,
        "max_turnover": max_turnover,
        "transaction_cost": transaction_cost,
        "init_method": init_method,
        "fbar_lookback": fbar_lookback,
        "filter_returns": filter_returns,
        "model_return_type": args.return_type,
        "portfolio_return_type": "simple",
        "frequency": args.frequency,
        "window": window,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(portfolio_config, f, indent=2, ensure_ascii=False)


    feats, rets = train.load_data(features_path, returns_path, args.layout, args.return_type)
    validate_input_contract(config_dict, feats, rets, args.layout, args.return_type)
    if args.realized_returns_path:
        realized_returns = load_realized_returns(args.realized_returns_path, args.layout)
    elif args.return_type == "simple":
        realized_returns = load_realized_returns(returns_path, args.layout)
    else:
        raise ValueError("Excess-return targets require --realized-returns-path for portfolio accounting")
    if realized_returns.shape != rets.shape:
        raise ValueError("Realized return panel must align with model targets")
    time_axis = load_dates(timestamps_path, len(feats)).strftime("%Y-%m-%d").tolist()
    T, N, P = feats.shape


    dataset = train.StockDataset(feats, rets, window=window)


    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)


    test_indices = list(range(max(val_end, window, args.fbar_lookback), T))
    if not test_indices:
        raise ValueError("No test periods have the required history")
    test_dates = [time_axis[idx] for idx in test_indices]


    if disable_itransformer:
        encoder = NoITransformerEncoder(input_dim=P, z_dim=z_dim, window=window,
                                        nhead=transformer_nhead, num_layers=transformer_layers_time,
                                        pooling=config_dict.get("noitr_pooling", "last"),
                                        dropout=transformer_dropout).to(device)
    else:
        encoder = TemporalCrossSectionEncoder(
            input_dim=P,
            z_dim=z_dim,
            nhead=transformer_nhead,
            num_layers_time=transformer_layers_time,
            num_layers_cross=transformer_layers_cross,
            dropout=transformer_dropout,
            window=window,
        ).to(device)
    vae = VAE_Network(
        input_dim=N,
        latent_dim=factor_dim,
        output_dim=N,
        hidden_dim=vae_hidden_dim,
        beta=vae_beta
    ).to(device)

    beta_network = BetaNetwork(
        z_dim=z_dim,
        factor_dim=factor_dim,
        hidden_dim=beta_hidden_dim if beta_hidden_dim and beta_hidden_dim > 0 else z_dim
    ).to(device)


    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder'])
    vae.load_state_dict(checkpoint['vae'])
    beta_network.load_state_dict(checkpoint['beta_network'])


    multi_tester = MultiPortfolioTester(
        num_tests=num_random_tests,
        portfolio_size=portfolio_size,
        max_turnover=max_turnover,
        transaction_cost=transaction_cost,
        init_method=init_method,
        fbar_lookback=fbar_lookback,
        filter_returns=filter_returns,
    )


    print(f"Backtesting {len(test_indices)} periods on {device}", flush=True)
    best_portfolio_manager, all_tests_df = multi_tester.run_multiple_tests(
        encoder, vae, beta_network, dataset, test_indices, device, test_dates=test_dates,
        realized_returns=realized_returns, periods_per_year=252 if args.frequency == "daily" else 12
    )


    all_tests_df.to_csv(all_tests_csv, index=False)


    portfolio_history_df = best_portfolio_manager.get_portfolio_history_df()
    portfolio_history_df['date_index'] = multi_tester.prediction_cache[2]
    portfolio_history_df.to_csv(portfolio_history_csv, index=False)


    performance_metrics = best_portfolio_manager.get_performance_metrics()
    if args.factors_path:
        performance_metrics.update(factor_adjusted_metrics(
            portfolio_history_df, pd.read_csv(args.factors_path),
            lags=5 if args.frequency == "daily" else 4))
    cached_pred, cached_returns, cached_indices = multi_tester.prediction_cache
    np.save(os.path.join(output_dir, "mu_pred.npy"), cached_pred)
    np.save(os.path.join(output_dir, "returns_true.npy"), cached_returns)
    np.save(os.path.join(output_dir, "prediction_indices.npy"), cached_indices)
    with open(os.path.join(output_dir, "prediction_metadata.json"), "w") as handle:
        json.dump({"return_type": "simple", "layout": "TNP", "source_model_return_type": args.return_type,
                   "dates": test_dates, "used_indices": cached_indices.tolist()}, handle, indent=2)
    performance_metrics.update({
        "portfolio_size": portfolio_size,
        "max_turnover_limit": max_turnover,
        "transaction_cost_rate": transaction_cost,
        "num_random_tests": num_random_tests,
        "init_method": init_method,
        "fbar_lookback": fbar_lookback,
        "filter_returns": filter_returns,
        "model_return_type": args.return_type,
        "portfolio_return_type": "simple",
        "frequency": args.frequency,
        "window": window,
    })
    results_df = pd.DataFrame([performance_metrics])
    results_df.to_csv(output_csv, index=False)


    print(f"Portfolio return={performance_metrics['total_return']:.6%}; saved to {output_dir}", flush=True)
    return performance_metrics, portfolio_history_df, all_tests_df


def main():
    run_enhanced_portfolio_experiment()


if __name__ == "__main__":
    main()
