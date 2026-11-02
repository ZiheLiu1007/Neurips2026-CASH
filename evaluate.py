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


from itransformer_encoder import TemporalCrossSectionEncoder, NoITransformerEncoder
from factor_network import VAE_Network
from beta_network import BetaNetwork, compute_factor_pricing_returns
import train
from runtime import validate_input_contract, load_dates, nw_ols, load_realized_returns, cross_sectional_ic


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


def compute_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    return cross_sectional_ic(predicted, actual)


def compute_rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if predicted.shape != actual.shape or predicted.ndim != 1:
        raise ValueError("Expected aligned cross-sectional return vectors")
    mask = np.isfinite(predicted) & np.isfinite(actual)
    return compute_ic(pd.Series(predicted[mask]).rank(method="average").to_numpy(),
                      pd.Series(actual[mask]).rank(method="average").to_numpy())


def compute_r2(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if predicted.shape != actual.shape:
        raise ValueError("Predictions and returns must have the same shape")


    mask = np.isfinite(predicted) & np.isfinite(actual)
    if mask.sum() == 0:
        return float("nan")

    pred_clean = np.asarray(predicted[mask], dtype=float)
    actual_clean = np.asarray(actual[mask], dtype=float)

    ss_res = np.sum((actual_clean - pred_clean) ** 2)
    ss_tot = np.sum(actual_clean ** 2)

    if ss_tot == 0:
        return float("nan")

    r2 = (1 - ss_res / ss_tot) * 100
    return r2


def compute_daily_metrics(mu_total: np.ndarray, mu_pred: np.ndarray,
                          returns_true: np.ndarray) -> Dict[str, List[float]]:


    if mu_total.ndim != 2 or mu_total.shape != mu_pred.shape or mu_total.shape != returns_true.shape:
        raise ValueError("Expected aligned (T, N) prediction and return panels")
    T = mu_total.shape[0]

    daily_metrics = {
        'date_index': [],
        'ic_total': [],
        'rank_ic_total': [],
        'r2_total': [],
        'r2_pred': [],
        'num_stocks': []
    }

    for t in range(T):

        mu_total_t = mu_total[t]
        mu_pred_t = mu_pred[t]
        returns_t = returns_true[t]


        ic_t = compute_ic(mu_pred_t, returns_t)
        rank_ic_t = compute_rank_ic(mu_pred_t, returns_t)
        r2_total_t = compute_r2(mu_total_t, returns_t)
        r2_pred_t = compute_r2(mu_pred_t, returns_t)


        mask = np.isfinite(mu_pred_t) & np.isfinite(returns_t)
        num_stocks = mask.sum()

        daily_metrics['date_index'].append(t)
        daily_metrics['ic_total'].append(ic_t)
        daily_metrics['rank_ic_total'].append(rank_ic_t)
        daily_metrics['r2_total'].append(r2_total_t)
        daily_metrics['r2_pred'].append(r2_pred_t)
        daily_metrics['num_stocks'].append(num_stocks)

    return daily_metrics


def compute_aggregate_metrics(daily_metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:


    metrics = ['ic_total', 'rank_ic_total', 'r2_total', 'r2_pred']
    aggregate_metrics = {}

    for metric in metrics:
        values = np.array(daily_metrics[metric])


        valid_mask = np.isfinite(values)
        if valid_mask.sum() == 0:
            aggregate_metrics[metric] = {
                'mean': float('nan'),
                'std': float('nan'),
                'worst': float('nan'),
                'num_valid': 0
            }
            continue

        valid_values = values[valid_mask]


        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)


        if metric.startswith('r2'):
            worst_val = np.min(valid_values)
        else:
            worst_val = np.min(valid_values)

        aggregate_metrics[metric] = {
            'mean': mean_val,
            'std': std_val,
            'worst': worst_val,
            'num_valid': len(valid_values)
        }

    return aggregate_metrics


def compute_ood_drop(train_metrics: Dict[str, Dict[str, float]],
                    test_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:


    ood_drop = {}
    metrics = ['ic_total', 'rank_ic_total', 'r2_total', 'r2_pred']

    for metric in metrics:
        train_mean = train_metrics[metric]['mean']
        test_mean = test_metrics[metric]['mean']

        if train_mean == 0:
            ood_drop[metric] = float("nan")
        else:
            ood_drop[metric] = abs(test_mean - train_mean) / abs(train_mean)

    return ood_drop


@torch.no_grad()
def compute_predictions_for_indices(encoder: TemporalCrossSectionEncoder,
                                   vae: VAE_Network,
                                   beta_network: BetaNetwork,
                                   dataset: train.StockDataset,
                                   indices: list,
                                   device: torch.device,
                                   fbar_lookback: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    if fbar_lookback < 1:
        raise ValueError("fbar_lookback must be positive")
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


    mu_total_list = []
    mu_pred_list = []
    returns_list = []
    used_indices = []


    original_to_dataset_idx = {orig_idx: idx for idx, orig_idx in enumerate(dataset.valid_indices)}

    for t_original in indices:
        if t_original not in original_to_dataset_idx:
            continue

        dataset_idx = original_to_dataset_idx[t_original]
        if t_original < fbar_lookback:
            continue


        beta_prev = all_beta[dataset_idx]


        f_curr = all_f[dataset_idx]


        history_returns = dataset.returns[t_original - fbar_lookback:t_original].to(device)
        f_bar = vae.get_factor_returns(history_returns, use_mean=True).mean(dim=0).cpu().numpy()


        mu_total = np.einsum("nk,k->n", beta_prev, f_curr)


        mu_pred = np.einsum("nk,k->n", beta_prev, f_bar)

        mu_total_list.append(mu_total)
        mu_pred_list.append(mu_pred)
        returns_list.append(all_returns[dataset_idx])
        used_indices.append(t_original)

    if not mu_pred_list:
        raise ValueError("No forecast periods have the required observable history")

    return (
        np.array(mu_total_list),
        np.array(mu_pred_list),
        np.array(returns_list),
        np.array(used_indices)
    )


def resolve_dates_from_indices(indices: np.ndarray, timestamps_path: Optional[str]) -> List[str]:
    if not timestamps_path:
        return [str(int(idx)) for idx in indices]
    dates = load_dates(timestamps_path)
    if any(int(idx) < 0 or int(idx) >= len(dates) for idx in indices):
        raise ValueError("Forecast indices exceed the timestamp axis")
    return [dates[int(idx)].strftime("%Y-%m-%d") for idx in indices]


def build_fit_summary_df(date_indices: np.ndarray,
                         dates: List[str],
                         mu_total: np.ndarray,
                         mu_pred: np.ndarray,
                         returns_true: np.ndarray) -> pd.DataFrame:
    if len(date_indices) == 0:
        return pd.DataFrame(columns=[
            'date_index', 'date', 'actual_return_mean',
            'predicted_return_mean', 'total_return_mean', 'num_stocks'
        ])

    return pd.DataFrame({
        'date_index': date_indices.astype(int),
        'date': dates,
        'actual_return_mean': np.nanmean(returns_true, axis=1),
        'predicted_return_mean': np.nanmean(mu_pred, axis=1),
        'total_return_mean': np.nanmean(mu_total, axis=1),
        'num_stocks': np.sum(np.isfinite(returns_true), axis=1),
    })


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="CASH模型增强评估")
        parser.add_argument("--model-dir", type=str, default="outputs/cash", help="模型目录")
        parser.add_argument("--layout", choices=["NTP", "TNP"], default="NTP")
        parser.add_argument("--return-type", choices=["simple", "excess"], default="excess")
        parser.add_argument("--features-path", type=str, default="price_and_factor.npy", help="特征文件路径")
        parser.add_argument("--returns-path", type=str, default="return.npy", help="收益文件路径")
        parser.add_argument("--timestamps-path", type=str, required=True, help="时间戳文件路径")
        parser.add_argument("--output-dir", type=str, default=".", help="结果输出目录")
        parser.add_argument("--cuda", type=int, default=0, help="GPU编号")
        parser.add_argument("--fbar-lookback", type=int, default=3, help="计算 mu_pred 时 f_bar 的回看期数")
        parser.add_argument("--frequency", choices=["daily", "monthly"], default="daily")
        args = parser.parse_args()


    model_dir = args.model_dir
    features_path = args.features_path
    returns_path = args.returns_path
    output_dir = args.output_dir
    output_csv = os.path.join(output_dir, "增强评估指标.csv")
    daily_metrics_csv = os.path.join(output_dir, "每日环境表现.csv")
    train_fit_csv = os.path.join(output_dir, "样本内每日拟合.csv")
    test_fit_csv = os.path.join(output_dir, "样本外每日拟合.csv")
    cuda_device = args.cuda

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
    disable_itransformer = bool(config_dict.get('disable_itransformer', False))
    vae_hidden_dim = config_dict.get('vae_hidden_dim', 32)
    vae_beta = config_dict.get('vae_beta', 1.0)
    beta_hidden_dim = config_dict.get('beta_hidden_dim', None)
    train_ratio = config_dict.get('train_ratio', 0.7)
    val_ratio = config_dict.get('val_ratio', 0.1)


    feats, rets = train.load_data(features_path, returns_path, args.layout, args.return_type)
    validate_input_contract(config_dict, feats, rets, args.layout, args.return_type)
    T, N, P = feats.shape
    load_dates(args.timestamps_path, T)


    dataset = train.StockDataset(feats, rets, window=window)


    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)


    train_indices = list(range(max(window, args.fbar_lookback), train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(max(val_end, window, args.fbar_lookback), T))
    if not test_indices:
        raise ValueError("No test periods have the required history")


    if disable_itransformer:
        encoder = NoITransformerEncoder(
            input_dim=P,
            z_dim=z_dim,
            window=window,
            nhead=transformer_nhead,
            num_layers=transformer_layers_time,
            pooling=config_dict.get("noitr_pooling", "last"),
            dropout=transformer_dropout,
        ).to(device)
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


    print(f"Evaluating {len(test_indices)} test periods on {device}", flush=True)
    mu_total_train, mu_pred_train, returns_train, train_used_indices = compute_predictions_for_indices(
        encoder,
        vae,
        beta_network,
        dataset,
        train_indices,
        device,
        fbar_lookback=int(args.fbar_lookback),
    )


    daily_metrics_train = compute_daily_metrics(mu_total_train, mu_pred_train, returns_train)
    aggregate_metrics_train = compute_aggregate_metrics(daily_metrics_train)


    mu_total_test, mu_pred_test, returns_test, test_used_indices = compute_predictions_for_indices(
        encoder,
        vae,
        beta_network,
        dataset,
        test_indices,
        device,
        fbar_lookback=int(args.fbar_lookback),
    )


    daily_metrics_test = compute_daily_metrics(mu_total_test, mu_pred_test, returns_test)
    aggregate_metrics_test = compute_aggregate_metrics(daily_metrics_test)


    ood_drop = compute_ood_drop(aggregate_metrics_train, aggregate_metrics_test)


    daily_df = pd.DataFrame({
        'date_index': test_used_indices,
        'ic_total': daily_metrics_test['ic_total'],
        'rank_ic_total': daily_metrics_test['rank_ic_total'],
        'r2_total': daily_metrics_test['r2_total'],
        'r2_pred': daily_metrics_test['r2_pred'],
        'num_stocks': daily_metrics_test['num_stocks']
    })


    daily_df['date'] = resolve_dates_from_indices(test_used_indices, getattr(args, "timestamps_path", None))


    daily_df = daily_df[['date_index', 'date', 'ic_total', 'rank_ic_total', 'r2_total', 'r2_pred', 'num_stocks']]


    daily_df.to_csv(daily_metrics_csv, index=False, encoding='utf-8')

    train_dates = resolve_dates_from_indices(train_used_indices, getattr(args, "timestamps_path", None))
    test_dates = resolve_dates_from_indices(test_used_indices, getattr(args, "timestamps_path", None))
    train_fit_df = build_fit_summary_df(train_used_indices, train_dates, mu_total_train, mu_pred_train, returns_train)
    test_fit_df = build_fit_summary_df(test_used_indices, test_dates, mu_total_test, mu_pred_test, returns_test)
    train_fit_df.to_csv(train_fit_csv, index=False, encoding='utf-8')
    test_fit_df.to_csv(test_fit_csv, index=False, encoding='utf-8')


    enhanced_metrics = []
    metrics_info = [
        ('IC', 'ic_total', '信息系数'),
        ('RankIC', 'rank_ic_total', '等级信息系数'),
        ('R²_total', 'r2_total', '总决定系数'),
        ('R²_pred', 'r2_pred', '预测决定系数')
    ]

    for metric_name, metric_key, description in metrics_info:

        train_mean = aggregate_metrics_train[metric_key]['mean']
        train_std = aggregate_metrics_train[metric_key]['std']
        train_worst = aggregate_metrics_train[metric_key]['worst']
        train_valid = aggregate_metrics_train[metric_key]['num_valid']


        test_mean = aggregate_metrics_test[metric_key]['mean']
        test_std = aggregate_metrics_test[metric_key]['std']
        test_worst = aggregate_metrics_test[metric_key]['worst']
        test_valid = aggregate_metrics_test[metric_key]['num_valid']


        ood_drop_val = ood_drop[metric_key]

        enhanced_metrics.append({
            'Metric': metric_name,
            'Description': description,
            'Train_Mean': f"{train_mean:.6f}",
            'Train_Std': f"{train_std:.6f}",
            'Train_Worst': f"{train_worst:.6f}",
            'Train_Valid_Envs': train_valid,
            'Test_Mean': f"{test_mean:.6f}",
            'Test_Std': f"{test_std:.6f}",
            'Test_Worst': f"{test_worst:.6f}",
            'Test_Valid_Envs': test_valid,
            'OOD_Drop': f"{ood_drop_val:.6f}"
        })


    enhanced_df = pd.DataFrame(enhanced_metrics)


    enhanced_df.to_csv(output_csv, index=False, encoding='utf-8')
    direct = {}
    for label, aggregate, daily in [
        ("train", aggregate_metrics_train, daily_metrics_train),
        ("test", aggregate_metrics_test, daily_metrics_test),
    ]:
        direct[label] = {k: {key: (float(value) if np.isfinite(value) else None)
                             for key, value in values.items()}
                         for k, values in aggregate.items()}
        for key, name in [("ic_total", "ICIR"), ("rank_ic_total", "RankICIR")]:
            value = aggregate[key]
            direct[label][name] = (float(value["mean"] / value["std"])
                                   if value["std"] > 0 else None)
        direct[label]["IC_Newey_West"] = nw_ols(
            daily["ic_total"], lags=5 if args.frequency == "daily" else 4)
    direct["OOD_Drop"] = {k: float(v) if np.isfinite(v) else None for k, v in ood_drop.items()}
    direct["conventions"] = {"R2": "mean of per-period uncentered R2 in percent",
                              "IC": "observable-history mu_pred",
                              "std": "cross-time population standard deviation",
                              "return_type": args.return_type}
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(direct, handle, indent=2, allow_nan=False)


    print(f"Evaluation saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
