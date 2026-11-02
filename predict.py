import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
from dataclasses import dataclass
from typing import Tuple


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


def compute_r2(mu: np.ndarray, r: np.ndarray) -> float:
    mu, r = np.asarray(mu), np.asarray(r)
    if mu.shape != r.shape:
        raise ValueError("Predictions and returns must have the same shape")


    mask = np.isfinite(mu) & np.isfinite(r)
    if mask.sum() == 0:
        return float("nan")

    mu_c, r_c = np.asarray(mu[mask], dtype=float), np.asarray(r[mask], dtype=float)
    ss_res = np.sum((r_c - mu_c) ** 2)
    ss_tot = np.sum(r_c ** 2)

    if ss_tot == 0:
        return float("nan")

    return (1 - ss_res / ss_tot) * 100


@torch.no_grad()
def compute_test_predictions(encoder: TemporalCrossSectionEncoder,
                             vae: VAE_Network,
                             beta_network: BetaNetwork,
                             dataset: train.StockDataset,
                             test_indices: list,
                             device: torch.device,
                             fbar_lookback: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:


    if fbar_lookback < 1:
        raise ValueError("fbar_lookback must be positive")
    encoder.eval()
    vae.eval()
    beta_network.eval()

    T_dataset = len(dataset)
    N = dataset.N


    all_f = []
    all_beta = []

    for idx in range(T_dataset):
        batch = dataset[idx]
        hist_features = batch["hist_features"].unsqueeze(0).to(device)
        returns = batch["returns"].unsqueeze(0).to(device)

        F_t = vae.get_factor_returns(returns, use_mean=True)


        Z = encoder(hist_features)


        beta_Z = beta_network(Z)

        all_f.append(F_t.cpu().numpy())
        all_beta.append(beta_Z.cpu().numpy())


    all_f = np.concatenate(all_f, axis=0)
    all_beta = np.concatenate(all_beta, axis=0)
    if not np.isfinite(all_f).all() or not np.isfinite(all_beta).all():
        raise FloatingPointError("Nonfinite factor or loading estimates")


    mu_total_list = []
    mu_pred_list = []
    returns_list = []


    original_to_dataset_idx = {orig_idx: idx for idx, orig_idx in enumerate(dataset.valid_indices)}

    for t_original in test_indices:

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


        returns_list.append(dataset.returns[t_original].numpy())

    if not mu_pred_list:
        raise ValueError("No forecast periods have the required observable history")

    return (
        np.array(mu_total_list),
        np.array(mu_pred_list),
        np.array(returns_list)
    )


def main(args):
    device = get_safe_device(args.cuda)


    config_path = os.path.join(args.model_dir, "config.json")
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


    feats, rets = train.load_data(args.features_path, args.returns_path, args.layout, args.return_type)
    validate_input_contract(config_dict, feats, rets, args.layout, args.return_type)
    T, N, P = feats.shape


    dataset = train.StockDataset(feats, rets, window=window)


    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)


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


    model_path = os.path.join(args.model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder'])
    vae.load_state_dict(checkpoint['vae'])
    beta_network.load_state_dict(checkpoint['beta_network'])


    print(f"Predicting {len(test_indices)} periods on {device}", flush=True)
    mu_total, mu_pred, returns_true = compute_test_predictions(
        encoder,
        vae,
        beta_network,
        dataset,
        test_indices,
        device,
        fbar_lookback=int(args.fbar_lookback),
    )


    r2_total = compute_r2(mu_total, returns_true)
    r2_pred = compute_r2(mu_pred, returns_true)


    print(f"R2_total={r2_total:.6f}%, R2_pred={r2_pred:.6f}%", flush=True)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, "mu_total.npy"), mu_total)
        np.save(os.path.join(args.output_dir, "mu_pred.npy"), mu_pred)
        np.save(os.path.join(args.output_dir, "returns_true.npy"), returns_true)

        np.save(os.path.join(args.output_dir, "prediction_indices.npy"), np.array(test_indices, dtype=int))
        metrics = {
            "R2_total": r2_total if np.isfinite(r2_total) else None,
            "R2_pred": r2_pred if np.isfinite(r2_pred) else None,
            "return_type": args.return_type,
            "R2_convention": "pooled uncentered R2 in percent; per-period statistics in evaluate.py",
            "fbar_lookback": int(max(1, int(args.fbar_lookback))),
        }
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASH模型测试")


    parser.add_argument("--model-dir", type=str, required=True,
                        help="模型保存目录（包含model.pt和config.json）")


    parser.add_argument("--layout", choices=["NTP", "TNP"], default="NTP")
    parser.add_argument("--return-type", choices=["simple", "excess"], default="excess")
    parser.add_argument("--features-path", type=str, default="price_and_factor.npy",
                        help="特征文件路径")
    parser.add_argument("--returns-path", type=str, default="return.npy",
                        help="收益文件路径")


    parser.add_argument("--output-dir", type=str, default=None,
                        help="结果保存目录（可选）")


    parser.add_argument("--cuda", type=int, default=0,
                        help="GPU编号")
    parser.add_argument("--fbar-lookback", type=int, default=3,
                        help="计算 mu_pred 时 f_bar 的回看期数")

    args = parser.parse_args()
    main(args)
