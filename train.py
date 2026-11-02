import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
import copy
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple


from itransformer_encoder import TemporalCrossSectionEncoder, NoITransformerEncoder
from factor_network import VAE_Network
from beta_network import BetaNetwork, compute_factor_pricing_returns, compute_prediction_loss
from sample_shuffling import ShuffleModule
from perturbation_generator import CaRIIntervention
from adversary_network import Discriminator, compute_discriminator_loss
from cash_model import Generator, compute_generator_loss
from runtime import cross_sectional_ic


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


def enable_runtime_acceleration(device: torch.device) -> None:

    if device.type != "cuda":
        return
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def resolve_num_workers(num_workers: int) -> int:
    if num_workers >= 0:
        return num_workers
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def resolve_amp_dtype(amp_mode: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    mode = amp_mode.lower()
    if mode == "off":
        return None
    if mode == "fp16":
        return torch.float16
    if mode in {"auto", "bf16"}:
        with torch.cuda.device(device):
            supports_bf16 = torch.cuda.is_bf16_supported()
        if supports_bf16:
            return torch.bfloat16
        if mode == "auto":
            return torch.float16
        raise ValueError("BF16 is not supported on the selected CUDA device")
    raise ValueError(f"未知amp模式: {amp_mode}")


def maybe_autocast(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def load_data(features_path: str, returns_path: str, layout: str = "NTP",
              return_type: str = "excess") -> Tuple[np.ndarray, np.ndarray]:
    if layout not in {"NTP", "TNP"}:
        raise ValueError("layout must be NTP or TNP")
    if return_type not in {"excess", "simple"}:
        raise ValueError("return_type must be excess or simple")
    feats = np.load(features_path, allow_pickle=False)
    rets = np.load(returns_path, allow_pickle=False)
    if feats.ndim != 3 or rets.ndim != 2:
        raise ValueError("features must have 3 axes and returns must have 2 axes")
    if feats.shape[:2] != rets.shape:
        raise ValueError("features and returns must use the same time/asset axes")
    if (not np.issubdtype(feats.dtype, np.number) or not np.issubdtype(rets.dtype, np.number)
            or np.iscomplexobj(feats) or np.iscomplexobj(rets)):
        raise ValueError("inputs must be numeric arrays")
    if layout == "NTP":
        feats = feats.transpose(1, 0, 2)
        rets = rets.transpose(1, 0)
    feats = np.asarray(feats, dtype=np.float32)
    rets = np.asarray(rets, dtype=np.float32)
    if min(feats.shape) < 1 or not np.isfinite(feats).all() or not np.isfinite(rets).all():
        raise ValueError("inputs must be nonempty, finite, externally prepared arrays")
    if return_type == "simple" and np.any(rets < -1):
        raise ValueError("simple returns must be decimal fractions and at least -1")
    return feats, rets


class StockDataset(Dataset):


    def __init__(self, features: np.ndarray, returns: np.ndarray, window: int = 20):


        if features.ndim != 3 or returns.ndim != 2 or features.shape[:2] != returns.shape:
            raise ValueError("Expected aligned (T, N, P) features and (T, N) returns")
        if not isinstance(window, int) or window < 1 or window >= features.shape[0]:
            raise ValueError("window must be a positive integer shorter than the panel")
        self.features = torch.tensor(features, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.window = window

        self.T, self.N, self.P = features.shape


        self.valid_indices = list(range(window, self.T))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]


        hist_features = self.features[t - self.window:t]


        current_return = self.returns[t]


        current_features = self.features[t - 1]

        return {
            "hist_returns": self.returns[t - self.window:t],
            "hist_features": hist_features,
            "current_features": current_features,
            "returns": current_return,
            "idx": t
        }


@dataclass
class TrainConfig:


    window: int = 20


    n_assets: int = 0
    layout: str = "NTP"
    return_type: str = "excess"
    input_shape: Optional[Tuple[int, int, int]] = None
    disable_adversarial: bool = False
    feature_dim: int = 318
    z_dim: int = 64
    factor_dim: int = 8


    transformer_nhead: int = 4
    transformer_layers_time: int = 2
    transformer_layers_cross: int = 2
    transformer_dropout: float = 0.1
    disable_itransformer: bool = False
    noitr_pooling: str = "last"


    vae_hidden_dim: int = 32
    vae_beta: float = 1.0


    beta_hidden_dim: int = None


    cari_noise_std: float = 0.1


    d1_hidden_dim: int = 64
    d1_use_inner_product: bool = False


    batch_size: int = 32
    epochs: int = 200
    lr_vae: float = 1e-3
    lr_generator: float = 1e-3
    lr_discriminator: float = 1e-3
    weight_decay: float = 1e-4


    lambda_vae: float = 1.0
    lambda_d1: float = 1.0
    lambda_d2: float = 1.0
    lambda_g1: float = 1.0
    lambda_g2: float = 1.0
    lambda_g3: float = 1.0
    lambda_g4: float = 0.1
    g3_mse_weight: float = 1.0
    g3_ic_weight: float = 0.0


    grl_d1: float = 1.0
    grl_d2: float = 1.0


    patience: int = 30


    train_ratio: float = 0.7
    val_ratio: float = 0.1
    val_metric: str = "mse"


@contextmanager
def frozen_parameters(*modules):
    parameters = list(dict.fromkeys(p for module in modules for p in module.parameters()))
    previous = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, enabled in zip(parameters, previous):
            p.requires_grad_(enabled)


def backward_step(loss, optimizer_names, optimizers, parameters, scaler):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training loss")
    use_scaler = scaler is not None and scaler.is_enabled()
    if use_scaler:
        scaler.scale(loss).backward()
        for name in optimizer_names:
            scaler.unscale_(optimizers[name])
    else:
        loss.backward()
    for group in parameters:
        torch.nn.utils.clip_grad_norm_(list(group), 1.0, error_if_nonfinite=not use_scaler)
    for name in optimizer_names:
        if use_scaler:
            scaler.step(optimizers[name])
        else:
            optimizers[name].step()


def train_one_epoch(vae: VAE_Network, encoder: TemporalCrossSectionEncoder,
                    beta_network: BetaNetwork, shuffle_module: ShuffleModule,
                    cari_module: CaRIIntervention, discriminator: Discriminator,
                    generator: Generator,
                    loader: DataLoader, optimizers: Dict, config: TrainConfig,
                    device: torch.device, epoch: int,
                    amp_dtype: Optional[torch.dtype] = None,
                    scaler: Optional[torch.amp.GradScaler] = None) -> Dict[str, float]:
    vae.train()
    encoder.train()
    beta_network.train()
    shuffle_module.train()
    cari_module.train()
    discriminator.train()
    generator.train()
    losses = {key: [] for key in ("vae", "d1", "d2", "g1", "g2", "g3", "g4", "perturbation", "total")}
    sample_counts = []

    for batch_idx, batch in enumerate(loader):
        hist_features = batch["hist_features"].to(device, non_blocking=True)
        current_features = batch["current_features"].to(device, non_blocking=True)
        returns = batch["returns"].to(device, non_blocking=True)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), maybe_autocast(device, amp_dtype):
            Z = encoder(hist_features)
            Z_shuffled = shuffle_module(Z)
            Z_bar = cari_module(Z)
        with maybe_autocast(device, amp_dtype):
            disc_losses = compute_discriminator_loss(
                discriminator, beta_network, vae, current_features,
                Z, Z_shuffled, Z_bar, returns,
                lambda_d1=config.lambda_d1, lambda_d2=config.lambda_d2,
            )
        if config.lambda_d1 > 0 or config.lambda_d2 > 0:
            backward_step(disc_losses["loss_total"], ["discriminator"], optimizers,
                          [discriminator.parameters()], scaler)

        perturbation_loss = returns.new_zeros(())
        if config.lambda_g2 > 0 or config.lambda_g4 > 0:
            with frozen_parameters(discriminator), maybe_autocast(device, amp_dtype):
                perturbed = cari_module(Z.detach())
                prediction_error = (compute_prediction_loss(discriminator.d2(perturbed), returns)
                                    if config.lambda_g2 > 0 else returns.new_zeros(()))
                norm_penalty = cari_module.regularization(Z.detach())
                perturbation_loss = -config.lambda_g2 * prediction_error + config.lambda_g4 * norm_penalty
                backward_step(perturbation_loss, ["perturbation"], optimizers,
                              [cari_module.parameters()], scaler)

        optimizers["generator"].zero_grad(set_to_none=True)
        optimizers["vae"].zero_grad(set_to_none=True)
        with frozen_parameters(discriminator, cari_module), maybe_autocast(device, amp_dtype):
            x_hat, mean, log_var = vae(returns)
            loss_vae = vae.loss_function(returns, x_hat, mean, log_var)
            gen_losses = compute_generator_loss(
                generator, discriminator, beta_network, vae, shuffle_module,
                hist_features, returns,
                lambda_g1=config.lambda_g1, lambda_g2=config.lambda_g2,
                lambda_g3=config.lambda_g3, lambda_g4=config.lambda_g4,
                lambda_grl_d1=config.grl_d1, lambda_grl_d2=config.grl_d2,
                g3_mse_weight=config.g3_mse_weight, g3_ic_weight=config.g3_ic_weight,
                perturb_target_norm=config.cari_noise_std,
            )
            main_loss = config.lambda_vae * loss_vae + gen_losses["loss_total"]
            backward_step(main_loss, ["generator", "vae"], optimizers,
                          [encoder.parameters(), beta_network.parameters(), vae.parameters()], scaler)
        if scaler is not None and scaler.is_enabled():
            scaler.update()

        losses["vae"].append(loss_vae.item())
        losses["d1"].append(disc_losses["loss_d1"].item())
        losses["d2"].append(disc_losses["loss_d2"].item())
        for number in range(1, 5):
            losses[f"g{number}"].append(gen_losses[f"loss_g{number}"].item())
        losses["perturbation"].append(perturbation_loss.item())
        losses["total"].append((main_loss.detach() + disc_losses["loss_total"].detach()).item())
        sample_counts.append(returns.shape[0])
    if not sample_counts:
        raise ValueError("training loader is empty")
    return {key: float(np.average(values, weights=sample_counts)) for key, values in losses.items()}


@torch.no_grad()
def validate(vae: VAE_Network, encoder: TemporalCrossSectionEncoder,
             beta_network: BetaNetwork, loader: DataLoader,
             config: TrainConfig, device: torch.device,
             debug: bool = False,
             amp_dtype: Optional[torch.dtype] = None) -> Dict[str, float]:

    vae.eval()
    encoder.eval()
    beta_network.eval()

    from beta_network import compute_factor_pricing_returns

    mse_list = []
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(loader):
        hist_features = batch["hist_features"].to(device, non_blocking=True)
        current_features = batch["current_features"].to(device, non_blocking=True)
        returns = batch["returns"].to(device, non_blocking=True)

        historical_returns = batch["hist_returns"].to(device, non_blocking=True)
        with maybe_autocast(device, amp_dtype):
            Z = encoder(hist_features)
            F_t = vae.get_factor_returns(historical_returns.flatten(0, 1), use_mean=True).reshape(returns.shape[0], config.window, config.factor_dim).mean(dim=1)
            beta_Z = beta_network(Z)
            y_pred = compute_factor_pricing_returns(beta_Z, F_t)
            mse = compute_prediction_loss(y_pred, returns)

        if not torch.isfinite(mse):
            raise FloatingPointError("Nonfinite validation loss")
        mse_list.append((mse.item(), returns.shape[0]))
        all_preds.append(y_pred.float().cpu().numpy())
        all_targets.append(returns.float().cpu().numpy())

    if len(mse_list) == 0:
        return {
            "objective": float("inf"),
            "mse": float("nan"),
            "ic": float("nan"),
        }


    mean_mse = float(np.average([v for v, _ in mse_list], weights=[n for _, n in mse_list]))
    ic_list = []
    for pred_batch, target_batch in zip(all_preds, all_targets):
        for pred_row, target_row in zip(pred_batch, target_batch):
            corr = cross_sectional_ic(pred_row, target_row)
            if np.isfinite(corr):
                ic_list.append(corr)
    mean_ic = float(np.mean(ic_list)) if ic_list else float("nan")

    if config.val_metric == "ic":
        objective = -mean_ic if np.isfinite(mean_ic) else float("inf")
    else:
        objective = mean_mse if np.isfinite(mean_mse) else float("inf")

    return {
        "objective": float(objective),
        "mse": mean_mse,
        "ic": mean_ic,
    }


def main(args):
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("invalid chronological split ratios")
    if args.epochs < 1 or args.window < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, window, batch_size and patience must be positive")
    if args.disable_adversarial:
        for name in ("lambda_d1", "lambda_d2", "lambda_g1", "lambda_g2", "lambda_g4"):
            setattr(args, name, 0.0)
    nonnegative = ("lambda_vae", "lambda_d1", "lambda_d2", "lambda_g1", "lambda_g2", "lambda_g3", "lambda_g4", "g3_mse_weight", "g3_ic_weight", "grl_d1", "grl_d2", "vae_beta", "weight_decay")
    if any(not np.isfinite(getattr(args, name)) or getattr(args, name) < 0 for name in nonnegative):
        raise ValueError("loss weights, KL weight, weight decay and reversal strengths must be finite and nonnegative")
    if any(not 0 < getattr(args, name) < float("inf") for name in ("lr_vae", "lr_generator", "lr_discriminator", "cari_noise_std")):
        raise ValueError("learning rates and perturbation target norm must be finite and positive")
    if os.path.exists(os.path.join(args.output_dir, "model.pt")):
        raise FileExistsError("output directory already contains model.pt")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_safe_device(args.cuda)
    enable_runtime_acceleration(device)
    amp_dtype = resolve_amp_dtype(args.amp_mode, device)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)


    feats, rets = load_data(args.features_path, args.returns_path, args.layout, args.return_type)
    T, N, P = feats.shape


    dataset = StockDataset(feats, rets, window=args.window)


    train_end = int(T * args.train_ratio)
    val_end = train_end + int(T * args.val_ratio)
    original_to_dataset_idx = {orig_idx: idx for idx, orig_idx in enumerate(dataset.valid_indices)}

    train_original = list(range(args.window, train_end))
    val_original = list(range(train_end, val_end))
    test_original = list(range(val_end, T))

    train_indices = [original_to_dataset_idx[idx] for idx in train_original if idx in original_to_dataset_idx]
    val_indices = [original_to_dataset_idx[idx] for idx in val_original if idx in original_to_dataset_idx]
    test_indices = [original_to_dataset_idx[idx] for idx in test_original if idx in original_to_dataset_idx]

    if not train_indices or not val_indices or not test_indices:
        raise ValueError("each chronological split must contain a target observation")
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)

    num_workers = resolve_num_workers(args.num_workers)
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)


    config = TrainConfig(
        n_assets=N, layout=args.layout, return_type=args.return_type, input_shape=(T, N, P),
        disable_adversarial=args.disable_adversarial,
        feature_dim=P,
        z_dim=args.z_dim,
        factor_dim=args.factor_dim,
        window=args.window,
        transformer_nhead=args.transformer_nhead,
        transformer_layers_time=args.transformer_layers_time,
        transformer_layers_cross=args.transformer_layers_cross,
        transformer_dropout=args.transformer_dropout,
        disable_itransformer=bool(args.disable_itransformer or args.disable_adversarial),
        noitr_pooling=str(args.noitr_pooling),
        vae_hidden_dim=args.vae_hidden_dim,
        vae_beta=args.vae_beta,
        beta_hidden_dim=args.beta_hidden_dim if args.beta_hidden_dim > 0 else None,
        cari_noise_std=args.cari_noise_std,
        d1_hidden_dim=args.d1_hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_vae=args.lr_vae,
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        weight_decay=args.weight_decay,
        lambda_vae=args.lambda_vae,
        lambda_d1=args.lambda_d1,
        lambda_d2=args.lambda_d2,
        lambda_g1=args.lambda_g1,
        lambda_g2=args.lambda_g2,
        lambda_g3=args.lambda_g3,
        lambda_g4=args.lambda_g4,
        g3_mse_weight=args.g3_mse_weight,
        g3_ic_weight=args.g3_ic_weight,
        grl_d1=args.grl_d1,
        grl_d2=args.grl_d2,
        patience=args.patience,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        val_metric=args.val_metric,
    )


    if config.disable_itransformer:
        encoder = NoITransformerEncoder(
            input_dim=P,
            z_dim=args.z_dim,
            pooling=args.noitr_pooling,
            window=args.window, nhead=args.transformer_nhead,
            num_layers=args.transformer_layers_time, dropout=args.transformer_dropout,
        ).to(device)
    else:
        encoder = TemporalCrossSectionEncoder(
            input_dim=P,
            z_dim=args.z_dim,
            nhead=args.transformer_nhead,
            num_layers_time=args.transformer_layers_time,
            num_layers_cross=args.transformer_layers_cross,
            dropout=args.transformer_dropout, window=args.window,
        ).to(device)


    print(f"Training on {device}: T={T}, N={N}, P={P}, epochs={args.epochs}", flush=True)

    vae = VAE_Network(
        input_dim=N,
        latent_dim=args.factor_dim,
        output_dim=N,
        beta=args.vae_beta, hidden_dim=args.vae_hidden_dim
    ).to(device)


    beta_network = BetaNetwork(
        z_dim=args.z_dim,
        factor_dim=args.factor_dim,
        hidden_dim=args.beta_hidden_dim if args.beta_hidden_dim > 0 else args.z_dim
    ).to(device)


    shuffle_module = ShuffleModule().to(device)


    cari_module = CaRIIntervention(noise_std=args.cari_noise_std, z_dim=args.z_dim).to(device)


    discriminator = Discriminator(
        feature_dim=P,
        z_dim=args.z_dim,
        d1_hidden_dim=args.d1_hidden_dim,
        d1_use_inner_product=config.d1_use_inner_product
    ).to(device)


    generator = Generator(
        encoder=encoder,
        cari_module=cari_module,
        z_dim=args.z_dim
    ).to(device)


    optimizers = {
        'vae': optim.AdamW(vae.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay),
        'generator': optim.AdamW(
            list(encoder.parameters()) + list(beta_network.parameters()),
            lr=args.lr_generator, weight_decay=args.weight_decay
        ),
        'perturbation': optim.AdamW(cari_module.parameters(), lr=args.lr_generator, weight_decay=args.weight_decay),
        'discriminator': optim.AdamW(discriminator.parameters(), lr=args.lr_discriminator, weight_decay=args.weight_decay)
    }


    schedulers = {
        'vae': optim.lr_scheduler.CosineAnnealingLR(optimizers['vae'], T_max=args.epochs, eta_min=1e-6),
        'generator': optim.lr_scheduler.CosineAnnealingLR(optimizers['generator'], T_max=args.epochs, eta_min=1e-6),
        'perturbation': optim.lr_scheduler.CosineAnnealingLR(optimizers['perturbation'], T_max=args.epochs, eta_min=1e-6),
        'discriminator': optim.lr_scheduler.CosineAnnealingLR(optimizers['discriminator'], T_max=args.epochs, eta_min=1e-6)
    }
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))


    os.makedirs(args.output_dir, exist_ok=True)

    best_val_objective = float("inf")
    best_val_mse = float("inf")
    best_val_ic = float("-inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    improved = False

    for epoch in range(1, args.epochs + 1):

        train_losses = train_one_epoch(
            vae, encoder, beta_network, shuffle_module, cari_module,
            discriminator, generator, train_loader, optimizers, config, device, epoch,
            amp_dtype=amp_dtype, scaler=scaler
        )


        val_stats = validate(vae, encoder, beta_network, val_loader, config, device, amp_dtype=amp_dtype)
        val_objective = float(val_stats["objective"])
        val_mse = float(val_stats["mse"])
        val_ic = float(val_stats["ic"])


        for scheduler in schedulers.values():
            scheduler.step()


        if val_objective < best_val_objective:
            best_val_objective = val_objective
            best_val_mse = val_mse
            best_val_ic = val_ic
            best_state = copy.deepcopy({
                'vae': vae.state_dict(),
                'encoder': encoder.state_dict(),
                'beta_network': beta_network.state_dict(),
                'discriminator': discriminator.state_dict(),
                'generator': generator.state_dict(),
                'cari_module': cari_module.state_dict(),
            })
            best_epoch = epoch
            no_improve = 0
            improved = True
        else:
            no_improve += 1
            improved = False


        if epoch % 10 == 0 or improved or epoch <= 5:
            print(f"[Epoch {epoch:3d}/{args.epochs}] "
                  f"vae={train_losses['vae']:.4f} "
                  f"d1={train_losses['d1']:.4f} "
                  f"d2={train_losses['d2']:.4f} "
                  f"g1={train_losses['g1']:.4f} "
                  f"g2={train_losses['g2']:.4f} "
                  f"g3={train_losses['g3']:.4f} "
                  f"g4={train_losses['g4']:.4f} "
                  f"total={train_losses['total']:.4f} "
                  f"val_obj={val_objective:.4f} "
                  f"val_mse={val_mse:.4f} "
                  f"val_ic={val_ic:.4f}"
                  + (" *" if improved else ""))


        if no_improve >= args.patience:
            print(f"\n早停: {args.patience} epochs 无改进")
            break


    if best_state is not None:
        torch.save(best_state, os.path.join(args.output_dir, "model.pt"))

    if best_state is None:
        raise FloatingPointError("no finite validation checkpoint was obtained")

    print(f"Saved best epoch {best_epoch} to {args.output_dir}", flush=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)


    train_summary = {
        "best_epoch": int(best_epoch),
        "best_val_objective": float(best_val_objective) if np.isfinite(best_val_objective) else None,
        "best_val_mse": float(best_val_mse) if np.isfinite(best_val_mse) else None,
        "best_val_ic": float(best_val_ic) if np.isfinite(best_val_ic) else None,
        "completed_epochs": int(epoch),
        "early_stopped": bool(no_improve >= args.patience),
        "seed": int(args.seed),
        "window": int(args.window),
        "z_dim": int(args.z_dim),
        "factor_dim": int(args.factor_dim),
        "transformer_dropout": float(args.transformer_dropout),
        "vae_hidden_dim": int(args.vae_hidden_dim),
        "vae_beta": float(args.vae_beta),
        "beta_hidden_dim": int(args.beta_hidden_dim),
        "cari_noise_std": float(args.cari_noise_std),
        "d1_hidden_dim": int(args.d1_hidden_dim),
        "batch_size": int(args.batch_size),
        "num_workers": int(num_workers),
        "epochs": int(args.epochs),
        "amp_mode": str(args.amp_mode),
        "lr_vae": float(args.lr_vae),
        "lr_generator": float(args.lr_generator),
        "lr_discriminator": float(args.lr_discriminator),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "lambda_vae": float(args.lambda_vae),
        "lambda_d1": float(args.lambda_d1),
        "lambda_d2": float(args.lambda_d2),
        "lambda_g1": float(args.lambda_g1),
        "lambda_g2": float(args.lambda_g2),
        "lambda_g3": float(args.lambda_g3),
        "lambda_g4": float(args.lambda_g4),
        "g3_mse_weight": float(args.g3_mse_weight),
        "g3_ic_weight": float(args.g3_ic_weight),
        "grl_d1": float(args.grl_d1),
        "grl_d2": float(args.grl_d2),
        "val_metric": str(args.val_metric),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "disable_itransformer": bool(config.disable_itransformer),
        "disable_adversarial": bool(config.disable_adversarial),
        "noitr_pooling": str(args.noitr_pooling),
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
        json.dump(train_summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASH 股票收益率预测模型训练")


    parser.add_argument("--features-path", type=str, default="price_and_factor.npy",
                        help="特征文件路径")
    parser.add_argument("--returns-path", type=str, default="return.npy",
                        help="收益文件路径")
    parser.add_argument("--output-dir", type=str, default="../CASH_outputs/cash",
                        help="输出目录")


    parser.add_argument("--layout", choices=["NTP", "TNP"], default="NTP")
    parser.add_argument("--return-type", choices=["excess", "simple"], default="excess")
    parser.add_argument("--disable-adversarial", action="store_true")
    parser.add_argument("--window", type=int, default=5,
                        help="历史窗口大小")


    parser.add_argument("--z-dim", type=int, default=32,
                        help="嵌入维度")
    parser.add_argument("--factor-dim", type=int, default=8,
                        help="因子维度K")


    parser.add_argument("--transformer-nhead", type=int, default=4,
                        help="注意力头数")
    parser.add_argument("--transformer-layers-time", type=int, default=2,
                        help="主干 iTransformer 层数（vanilla消融中为时间层数）")
    parser.add_argument("--transformer-layers-cross", type=int, default=2,
                        help="保留原命令兼容；正文主干不使用跨资产注意力")
    parser.add_argument("--transformer-dropout", type=float, default=0.1,
                        help="Dropout比例")
    parser.add_argument("--disable-itransformer", action="store_true",
                        help="关闭iTransformer编码器，使用vanilla Transformer编码器（消融）")
    parser.add_argument("--noitr-pooling", type=str, default="last", choices=["last", "mean"],
                        help="无iTransformer时的时间聚合方式")


    parser.add_argument("--vae-hidden-dim", type=int, default=32,
                        help="VAE隐藏层维度")
    parser.add_argument("--vae-beta", type=float, default=1.0,
                        help="VAE KL散度权重")


    parser.add_argument("--beta-hidden-dim", type=int, default=0,
                        help="Beta网络隐藏层维度（0表示使用z_dim）")


    parser.add_argument("--cari-noise-std", type=float, default=0.1,
                        help="扰动目标范数 b（保留原参数名）")


    parser.add_argument("--d1-hidden-dim", type=int, default=64,
                        help="D1隐藏层维度")


    parser.add_argument("--batch-size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader worker数，-1表示自动选择")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--lr-vae", type=float, default=1e-3,
                        help="VAE学习率")
    parser.add_argument("--lr-generator", type=float, default=1e-3,
                        help="生成器学习率")
    parser.add_argument("--lr-discriminator", type=float, default=1e-3,
                        help="判别器学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值")


    parser.add_argument("--lambda-vae", type=float, default=1.0,
                        help="VAE损失权重")
    parser.add_argument("--lambda-d1", type=float, default=1.0,
                        help="D1损失权重")
    parser.add_argument("--lambda-d2", type=float, default=1.0,
                        help="D2损失权重")
    parser.add_argument("--lambda-g1", type=float, default=1.0,
                        help="G1损失权重")
    parser.add_argument("--lambda-g2", type=float, default=1.0,
                        help="G2损失权重")
    parser.add_argument("--lambda-g3", type=float, default=1.0,
                        help="G3损失权重")
    parser.add_argument("--lambda-g4", type=float, default=0.1,
                        help="G4损失权重")
    parser.add_argument("--g3-mse-weight", type=float, default=1.0,
                        help="G3主任务中MSE项权重")
    parser.add_argument("--g3-ic-weight", type=float, default=0.0,
                        help="G3主任务中IC对齐项权重")


    parser.add_argument("--grl-d1", type=float, default=1.0,
                        help="D1梯度反转强度")
    parser.add_argument("--grl-d2", type=float, default=1.0,
                        help="D2梯度反转强度")


    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--val-metric", type=str, default="mse", choices=["mse", "ic"],
                        help="早停指标：mse或ic")


    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--cuda", type=int, default=0,
                        help="GPU编号")
    parser.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "off", "bf16", "fp16"],
                        help="混合精度模式")

    args = parser.parse_args()
    main(args)
