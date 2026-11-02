import argparse
import json
import random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from cash_model import CASH, ModelConfig
from classification_adversary import dependency_loss
from cash_model import frozen, adversarial_input
from runtime import load_inputs, split_indices, device_for
from beta_network import pricing_returns


def make_optimizers(model, args):

    main = torch.optim.AdamW(
        [
            {
                "params": list(model.encoder.parameters())
                + list(model.beta.parameters()),
                "lr": args.lr_generator,
            },
            {"params": model.factor.parameters(), "lr": args.lr_vae},
        ],
        weight_decay=args.weight_decay,
    )
    heads = torch.optim.AdamW(
        list(model.dcls.parameters()) + list(model.dpred.parameters()),
        lr=args.lr_discriminator,
        weight_decay=args.weight_decay,
    )
    perturb = torch.optim.AdamW(
        model.perturb.parameters(), lr=args.lr_generator, weight_decay=args.weight_decay
    )
    return {"main": main, "heads": heads, "perturb": perturb}


def step(loss, optimizer):
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Nonfinite optimization objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()


def train_batch(model, history, returns, optimizers, args):
    model.train()
    active = not model.config.disable_adversarial
    cls_on = active and (args.lambda_d1 != 0 or args.lambda_g1 != 0)
    pred_on = active and (args.lambda_d2 != 0 or args.lambda_g2 != 0)
    stats = {}

    with torch.no_grad():
        z = model.encoder(history)
        shifted = model.perturb(z)
    x = model.covariates(history)
    head_loss = None
    if cls_on and args.lambda_d1 != 0:
        value = dependency_loss(model.dcls, x, z)
        head_loss = args.lambda_d1 * value
        stats["dcls"] = value.item()
    if pred_on and args.lambda_d2 != 0:
        value = F.mse_loss(model.dpred(shifted), returns)
        head_loss = (
            args.lambda_d2 * value
            if head_loss is None
            else head_loss + args.lambda_d2 * value
        )
        stats["dpred"] = value.item()
    if head_loss is not None:
        step(head_loss, optimizers["heads"])

    if pred_on:
        with frozen(model.dpred):
            shifted_loss = F.mse_loss(model.dpred(model.perturb(z)), returns)
            norm_loss = model.perturb.regularization(z)
            perturb_loss = -args.lambda_g2 * shifted_loss + args.lambda_g4 * norm_loss
            step(perturb_loss, optimizers["perturb"])
            stats.update(
                perturb_objective=perturb_loss.item(), norm_regularizer=norm_loss.item()
            )

    with frozen(model.dcls, model.dpred, model.perturb):
        z = model.encoder(history)
        factors = model.factor.factors(returns)
        task = F.mse_loss(pricing_returns(model.beta(z), factors), returns)
        vae = model.factor.loss(returns, args.vae_beta)
        total = args.lambda_g3 * task + args.lambda_vae * vae
        grl = not model.config.disable_grl
        if cls_on and args.lambda_g1 != 0:

            total = total + args.lambda_g1 * dependency_loss(
                model.dcls, x, adversarial_input(z, grl)
            )
        if pred_on and args.lambda_g2 != 0:
            shifted = adversarial_input(model.perturb(z), grl)
            total = total + args.lambda_g2 * F.mse_loss(model.dpred(shifted), returns)
        if pred_on:
            total = total + args.lambda_g4 * model.perturb.regularization(z)
        step(total, optimizers["main"])
    stats.update(task=task.item(), vae=vae.item(), main=total.item())
    if not all(np.isfinite(v) for v in stats.values()):
        raise FloatingPointError("Nonfinite loss; inspect inputs and training settings")
    return stats


@torch.no_grad()
def validate(model, x, r, indices, batch_size, device):
    model.eval()
    predictions, targets = [], []
    for start in range(0, len(indices), batch_size):
        ts = indices[start : start + batch_size]
        histories = torch.stack([x[t - model.config.window : t] for t in ts]).to(device)
        targets.append(r[ts].numpy())
        predictions.append(model(histories, r[ts].to(device)).cpu().numpy())
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite validation predictions")
    from evaluate import cross_section_correlations

    ic = cross_section_correlations(pred, target)
    finite = ic[np.isfinite(ic)]
    return {
        "mse": float(np.mean((pred.astype(float) - target) ** 2)),
        "ic": float(finite.mean()) if len(finite) else None,
    }


def build_parser():
    p = argparse.ArgumentParser(description="CASH")
    p.add_argument("--features-path", required=True)
    p.add_argument("--returns-path", required=True)
    p.add_argument("--layout", choices=["NTP", "TNP"], default="NTP")
    p.add_argument("--return-type", choices=["simple", "excess"], default="excess")
    p.add_argument("--output-dir", default="../CASH_outputs/model")
    for name, default in [
        ("window", 3),
        ("z-dim", 32),
        ("factor-dim", 8),
        ("transformer-nhead", 4),
        ("transformer-layers-time", 2),
        ("vae-hidden-dim", 32),
        ("beta-hidden-dim", 0),
        ("d1-hidden-dim", 64),
        ("batch-size", 8),
        ("epochs", 200),
        ("patience", 50),
        ("seed", 42),
        ("cuda", 0),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    for name, default in [
        ("transformer-dropout", 0.1),
        ("perturb-target-norm", 0.1),
        ("vae-beta", 1.0),
        ("lr-vae", 1e-3),
        ("lr-generator", 1e-3),
        ("lr-discriminator", 1e-3),
        ("weight-decay", 1e-4),
        ("lambda-vae", 1.0),
        ("lambda-d1", 1.0),
        ("lambda-d2", 1.0),
        ("lambda-g1", 1.0),
        ("lambda-g2", 1.0),
        ("lambda-g3", 1.0),
        ("lambda-g4", 0.1),
        ("train-ratio", 0.7),
        ("val-ratio", 0.1),
    ]:
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--val-metric", choices=["mse", "ic"], default="mse")
    p.add_argument("--disable-itransformer", action="store_true")
    p.add_argument("--disable-adversarial", action="store_true")
    p.add_argument("--disable-grl", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    for name, value in vars(args).items():
        if name.startswith(("lambda_", "lr_")) or name in ("weight_decay", "vae_beta"):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
    if min(args.lr_vae, args.lr_generator, args.lr_discriminator) <= 0:
        raise ValueError("Learning rates must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    x, r = load_inputs(
        args.features_path, args.returns_path, args.layout, args.return_type
    )
    indices = split_indices(len(x), args.window, args.train_ratio, args.val_ratio)
    fields = ModelConfig.__dataclass_fields__
    kwargs = {k: getattr(args, k) for k in fields if hasattr(args, k)}
    config = ModelConfig(n_assets=x.shape[1], feature_dim=x.shape[2], **kwargs)
    device = device_for(args.cuda)
    model = CASH(config).to(device)
    optimizers = make_optimizers(model, args)
    schedulers = {
        key: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=1e-6
        )
        for key, opt in optimizers.items()
    }
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "best.pt").exists():
        raise FileExistsError(
            "Checkpoint already exists; choose a new output directory"
        )
    training = vars(args).copy()

    for k in ["features_path", "returns_path", "output_dir"]:
        training.pop(k)
    metadata = {
        "implementation": "body-first-v2",
        "model": asdict(config),
        "training": training,
        "input_shape": list(x.shape),
        "target": (
            "excess_decimal_return"
            if args.return_type == "excess"
            else "raw_decimal_simple_return"
        ),
    }
    (folder / "config.json").write_text(json.dumps(metadata, indent=2))
    best, stale, log = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(indices["train"]).tolist()
        batches = []
        for start in range(0, len(order), args.batch_size):
            ts = order[start : start + args.batch_size]
            history = torch.stack([x[t - args.window : t] for t in ts]).to(device)
            batches.append(
                train_batch(model, history, r[ts].to(device), optimizers, args)
            )
        val = validate(model, x, r, indices["val"], args.batch_size, device)
        objective = (
            val["mse"]
            if args.val_metric == "mse"
            else (-val["ic"] if val["ic"] is not None else float("inf"))
        )
        row = {
            "epoch": epoch,
            **{
                k: float(np.mean([b[k] for b in batches if k in b]))
                for k in set().union(*batches)
            },
            "val_mse": val["mse"],
            "val_ic": val["ic"],
        }
        log.append(row)
        print(json.dumps(row), flush=True)
        if np.isfinite(objective) and objective < best:
            best, stale = objective, 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch}, folder / "best.pt"
            )
        else:
            stale += 1
        for key, scheduler in schedulers.items():
            if (
                key == "main"
                or (
                    key == "heads" and any("dcls" in b or "dpred" in b for b in batches)
                )
                or (key == "perturb" and any("perturb_objective" in b for b in batches))
            ):
                scheduler.step()
        (folder / "training_log.json").write_text(
            json.dumps(log, indent=2, allow_nan=False)
        )
        if stale >= args.patience:
            break
    if not (folder / "best.pt").exists():
        raise RuntimeError("No finite validation score; no valid checkpoint produced")


if __name__ == "__main__":
    main()
