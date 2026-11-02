from pathlib import Path
import json
import numpy as np
import torch
from cash_model import CASH, ModelConfig
from beta_network import pricing_returns


def device_for(index=0):
    return torch.device(f"cuda:{index}" if torch.cuda.is_available() else "cpu")


def float_tensor(array):
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("Inputs must be real numeric arrays")
    if not np.isfinite(array).all():
        raise ValueError("Inputs must be finite; clean missing values externally")
    if np.any(np.abs(array.astype(np.float64)) > np.finfo(np.float32).max):
        raise ValueError("Inputs exceed the float32 range")
    return torch.as_tensor(array.astype("float32"))


def load_returns(path, layout="NTP", return_type="excess"):
    if return_type not in ("simple", "excess"):
        raise ValueError("return_type must be simple or excess")
    r = np.load(path, allow_pickle=False)
    if r.ndim != 2 or not all(r.shape):
        raise ValueError("Expected a nonempty return matrix")
    if layout == "NTP":
        r = r.T
    elif layout != "TNP":
        raise ValueError("layout must be NTP or TNP")
    r = float_tensor(r)
    if return_type == "simple" and torch.any(r < -1).item():
        raise ValueError("Raw decimal simple returns cannot be below -1")
    return r


def load_inputs(features_path, returns_path, layout="NTP", return_type="excess"):
    x = np.load(features_path, allow_pickle=False)
    if x.ndim != 3 or not all(x.shape):
        raise ValueError("Expected a nonempty feature [N,T,P] array")
    if layout == "NTP":
        x = x.transpose(1, 0, 2)
    elif layout != "TNP":
        raise ValueError("layout must be NTP or TNP")
    r = load_returns(returns_path, layout, return_type)
    if x.shape[:2] != r.shape:
        raise ValueError("Expected aligned feature [T,N,P] and return [T,N] arrays")
    return float_tensor(x), r


def split_indices(t, window, train_ratio=0.7, val_ratio=0.1):
    if window < 1:
        raise ValueError("window must be positive")
    if not 0 < train_ratio < train_ratio + val_ratio < 1:
        raise ValueError("Invalid chronological split ratios")
    a = int(t * train_ratio)
    b = a + int(t * val_ratio)
    parts = {
        "train": list(range(window, a)),
        "val": list(range(max(window, a), b)),
        "test": list(range(max(window, b), t)),
    }
    if any(not v for v in parts.values()):
        raise ValueError("Not enough periods for the requested window and split")
    return parts


def load_model(model_dir, device):
    folder = Path(model_dir)
    metadata = json.loads((folder / "config.json").read_text())
    model = CASH(ModelConfig(**metadata["model"])).to(device)
    checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, metadata


@torch.no_grad()
def predict(model, x, r, indices, lookback=3, device="cpu"):

    if lookback < 1:
        raise ValueError("lookback must be positive")
    model.eval()
    window = model.config.window
    if x.ndim != 3 or r.ndim != 2 or x.shape[:2] != r.shape:
        raise ValueError("Expected aligned feature [T,N,P] and return [T,N] tensors")
    if r.shape[1] != model.config.n_assets or x.shape[2] != model.config.feature_dim:
        raise ValueError("Asset universe and feature dimensions must match training")
    indices = list(indices)
    if not indices or any(t < max(window, lookback) or t >= len(x) for t in indices):
        raise ValueError(
            "Forecast indices must be nonempty and have observable history"
        )
    means = model.factor.factors(r.to(device))
    out = {"mu_pred": [], "mu_total": [], "returns": [], "indices": []}
    for t in indices:
        history = x[t - window : t].unsqueeze(0).to(device)
        beta = model.beta(model.encoder(history))
        fbar = means[t - lookback : t].mean(dim=0, keepdim=True)
        out["mu_pred"].append(pricing_returns(beta, fbar)[0].cpu().numpy())

        out["mu_total"].append(pricing_returns(beta, means[t : t + 1])[0].cpu().numpy())
        out["returns"].append(r[t].cpu().numpy())
        out["indices"].append(t)
    out = {k: np.asarray(v) for k, v in out.items()}
    if any(not np.isfinite(value).all() for value in out.values()):
        raise FloatingPointError("Nonfinite predictions; inspect inputs and checkpoint")
    return out


def add_prediction_arguments(parser, timestamps=False):
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--returns-path", required=True)
    parser.add_argument("--layout", choices=["NTP", "TNP"], default="NTP")
    parser.add_argument("--return-type", choices=["simple", "excess"], default="excess")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fbar-lookback", type=int, default=3)
    parser.add_argument("--cuda", type=int, default=0)
    if timestamps:
        parser.add_argument("--timestamps-path", required=True)


def prediction_inputs(args):
    device = device_for(args.cuda)
    model, metadata = load_model(args.model_dir, device)
    trained_type = metadata["training"].get("return_type", "simple")
    if args.return_type != trained_type:
        raise ValueError("return_type must match the training checkpoint")
    x, r = load_inputs(
        args.features_path, args.returns_path, args.layout, args.return_type
    )
    if "input_shape" in metadata and list(x.shape) != metadata["input_shape"]:
        raise ValueError("Use the same input panel and time axis as training")
    split = metadata["training"]
    indices = split_indices(
        len(x), model.config.window, split["train_ratio"], split["val_ratio"]
    )
    return model, x, r, indices, device


def dates_for(path, length):
    import pandas as pd

    raw = np.load(path, allow_pickle=False)
    if raw.ndim != 1 or len(raw) != length:
        raise ValueError("Timestamp count does not match input periods")
    dates = pd.to_datetime(raw.astype(str), errors="raise")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and chronological")
    return dates
