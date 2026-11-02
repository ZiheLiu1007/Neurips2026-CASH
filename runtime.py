import numpy as np
import pandas as pd


def cross_sectional_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted, actual = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if predicted.shape != actual.shape or predicted.ndim != 1:
        raise ValueError("Expected aligned cross-sectional return vectors")
    mask = np.isfinite(predicted) & np.isfinite(actual)
    if mask.sum() < 2:
        return float("nan")
    a, b = predicted[mask], actual[mask]
    a, b = a - a.mean(), b - b.mean()
    sa, sb = np.max(np.abs(a)), np.max(np.abs(b))
    if sa == 0 or sb == 0:
        return float("nan")
    a, b = a / sa, b / sb
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


def validate_input_contract(config, features, returns, layout, return_type):
    if layout != config.get("layout", "NTP"):
        raise ValueError("layout must match the training configuration")
    if return_type != config.get("return_type", "excess"):
        raise ValueError("return_type must match the training configuration")
    if features.shape[:2] != returns.shape:
        raise ValueError("Feature and return panels must be aligned")
    if features.shape[1] != config["n_assets"]:
        raise ValueError("Asset universe must match training")
    if features.shape[2] != config["feature_dim"]:
        raise ValueError("Feature dimension must match training")
    if config.get("input_shape") is not None and list(features.shape) != config["input_shape"]:
        raise ValueError("Use the same input panel and time axis as training")


def load_realized_returns(path, layout="NTP"):
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("Expected a nonempty return matrix")
    if layout == "NTP":
        values = values.T
    elif layout != "TNP":
        raise ValueError("layout must be NTP or TNP")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Realized returns must be real numbers")
    if not np.isfinite(values).all() or np.any(values < -1):
        raise ValueError("Realized returns must be finite raw decimal simple returns")
    return values.astype(np.float64)


def load_dates(path, expected_length=None):
    raw = np.load(path, allow_pickle=False)
    if raw.ndim == 2 and raw.shape[0]:
        if not np.all(raw == raw[:1]):
            raise ValueError("Timestamp rows must describe the same time axis")
        raw = raw[0]
    if raw.ndim != 1 or not len(raw):
        raise ValueError("Expected a nonempty timestamp vector")
    if expected_length is not None and len(raw) != expected_length:
        raise ValueError("Timestamp count must match the input panel")
    dates = pd.to_datetime(raw.astype(str), errors="raise")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and chronological")
    return dates


def nw_ols(y, regressors=None, lags=5):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not isinstance(lags, int) or lags < 0:
        raise ValueError("Expected a return vector and nonnegative integer lags")
    x = np.ones((len(y), 1))
    if regressors is not None:
        regressors = np.asarray(regressors, dtype=float)
        if regressors.ndim == 1:
            regressors = regressors[:, None]
        if regressors.ndim != 2 or len(regressors) != len(y):
            raise ValueError("Regressors must align with the return vector")
        x = np.column_stack([x, regressors])
    good = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[good], x[good]
    n, k = x.shape
    lag = min(lags, max(0, n - 1))
    result = {"intercept": None, "intercept_nw_t": None, "n": n, "lags": lag}
    if n <= k or np.linalg.matrix_rank(x) < k:
        return result
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    influence = np.linalg.pinv(x)[0] * (y - x @ beta)
    variance = float(influence @ influence)
    for j in range(1, lag + 1):
        variance += 2 * (1 - j / (lag + 1)) * float(influence[j:] @ influence[:-j])
    se = np.sqrt(max(0.0, variance))
    result["intercept"] = float(beta[0])
    result["intercept_nw_t"] = float(beta[0] / se) if se > 0 else None
    return result
