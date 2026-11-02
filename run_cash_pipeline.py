from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent


def option_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_options(cmd: List[str], options: Dict[str, Any]) -> None:
    for key, value in options.items():
        if value is None:
            continue
        opt = option_name(key)
        if isinstance(value, bool):
            if value:
                cmd.append(opt)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cmd.extend([opt, str(item)])
        else:
            cmd.extend([opt, str(value)])


def run(cmd: List[str], dry_run: bool) -> None:
    print(shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def selected_stages(raw: Iterable[str]) -> List[str]:
    stages = list(raw)
    if "all" in stages:
        return ["train", "predict", "evaluate", "portfolio"]
    aliases = {"test": "predict", "compute": "evaluate", "portfolio_backtest": "portfolio"}
    return list(dict.fromkeys(aliases.get(stage, stage) for stage in stages))


def build_data_paths(cfg: Dict[str, Any], include_timestamps: bool) -> Dict[str, str]:
    data_cfg = cfg.get("data", {})
    paths = {
        "features_path": data_cfg.get("features_path"),
        "returns_path": data_cfg.get("returns_path"),
        "layout": data_cfg.get("layout", "NTP"),
        "return_type": data_cfg.get("return_type", "excess"),
    }
    if include_timestamps:
        paths["timestamps_path"] = data_cfg.get("timestamps_path")
    return {k: v for k, v in paths.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CASH from Python presets or an external JSON config.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.py")
    parser.add_argument("--preset", default="main")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "train", "predict", "evaluate", "portfolio", "portfolio_backtest", "test", "compute"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.preset)
    if not isinstance(cfg, dict):
        raise ValueError("The pipeline requires a single experiment configuration")
    output_root = Path(cfg.get("output_root", "../CASH_outputs/cash_run")).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    model_dir = output_root / "model"
    pred_dir = output_root / "predictions"
    metric_dir = output_root / "metrics"
    portfolio_dir = output_root / "portfolio"

    data_without_timestamps = build_data_paths(cfg, include_timestamps=False)
    data_with_timestamps = build_data_paths(cfg, include_timestamps=True)
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("evaluation", {})
    test_eval_cfg = {
        "fbar_lookback": eval_cfg.get("fbar_lookback"),
    }
    compute_eval_cfg = {
        "fbar_lookback": eval_cfg.get("fbar_lookback"),
        "frequency": eval_cfg.get("frequency"),
    }
    if eval_cfg.get("calibrate_mode", "none") not in (None, "none"):
        raise ValueError("Target-based prediction calibration is not supported")
    portfolio_cfg = dict(cfg.get("portfolio", {}))
    if eval_cfg.get("frequency") is not None:
        portfolio_cfg.setdefault("frequency", eval_cfg["frequency"])
    portfolio_data = dict(data_with_timestamps)
    portfolio_data["realized_returns_path"] = cfg.get("data", {}).get("realized_returns_path")
    stages = selected_stages(args.stages)
    if "portfolio" in stages and portfolio_data["return_type"] == "excess" and not portfolio_data["realized_returns_path"]:
        raise ValueError("Excess targets require data.realized_returns_path with raw simple returns")

    for stage in stages:
        if stage == "train":
            cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]
            append_options(cmd, data_without_timestamps)
            append_options(cmd, train_cfg)
            cmd.extend(["--output-dir", str(model_dir)])
            run(cmd, args.dry_run)
        elif stage == "predict":
            cmd = [sys.executable, str(PROJECT_ROOT / "predict.py"), "--model-dir", str(model_dir)]
            append_options(cmd, data_without_timestamps)
            append_options(cmd, test_eval_cfg)
            cmd.extend(["--output-dir", str(pred_dir)])
            run(cmd, args.dry_run)
        elif stage == "evaluate":
            cmd = [sys.executable, str(PROJECT_ROOT / "evaluate.py"), "--model-dir", str(model_dir)]
            append_options(cmd, data_with_timestamps)
            append_options(cmd, compute_eval_cfg)
            cmd.extend(["--output-dir", str(metric_dir)])
            run(cmd, args.dry_run)
        elif stage == "portfolio":
            cmd = [sys.executable, str(PROJECT_ROOT / "portfolio_backtest.py"), "--model-dir", str(model_dir)]
            append_options(cmd, portfolio_data)
            append_options(cmd, test_eval_cfg)
            append_options(cmd, portfolio_cfg)
            cmd.extend(["--output-dir", str(portfolio_dir)])
            run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
