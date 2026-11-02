from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def option_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def selected_stages(raw: Iterable[str]) -> List[str]:
    stages = list(raw)
    if "all" in stages:
        return ["train", "test", "compute", "portfolio"]
    return stages


def build_data_paths(cfg: Dict[str, Any], include_timestamps: bool) -> Dict[str, str]:
    data_cfg = cfg.get("data", {})
    paths = {
        "features_path": data_cfg.get("features_path"),
        "returns_path": data_cfg.get("returns_path"),
    }
    if include_timestamps:
        paths["timestamps_path"] = data_cfg.get("timestamps_path")
    result = {k: str(project_path(v)) for k, v in paths.items() if v}
    result["return_type"] = data_cfg.get("return_type", "excess")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CASH pipeline from a JSON config."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "train", "test", "compute", "portfolio"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    output_root = project_path(cfg.get("output_root", "../CASH_outputs/cash_run"))
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
        "cuda": train_cfg.get("cuda", 0),
    }
    compute_eval_cfg = {
        "fbar_lookback": eval_cfg.get("fbar_lookback"),
        "frequency": eval_cfg.get("frequency", "daily"),
        "cuda": train_cfg.get("cuda", 0),
    }
    portfolio_cfg = {
        "frequency": eval_cfg.get("frequency", "daily"),
        **cfg.get("portfolio", {}),
    }
    if cfg.get("data", {}).get("realized_returns_path"):
        portfolio_cfg["realized_returns_path"] = cfg["data"]["realized_returns_path"]
    for key in ("factors_path", "realized_returns_path"):
        if portfolio_cfg.get(key):
            portfolio_cfg[key] = str(project_path(portfolio_cfg[key]))
    layout = cfg.get("data", {}).get("layout", "NTP")
    data_without_timestamps["layout"] = layout
    data_with_timestamps["layout"] = layout

    for stage in selected_stages(args.stages):
        if stage == "train":
            cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]
            append_options(cmd, data_without_timestamps)
            append_options(cmd, train_cfg)
            cmd.extend(["--output-dir", str(model_dir)])
            run(cmd, args.dry_run)
        elif stage == "test":
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "predict.py"),
                "--model-dir",
                str(model_dir),
            ]
            append_options(cmd, data_without_timestamps)
            append_options(cmd, test_eval_cfg)
            cmd.extend(["--output-dir", str(pred_dir)])
            run(cmd, args.dry_run)
        elif stage == "compute":
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "evaluate.py"),
                "--model-dir",
                str(model_dir),
            ]
            append_options(cmd, data_with_timestamps)
            append_options(cmd, compute_eval_cfg)
            cmd.extend(["--output-dir", str(metric_dir)])
            run(cmd, args.dry_run)
        elif stage == "portfolio":
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "portfolio_backtest.py"),
                "--model-dir",
                str(model_dir),
            ]
            append_options(cmd, data_with_timestamps)
            append_options(cmd, test_eval_cfg)
            append_options(cmd, portfolio_cfg)
            cmd.extend(["--output-dir", str(portfolio_dir)])
            run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
