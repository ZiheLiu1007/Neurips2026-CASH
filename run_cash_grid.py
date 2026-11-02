from __future__ import annotations

import argparse
import copy
import pprint
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent


def deep_update(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or materialize a CASH grid.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.py")
    parser.add_argument("--preset", default="ablations", help="Grid preset: ablations or sensitivity")
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--base-preset", default="main")
    parser.add_argument("--grid", type=Path, default=None, help="Optional external Python or JSON grid")
    parser.add_argument("--output-root", type=Path, default=Path("../CASH_outputs/grid"))
    parser.add_argument("--materialize-dir", type=Path, default=None)
    parser.add_argument("--stages", nargs="+", default=["all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config or args.config, args.base_preset)
    grid_items: List[Dict[str, Any]] = load_config(args.grid or args.config, args.preset)
    if not isinstance(base_cfg, dict) or not isinstance(grid_items, list):
        raise ValueError("A grid requires a base configuration dict and a list of patches")
    args.output_root = args.output_root.expanduser()
    if not args.output_root.is_absolute():
        args.output_root = PROJECT_ROOT / args.output_root
    args.output_root = args.output_root.expanduser().resolve()

    materialize_dir = args.materialize_dir
    if materialize_dir:
        materialize_dir.mkdir(parents=True, exist_ok=True)

    seen_names = set()
    for idx, item in enumerate(grid_items, start=1):
        if not isinstance(item, dict) or "name" not in item or "patch" not in item or not isinstance(item["patch"], dict):
            raise ValueError("Each grid entry requires a name and a patch dictionary")
        run_name = str(item["name"])
        if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name or run_name in seen_names:
            raise ValueError(f"Invalid or repeated run name: {run_name}")
        seen_names.add(run_name)
        patch = item.get("patch", {})
        cfg = deep_update(copy.deepcopy(base_cfg), patch)
        cfg["output_root"] = str(args.output_root / f"{idx:02d}_{run_name}")

        cfg_path = (
            materialize_dir / f"{idx:02d}_{run_name}.py"
            if materialize_dir
            else args.output_root / f"{idx:02d}_{run_name}.py"
        )
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_path.resolve()
        cfg_path.write_text("CONFIG = " + pprint.pformat(cfg, sort_dicts=False) + "\n", encoding="utf-8")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_cash_pipeline.py"),
            "--config",
            str(cfg_path),
            "--stages",
            *args.stages,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
