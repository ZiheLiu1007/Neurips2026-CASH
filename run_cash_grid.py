from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def deep_update(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or materialize a CASH grid.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("../CASH_outputs/grid")
    )
    parser.add_argument("--materialize-dir", type=Path, default=None)
    parser.add_argument("--stages", nargs="+", default=["all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(
        s not in ["all", "train", "test", "compute", "portfolio"] for s in args.stages
    ):
        parser.error("Unknown pipeline stage")
    args.output_root = args.output_root.expanduser()
    if not args.output_root.is_absolute():
        args.output_root = PROJECT_ROOT / args.output_root

    base_cfg = load_json(args.base_config)
    grid_items: List[Dict[str, Any]] = load_json(args.grid)

    materialize_dir = args.materialize_dir
    if materialize_dir:
        materialize_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(grid_items, start=1):
        run_name = str(item["name"])
        patch = item.get("patch", {})
        cfg = deep_update(copy.deepcopy(base_cfg), patch)
        cfg["output_root"] = str(args.output_root / f"{idx:02d}_{run_name}")

        cfg_path = (
            materialize_dir / f"{idx:02d}_{run_name}.json"
            if materialize_dir
            else args.output_root / f"{idx:02d}_{run_name}.json"
        )
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_cash_pipeline.py"),
            "--config",
            str(cfg_path.resolve()),
            "--stages",
            *args.stages,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
