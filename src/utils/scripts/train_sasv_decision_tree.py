#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_UNIFIED_ROOT = _SCRIPT_DIR.parents[2]
if str(_UNIFIED_ROOT) not in sys.path:
    sys.path.insert(0, str(_UNIFIED_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.sasv_tree_training import train_decision_tree_from_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train SASV decision tree (standalone)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="", help="Override General.output_dir")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("General", {})["output_dir"] = args.output_dir

    out_base = config.get("General", {}).get("output_dir", "./outputs")
    import os
    from datetime import datetime

    output_dir = os.path.join(out_base, f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M')}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    path = train_decision_tree_from_config(config, output_dir, device)
    print(f"Done. Tree: {path}", flush=True)
    print(f"Test with fusion_mode: tree and fusion_tree_path: {path}", flush=True)


if __name__ == "__main__":
    main()
