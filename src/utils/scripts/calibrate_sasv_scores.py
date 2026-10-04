#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.config import load_config
from src.utils.sasv_score_calibration import run_score_calibration_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate ECAPA and AASIST scores")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_split", choices=("train", "val", "test"), default="train")
    p.add_argument("--eval_split", choices=("train", "val", "test"), default="val")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--method", choices=("sigmoid", "isotonic"), default="sigmoid")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--no_grid_search", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    runner_cfg = config.setdefault("Runner", {})
    runner_cfg["type"] = "score_calibration"
    cal = runner_cfg.setdefault("ScoreCalibration", {})
    cal["train_split"] = args.train_split
    cal["eval_split"] = args.eval_split
    if args.max_train_samples > 0:
        cal["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples > 0:
        cal["max_eval_samples"] = args.max_eval_samples
    cal["batch_size"] = args.batch_size
    cal["num_workers"] = args.num_workers
    cal["method"] = args.method
    cal["cv"] = args.cv
    cal["force_recompute"] = args.force_recompute
    cal["grid_search"] = not args.no_grid_search

    base = config.get("General", {}).get("output_dir", "./outputs")
    output_dir = args.output_dir or os.path.join(base, "score_calibration_cli")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    run_score_calibration_from_config(config, output_dir, device, local_rank=0)


if __name__ == "__main__":
    main()
