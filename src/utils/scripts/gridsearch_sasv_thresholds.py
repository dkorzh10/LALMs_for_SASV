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
from src.utils.sasv_threshold_gridsearch import (
    parse_float_list,
    run_threshold_gridsearch_from_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search CM+ECAPA thresholds")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", choices=("train", "val", "test"), default="val")
    p.add_argument("--dataset_path", default="")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="")
    p.add_argument("--cache", default="")
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--grid_only", action="store_true")
    p.add_argument("--cm_thresholds", default="0.3,0.4,0.5,0.6,0.7")
    p.add_argument("--ecapa_thresholds", default="0.25,0.35,0.45,0.55,0.65")
    p.add_argument("--q_values", default="0.5")
    p.add_argument("--asv_signal", choices=("ecapa", "w2v", "mean"), default="ecapa")
    p.add_argument(
        "--metric",
        choices=("eer_sasv", "t_eer", "min_a_dcf", "accuracy"),
        default="eer_sasv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    runner_cfg = config.setdefault("Runner", {})
    runner_cfg["type"] = "fusion_gridsearch"
    grid = runner_cfg.setdefault("FusionGridsearch", {})
    grid["dataset"] = args.dataset
    if args.dataset_path:
        grid["dataset_path"] = args.dataset_path
    if args.max_samples > 0:
        grid["max_samples"] = args.max_samples
    grid["batch_size"] = args.batch_size
    grid["num_workers"] = args.num_workers
    grid["grid_only"] = args.grid_only
    grid["force_recompute"] = args.force_recompute
    grid["asv_signal"] = args.asv_signal
    grid["metric"] = args.metric
    grid["cm_thresholds"] = parse_float_list(args.cm_thresholds, [])
    grid["ecapa_thresholds"] = parse_float_list(args.ecapa_thresholds, [])
    grid["q_values"] = parse_float_list(args.q_values, [])
    if args.cache:
        grid["cache_features"] = args.cache

    base = config.get("General", {}).get("output_dir", "./outputs")
    output_dir = args.output_dir or os.path.join(base, "threshold_gridsearch_cli")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    run_threshold_gridsearch_from_config(config, output_dir, device, local_rank=0)


if __name__ == "__main__":
    main()
