#!/usr/bin/env python3
"""Recompute ASVspoof5 SASV metrics from saved predictions JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_paths() -> None:
    unified_src = Path(__file__).resolve().parents[2]
    unified_src_str = str(unified_src)
    if unified_src_str not in sys.path:
        sys.path.insert(0, unified_src_str)


def main() -> None:
    _bootstrap_paths()

    from analysis.plotter_test import load_test_data
    from epochs.utils.sasv_metrics import (
        print_sasv_metrics_summary,
        recompute_sasv_metrics_from_predictions,
        resolve_run_plots_dir,
    )

    parser = argparse.ArgumentParser(
        description="Recompute ASVspoof5 SASV metrics from predictions_*.jsonl",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Test log directory containing predictions_test_epoch_*.jsonl",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for DET/DCF plots (default: <run_dir>/plots/recomputed)",
    )
    parser.add_argument(
        "--plot-prefix",
        default="sasv_from_jsonl",
        help="Filename prefix for saved plots",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write metrics dict as JSON",
    )
    parser.add_argument(
        "--no-subsystem-proxy",
        action="store_true",
        help="Skip LLM probability proxies for asv_scores/cm_scores",
    )
    args = parser.parse_args()

    predictions, _ = load_test_data(args.log_dir)
    if not predictions:
        raise SystemExit(f"No predictions found in {args.log_dir}")

    plot_dir = args.plot_dir or resolve_run_plots_dir(args.log_dir)

    metrics = recompute_sasv_metrics_from_predictions(
        predictions,
        plot_dir=plot_dir,
        plot_prefix=args.plot_prefix,
        proxy_subsystem_scores=not args.no_subsystem_proxy,
    )
    print(f"Loaded {len(predictions)} predictions from {args.log_dir}", flush=True)
    print_sasv_metrics_summary(metrics)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: float(v) for k, v in metrics.items()}
        out_path.write_text(json.dumps(serializable, indent=2) + "\n")
        print(f"Wrote metrics to {out_path}", flush=True)


if __name__ == "__main__":
    main()
