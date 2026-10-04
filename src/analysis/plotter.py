"""
Main entry point for experiment plotting.
Detects run type (train/test/both) and delegates to plotter_train, plotter_test, or plotter_distillation.
"""
import os
import re
import glob
import yaml
from pathlib import Path

from .plotter_common import detect_run_type, backfill_parse_metrics
from .plotter_train import Plotter
from .plotter_test import plot_test_run
from .plotter_distillation import DistillationPlotter
from .plotter_arcface_layers import save_arcface_llama_layer_weights_from_run

__all__ = [
    "Plotter",
    "DistillationPlotter",
    "plot_test_run",
    "detect_run_type",
    "load_test_data",
    "compute_test_metrics",
    "extract_answer",
    "extract_answer_from_gt",
]


def _is_distillation_run(experiment_dir: str) -> bool:
    """Detect if run is distillation (logs/iteration_*/ or logs/distillation/ or config)."""
    config_path = os.path.join(experiment_dir, "config_resolved.yaml")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if cfg.get("Runner", {}).get("trainer") == "Distillation":
                return True
        except (yaml.YAMLError, OSError):
            pass
    logs_dir = os.path.join(experiment_dir, "logs")
    if os.path.isdir(logs_dir):
        for name in os.listdir(logs_dir):
            if name.startswith("iteration_"):
                return True
        if os.path.isdir(os.path.join(logs_dir, "distillation")):
            return True
    return False

# Re-export for backward compatibility
from .plotter_common import extract_answer, extract_answer_from_gt
from .plotter_test import load_test_data, compute_test_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to run directory (containing logs/)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for plots (default: experiment_dir/plots)")
    parser.add_argument("--run_type", type=str, default=None, choices=["train", "test"], help="Force run type (default: auto-detect)")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    output_dir = args.output_dir if args.output_dir else os.path.join(experiment_dir, "plots")

    run_type, test_log_dirs = detect_run_type(experiment_dir)
    if args.run_type:
        run_type = args.run_type
        if run_type == "test" and not test_log_dirs:
            logs_path = Path(experiment_dir) / "logs"
            if logs_path.exists():
                test_log_dirs = [str(d) for d in logs_path.iterdir() if d.is_dir() and d.name.startswith("test_")]

    if run_type == "unknown":
        print("Error: Could not detect run type. Expected logs/ with metrics.jsonl (train) or logs/test_*/ (test)")
        exit(1)

    print(f"Detected run type: {run_type}")
    print(f"Saving plots to: {output_dir}")
    print(f"  (absolute: {os.path.abspath(output_dir)})")

    if run_type in ("train", "both"):
        run_dir = os.path.join(experiment_dir, "logs")
        if not os.path.exists(run_dir) and os.path.exists(os.path.join(experiment_dir, "metrics.jsonl")):
            run_dir = experiment_dir
        if os.path.exists(run_dir):
            n = backfill_parse_metrics(experiment_dir)
            if n:
                print(f"Recomputed strict parse metrics for {n} validation epoch(s) from predictions")
            if _is_distillation_run(experiment_dir):
                print("\n--- Distillation plots ---")
                print(f"Loading from {run_dir}")
                plotter = DistillationPlotter(run_dir, output_dir)
                plotter.generate_plots()
            else:
                print("\n--- Training plots ---")
                print(f"Loading metrics from {run_dir}")
                plotter = Plotter(run_dir, output_dir)
                plotter.generate_plots()
                # Also generate confusion matrix from validation samples if present
                plot_test_run(run_dir, output_dir, "validation")
        else:
            print("Warning: No train logs found")

    if run_type in ("test", "both"):
        for test_log_dir in test_log_dirs:
            dataset_name = os.path.basename(test_log_dir).replace("test_", "")
            sub_output = os.path.join(output_dir, dataset_name) if len(test_log_dirs) > 1 else output_dir
            print(f"\n--- Test plots: {dataset_name} ---")
            plot_test_run(test_log_dir, sub_output, dataset_name)

            samples_file = glob.glob(os.path.join(test_log_dir, "samples_test_epoch_*.jsonl"))
            if not samples_file:
                samples_file = glob.glob(os.path.join(test_log_dir, "predictions_test_epoch_*.jsonl"))
            if samples_file:
                latest = max(
                    samples_file,
                    key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)) if re.search(r"epoch_(\d+)", p) else 0,
                )
                print(f"\nSample generations from: {latest}")
