"""Shared helpers for distillation plotting."""
import os
import json
import glob
import re
from typing import Dict, List, Any
from collections import defaultdict


def load_distillation_metrics(run_dir: str) -> List[Dict[str, Any]]:
    """Load distillation metrics from metrics.jsonl (type=distillation_*)."""
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return []
    out = []
    with open(metrics_path, "r") as f:
        for line in f:
            try:
                m = json.loads(line)
                if m.get("type", "").startswith("distillation"):
                    out.append(m)
            except json.JSONDecodeError:
                pass
    return out


def load_distillation_form_stats(run_dir: str) -> Dict[int, Dict[str, Any]]:
    """Load form_stats_iter_*.json from logs/iteration_i/dataset_forming/ (or legacy logs/distillation/)."""
    out = {}
    # New structure: iteration_i/dataset_forming/form_stats_iter_i.json
    for name in os.listdir(run_dir):
        m = re.match(r"iteration_(\d+)", name)
        if m:
            it = int(m.group(1))
            df_dir = os.path.join(run_dir, name, "dataset_forming")
            stats_path = os.path.join(df_dir, f"form_stats_iter_{it}.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    out[it] = json.load(f)
    # Legacy: logs/distillation/form_stats_iter_*.json
    if not out:
        distill_dir = os.path.join(run_dir, "distillation")
        if os.path.isdir(distill_dir):
            for fname in os.listdir(distill_dir):
                m = re.match(r"form_stats_iter_(\d+)\.json", fname)
                if m:
                    it = int(m.group(1))
                    with open(os.path.join(distill_dir, fname), "r") as f:
                        out[it] = json.load(f)
    return out


def get_distillation_iterations(run_dir: str) -> List[int]:
    """Get list of distillation iteration numbers from logs."""
    metrics = load_distillation_metrics(run_dir)
    iters = sorted(set(m.get("distillation_iter", 0) for m in metrics if "distillation_iter" in m))
    if not iters:
        stats = load_distillation_form_stats(run_dir)
        iters = sorted(stats.keys())
    return iters
