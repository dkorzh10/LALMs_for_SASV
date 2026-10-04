"""Dataset forming plots for distillation."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any


def plot_dataset_forming(run_dir: str, output_dir: str, distillation_iter: int):
    """Generate dataset_forming plots for one iteration."""
    os.makedirs(output_dir, exist_ok=True)
    # New structure: iteration_i/dataset_forming/form_stats_iter_i.json
    stats_path = os.path.join(run_dir, f"iteration_{distillation_iter}", "dataset_forming", f"form_stats_iter_{distillation_iter}.json")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(run_dir, "distillation", f"form_stats_iter_{distillation_iter}.json")  # legacy
    if not os.path.exists(stats_path):
        return
    with open(stats_path, "r") as f:
        stats = json.load(f)

    lengths_chars = stats.get("lengths_chars", [])
    lengths_tokens = stats.get("lengths_tokens", [])

    if lengths_chars:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths_chars, bins=50, color="steelblue", edgecolor="navy", alpha=0.85)
        plt.xlabel("Text length (chars)")
        plt.ylabel("Count")
        plt.title(f"Dataset Forming iter {distillation_iter}: Text Length (chars) Distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "text_len_chars_distribution.png"), dpi=150)
        plt.close()

    if lengths_tokens:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths_tokens, bins=50, color="seagreen", edgecolor="darkgreen", alpha=0.85)
        plt.xlabel("Text length (tokens)")
        plt.ylabel("Count")
        plt.title(f"Dataset Forming iter {distillation_iter}: Text Length (tokens) Distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "text_len_tokens_distribution.png"), dpi=150)
        plt.close()

    n_raw = stats.get("n_raw")
    n_final = stats.get("n_final")
    n_after_skeptic = stats.get("n_after_skeptic")
    n_rollouts_skeptic = stats.get("n_rollouts_skeptic")
    n_rollouts_main = stats.get("n_rollouts_main")
    if n_raw is not None and n_final is not None and n_raw > 0:
        plt.figure(figsize=(8, 5))
        if n_after_skeptic is not None:
            # Three-stage funnel: Raw -> Candidates (after skeptic) -> Clean (after all)
            labels = ["Raw", "Candidates\n(after skeptic)", "Clean\n(after all)"]
            vals = [n_raw, n_after_skeptic, n_final]
            colors = ["coral", "gold", "seagreen"]
        else:
            labels = ["Raw", "Clean"]
            vals = [n_raw, n_final]
            colors = ["coral", "seagreen"]
        plt.bar(labels, vals, color=colors[: len(labels)], alpha=0.8)
        plt.ylabel("Sample count")
        plt.title(f"Dataset Forming iter {distillation_iter}: Conversion ratio")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "filtering_funnel.png"), dpi=150)
        plt.close()

    # Rollout-level: all rollouts (incl. rejected) vs filtered samples
    if n_rollouts_main is not None and n_rollouts_main > 0 and n_final is not None:
        plt.figure(figsize=(8, 5))
        labels = []
        vals = []
        colors = []
        if n_rollouts_skeptic is not None and n_rollouts_skeptic > 0:
            labels.extend(["All rollouts\n(skeptic)", "All rollouts\n(main)"])
            vals.extend([n_rollouts_skeptic, n_rollouts_main])
            colors.extend(["coral", "gold"])
        else:
            labels.append("All rollouts\n(main)")
            vals.append(n_rollouts_main)
            colors.append("gold")
        labels.append("Clean\n(samples)")
        vals.append(n_final)
        colors.append("seagreen")
        plt.bar(labels, vals, color=colors, alpha=0.8)
        plt.ylabel("Count")
        ratio = n_final / n_rollouts_main
        plt.title(f"Dataset Forming iter {distillation_iter}: Rollouts vs filtered (ratio clean/all_main={ratio:.3f})")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "conversion_ratio_rollouts.png"), dpi=150)
        plt.close()
