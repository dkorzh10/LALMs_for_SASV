"""Distillation run plotting: iteration/phase layout, inter-iteration metrics."""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from .plotter_train import Plotter
from .plotter_distillation_common import load_distillation_metrics, load_distillation_form_stats, get_distillation_iterations
from .plotter_dataset_forming import plot_dataset_forming


class DistillationPlotter:
    """Plotting for Distillation runs. Layout: plots/ root + plots/iteration_N/{dataset_forming,sft,grpo}."""

    def __init__(self, run_dir: str, output_dir: str):
        self.run_dir = run_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_plots(self):
        """Generate all distillation plots."""
        metrics = load_distillation_metrics(self.run_dir)
        form_stats = load_distillation_form_stats(self.run_dir)
        iters = get_distillation_iterations(self.run_dir)

        # Inter-iteration plots at root
        self._plot_val_accuracy_per_iteration(metrics)
        self._plot_text_len_per_iteration(form_stats)
        self._plot_filtering_ratio_per_iteration(form_stats)

        # Per-iteration subfolders
        for it in iters:
            iter_dir = os.path.join(self.output_dir, f"iteration_{it}")
            df_dir = os.path.join(iter_dir, "dataset_forming")
            plot_dataset_forming(self.run_dir, df_dir, it)

            # SFT: full logs same structure as sft_exp/logs/ -> standard Plotter
            sft_log_dir = os.path.join(self.run_dir, f"iteration_{it}", "sft")
            sft_plot_dir = os.path.join(iter_dir, "sft")
            if os.path.exists(sft_log_dir) and os.path.exists(os.path.join(sft_log_dir, "metrics.jsonl")):
                from .plotter_train import Plotter
                Plotter(sft_log_dir, sft_plot_dir).generate_plots()

            # GRPO: full logs same structure -> standard Plotter
            grpo_log_dir = os.path.join(self.run_dir, f"iteration_{it}", "grpo")
            grpo_plot_dir = os.path.join(iter_dir, "grpo")
            if os.path.exists(grpo_log_dir) and os.path.exists(os.path.join(grpo_log_dir, "metrics.jsonl")):
                from .plotter_train import Plotter
                Plotter(grpo_log_dir, grpo_plot_dir).generate_plots()

    def _plot_val_accuracy_per_iteration(self, metrics: List[Dict[str, Any]]):
        by_iter = {}
        for m in metrics:
            if "distillation_iter" in m and "accuracy" in str(m.keys()):
                it = m["distillation_iter"]
                acc = m.get("accuracy", m.get("last_accuracy"))
                if acc is not None:
                    by_iter.setdefault(it, []).append(acc)
        if not by_iter:
            return
        iters_sorted = sorted(by_iter.keys())
        accs = [np.mean(by_iter[it]) for it in iters_sorted]
        plt.figure(figsize=(10, 6))
        plt.plot(iters_sorted, accs, marker="o", linewidth=2, markersize=8)
        plt.xlabel("Distillation iteration")
        plt.ylabel("Validation accuracy")
        plt.title("Val accuracy per distillation iteration")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "val_accuracy_per_iteration.png"), dpi=150)
        plt.close()

    def _plot_text_len_per_iteration(self, form_stats: Dict[int, Dict[str, Any]]):
        if not form_stats:
            return
        iters = sorted(form_stats.keys())
        chars_mean = []
        tokens_mean = []
        for it in iters:
            s = form_stats[it]
            lc = s.get("lengths_chars", [])
            lt = s.get("lengths_tokens", [])
            chars_mean.append(np.mean(lc) if lc else 0)
            tokens_mean.append(np.mean(lt) if lt else 0)
        if chars_mean and any(c > 0 for c in chars_mean):
            plt.figure(figsize=(10, 6))
            plt.plot(iters, chars_mean, marker="o", label="chars", linewidth=2)
            plt.xlabel("Distillation iteration")
            plt.ylabel("Mean text length (chars)")
            plt.title("Text length per iteration (chars)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "text_len_per_iteration.png"), dpi=150)
            plt.close()
        if tokens_mean and any(t > 0 for t in tokens_mean):
            plt.figure(figsize=(10, 6))
            plt.plot(iters, tokens_mean, marker="s", label="tokens", color="seagreen", linewidth=2)
            plt.xlabel("Distillation iteration")
            plt.ylabel("Mean text length (tokens)")
            plt.title("Text length per iteration (tokens)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "text_len_tokens_per_iteration.png"), dpi=150)
            plt.close()

    def _plot_filtering_ratio_per_iteration(self, form_stats: Dict[int, Dict[str, Any]]):
        if not form_stats:
            return
        iters = sorted(form_stats.keys())
        ratios = []
        for it in iters:
            s = form_stats[it]
            n_raw = s.get("n_raw", 1)
            n_final = s.get("n_final", 0)
            ratios.append(n_final / n_raw if n_raw > 0 else 0)
        plt.figure(figsize=(10, 6))
        plt.bar(iters, ratios, color="steelblue", alpha=0.8)
        plt.xlabel("Distillation iteration")
        plt.ylabel("Filtering ratio (final/raw)")
        plt.title("Filtering ratio per iteration")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "filtering_ratio_per_iteration.png"), dpi=150)
        plt.close()
