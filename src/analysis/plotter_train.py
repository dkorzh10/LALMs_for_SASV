"""
Training run plotting: loss curves, learning rate, validation metrics, GRPO, etc.
"""
import os
import re
import json
import glob
from numbers import Real
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

from .plotter_common import compute_sasv_subsystem_accuracies


class Plotter:
    def __init__(self, run_dir: str, output_dir: str):
        self.run_dir = run_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_metrics(self) -> List[Dict[str, Any]]:
        """Load metrics.jsonl, handling distributed training (multiple files if present).
        When metrics_rank*.jsonl exist, only those are loaded (no merge with metrics.jsonl)
        to avoid duplicates. Batch metrics with the same iteration are aggregated by mean
        to remove vertical lines from multi-rank logging."""
        metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        metrics = []

        rank_files = sorted(glob.glob(os.path.join(self.run_dir, "metrics_rank*.jsonl")))
        if rank_files:
            # Distributed: load only rank files so we have one source of truth
            for fpath in rank_files:
                with open(fpath, "r") as f:
                    for line in f:
                        try:
                            metrics.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        elif os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                for line in f:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        sorted_metrics = []
        for m in metrics:
            if "iteration" not in m:
                m["iteration_sort"] = m.get("epoch", 0) * 1000000 + 999999
            else:
                m["iteration_sort"] = m["iteration"]
            sorted_metrics.append(m)

        sorted_metrics.sort(key=lambda x: x["iteration_sort"])
        return self._aggregate_batch_metrics_by_iteration(sorted_metrics)

    def _aggregate_batch_metrics_by_iteration(
        self, metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collapse multiple train_batch/grpo_batch entries per (epoch, iteration) (e.g. from ranks) into one by averaging.
        Uses (type, epoch, iteration) as key so we don't merge across epochs."""
        batch_types = {"train_batch", "grpo_batch"}
        numeric_keys = {"loss", "lr", "reward", "kl_div", "correct_reasons_overlap", "epoch",
                       "skeptic_rejected_per_check_batch", "skeptic_accepted_per_check_batch"}
        by_key: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        others: List[Dict[str, Any]] = []

        for m in metrics:
            t = m.get("type")
            if t in batch_types and "iteration" in m:
                ep = m.get("epoch", 0)
                by_key[(t, ep, m["iteration"])].append(m)
            else:
                others.append(m)

        aggregated = []
        for step, ((t, ep, it), group) in enumerate(sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))):
            out = {"type": t, "epoch": ep, "iteration": it, "step": step, "iteration_sort": ep * 1000000 + it}
            for key in numeric_keys:
                vals = [g[key] for g in group if key in g and g[key] is not None]
                if vals:
                    out[key] = int(np.round(np.mean(vals))) if key == "epoch" else float(np.mean(vals))
            aggregated.append(out)
        aggregated.extend(others)
        aggregated.sort(key=lambda x: x.get("iteration_sort", 0))
        return aggregated

    def _load_judge_logs_series(self) -> List[Dict[str, Any]]:
        """Load reward and correct_reasons_overlap from judge_logs/*.json (one row per epoch/iter, aggregated over ranks).
        Returns list of {epoch, iteration, reward_mean, overlap_mean} sorted by (epoch, iteration).
        Plots can use this from the first GRPO step without needing metrics.jsonl or validation."""
        judge_dir = os.path.join(self.run_dir, "judge_logs")
        if not os.path.isdir(judge_dir):
            return []
        # epoch_N_iter_M[_rankR].json only (GRPO batches; skeptic_epoch has different format: audio_id, rollouts, has_passed)
        pattern = re.compile(r"^epoch_(\d+)_iter_(\d+)(?:_rank\d+)?\.json$")
        by_key: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for fname in os.listdir(judge_dir):
            if not fname.endswith(".json") or fname.startswith("skeptic_"):
                continue
            m = pattern.search(fname)
            if not m:
                continue
            epoch, iteration = int(m.group(1)), int(m.group(2))
            fpath = os.path.join(judge_dir, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue
            rewards = []
            overlaps = []
            text_lengths = []
            for sample in data:
                for g in sample.get("generations", []):
                    if "reward" in g:
                        rewards.append(float(g["reward"]))
                    ov = g.get("correct_reasons_overlap")
                    if ov is not None:
                        overlaps.append(float(ov))
                    if "text" in g:
                        text_lengths.append(len(str(g["text"])))
            if not rewards:
                continue
            by_key[(epoch, iteration)].append({
                "epoch": epoch,
                "iteration": iteration,
                "reward_mean": float(np.mean(rewards)),
                "overlap_mean": float(np.mean(overlaps)) if overlaps else None,
                "text_length_mean": float(np.mean(text_lengths)) if text_lengths else None,
            })
        # Aggregate rank files for same (epoch, iter)
        out = []
        for (epoch, it), group in sorted(by_key.items()):
            r_vals = [x["reward_mean"] for x in group]
            o_vals = [x["overlap_mean"] for x in group if x.get("overlap_mean") is not None]
            tl_vals = [x["text_length_mean"] for x in group if x.get("text_length_mean") is not None]
            out.append({
                "epoch": epoch,
                "iteration": it,
                "reward_mean": float(np.mean(r_vals)),
                "overlap_mean": float(np.mean(o_vals)) if o_vals else None,
                "text_length_mean": float(np.mean(tl_vals)) if tl_vals else None,
            })
        return out

    def _aggregate_validation_by_epoch(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate multiple validation entries per epoch by averaging them."""
        epoch_data = defaultdict(
            lambda: {
                "count": 0,
                "numeric_metrics": defaultdict(list),
                "metadata_metrics": defaultdict(list),
            }
        )

        for m in metrics:
            if m.get("type") == "validation":
                epoch = m["epoch"]
                epoch_data[epoch]["count"] += 1
                for k, v in m.items():
                    if k not in ["epoch", "type"]:
                        if isinstance(v, Real) and not isinstance(v, bool):
                            epoch_data[epoch]["numeric_metrics"][k].append(v)
                        elif v is not None:
                            epoch_data[epoch]["metadata_metrics"][k].append(v)

        aggregated = []
        for epoch in sorted(epoch_data.keys()):
            entry = {"epoch": epoch, "type": "validation"}
            for k, values in epoch_data[epoch]["numeric_metrics"].items():
                entry[k] = sum(values) / len(values)
            for k, values in epoch_data[epoch]["metadata_metrics"].items():
                unique_values = []
                for value in values:
                    if value not in unique_values:
                        unique_values.append(value)
                if len(unique_values) == 1:
                    entry[k] = unique_values[0]
            aggregated.append(entry)

        return aggregated

    def plot_learning_rate(self, metrics: List[Dict[str, Any]]):
        batch_metrics = [m for m in metrics if m.get("type") in ("train_batch", "grpo_batch") and "lr" in m]
        if not batch_metrics:
            return
        lrs = [m["lr"] for m in batch_metrics]
        # Calculate global iteration: epoch * max_iter + iteration
        # Use all batch metrics (not just those with lr) to get correct max_iter
        all_batch_metrics = [m for m in metrics if m.get("type") in ("train_batch", "grpo_batch") and "iteration" in m]
        max_iter = max((m.get("iteration", 0) for m in all_batch_metrics), default=0) + 1
        iterations = [m.get("epoch", 0) * max_iter + m.get("iteration", 0) for m in batch_metrics]

        plt.figure(figsize=(12, 6))
        plt.plot(iterations, lrs, color="orange", linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.8)
        plt.savefig(os.path.join(self.output_dir, "learning_rate.png"), dpi=150)
        plt.close()

    def plot_losses(self, metrics: List[Dict[str, Any]]):
        train_loss = []
        train_iterations = []

        val_metrics_agg = self._aggregate_validation_by_epoch(metrics)
        val_loss = [m["loss"] for m in val_metrics_agg if "loss" in m]
        val_epochs = [m["epoch"] for m in val_metrics_agg]

        batch_metrics = [m for m in metrics if m.get("type") in ("train_batch", "grpo_batch") and "loss" in m]
        train_loss = [m["loss"] for m in batch_metrics]
        # Calculate global iteration: epoch * max_iter + iteration
        # Use all batch metrics (not just those with loss) to get correct max_iter
        all_batch_metrics = [m for m in metrics if m.get("type") in ("train_batch", "grpo_batch") and "iteration" in m]
        max_iter = max((m.get("iteration", 0) for m in all_batch_metrics), default=0) + 1
        train_iterations = [m.get("epoch", 0) * max_iter + m.get("iteration", 0) for m in batch_metrics]

        if not train_loss:
            return

        # For log scale: clamp to avoid log(0) or log(negative)
        train_loss_safe = np.maximum(np.asarray(train_loss, dtype=float), 1e-10)

        plt.figure(figsize=(12, 6))
        plt.plot(train_iterations, train_loss, label="Raw", alpha=0.4, color="blue")
        if len(train_loss) > 10:
            ma = np.convolve(train_loss, np.ones(10) / 10, mode="valid")
            plt.plot(train_iterations[9:], ma, label="Smoothed (MA-10)", color="blue", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.8)
        plt.savefig(os.path.join(self.output_dir, "training_loss.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(train_iterations, train_loss_safe, alpha=0.4, color="blue")
        if len(train_loss) > 10:
            ma = np.convolve(train_loss_safe, np.ones(10) / 10, mode="valid")
            plt.plot(train_iterations[9:], ma, color="blue", linewidth=2)
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss Over Time (Log Scale)", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.5, which="both")
        plt.savefig(os.path.join(self.output_dir, "training_loss_log.png"), dpi=150)
        plt.close()

        last_n = min(100, len(train_loss))
        if last_n > 5:
            plt.figure(figsize=(12, 6))
            plt.plot(train_iterations[-last_n:], train_loss[-last_n:], alpha=0.6, color="blue", marker="o", markersize=3)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"Training Loss (Last {last_n} logs)", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.8)
            plt.savefig(os.path.join(self.output_dir, "training_loss_recent.png"), dpi=150)
            plt.close()

        if val_loss:
            plt.figure(figsize=(10, 6))
            plt.plot(val_epochs, val_loss, marker="o", color="red", linewidth=2, markersize=8)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Validation Loss per Epoch", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.8)
            plt.savefig(os.path.join(self.output_dir, "validation_loss.png"), dpi=150)
            plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes[0, 0].plot(train_iterations, train_loss_safe, alpha=0.3, color="royalblue", label="Batch")
        if len(train_loss) > 10:
            ma = np.convolve(train_loss_safe, np.ones(10) / 10, mode="valid")
            axes[0, 0].plot(train_iterations[9:], ma, color="navy", linewidth=2, label="Smoothed")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlabel("Iteration", fontsize=12)
        axes[0, 0].set_ylabel("Loss (log scale)", fontsize=12)
        axes[0, 0].set_title("Training Loss", fontsize=14, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3, which="both")
        axes[0, 0].legend()

        lr_batch_metrics = [m for m in metrics if m.get("type") in ("train_batch", "grpo_batch") and "lr" in m]
        lrs = [m["lr"] for m in lr_batch_metrics]
        lr_iterations = [m.get("epoch", 0) * max_iter + m.get("iteration", 0) for m in lr_batch_metrics] if lrs else []
        if lrs:
            axes[0, 1].plot(lr_iterations, lrs, color="darkorange", linewidth=2)
            axes[0, 1].set_xlabel("Iteration", fontsize=12)
            axes[0, 1].set_ylabel("LR", fontsize=12)
            axes[0, 1].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
            axes[0, 1].grid(True, alpha=0.8)

        if val_loss:
            axes[1, 0].plot(val_epochs, val_loss, marker="o", color="crimson", linewidth=2, markersize=8)
            axes[1, 0].set_xlabel("Epoch", fontsize=12)
            axes[1, 0].set_ylabel("Loss", fontsize=12)
            axes[1, 0].set_title("Validation Loss", fontsize=14, fontweight="bold")
            axes[1, 0].grid(True, alpha=0.8)

        accs = {}
        for m in val_metrics_agg:
            for k, v in m.items():
                if k.startswith("accuracy") or k == "token_accuracy" or k == "ans_parsed" or k == "acc2parse":
                    if k not in accs:
                        accs[k] = []
                    accs[k].append(v)
        
        if "ans_parsed" in accs:
            accs["parse_rate"] = accs.pop("ans_parsed")
        if "acc2parse" in accs:
            accs["accuracy_on_parsed"] = accs.pop("acc2parse")

        if accs:
            colors = ["forestgreen", "darkviolet", "darkcyan", "chocolate", "deeppink"]
            for i, (k, v) in enumerate(accs.items()):
                label = k.replace("accuracy_", "").capitalize()
                axes[1, 1].plot(val_epochs, v, marker="s", label=label, linewidth=2, color=colors[i % len(colors)])
            axes[1, 1].set_xlabel("Epoch", fontsize=12)
            axes[1, 1].set_ylabel("Accuracy", fontsize=12)
            axes[1, 1].set_title("Validation Accuracies", fontsize=14, fontweight="bold")
            axes[1, 1].set_ylim([0, 1.05])
            axes[1, 1].legend(loc="lower right", fontsize=10)
            axes[1, 1].grid(True, alpha=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_overview.png"), dpi=150)
        plt.close()

    def plot_validation_accuracy(self, metrics: List[Dict[str, Any]]):
        val_metrics_agg = self._aggregate_validation_by_epoch(metrics)
        accs = {}
        reason_accs = {}
        val_epochs = []

        for m in val_metrics_agg:
            val_epochs.append(m["epoch"])
            for k, v in m.items():
                if k.startswith("accuracy_reason_"):
                    if k not in reason_accs:
                        reason_accs[k] = []
                    reason_accs[k].append(v)
                elif k.startswith("accuracy"):
                    if k not in accs:
                        accs[k] = []
                    accs[k].append(v)

        if not accs:
            return

        plt.figure(figsize=(10, 5))
        for k, v in accs.items():
            label = k.replace("accuracy_", "").capitalize()
            plt.plot(val_epochs, v, marker="o", label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracies per Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "validation_accuracy.png"))
        plt.close()

        if reason_accs:
            self._plot_per_reason_accuracy(val_epochs, reason_accs)

    def _plot_per_reason_accuracy(self, epochs: List[int], reason_accs: Dict[str, List[float]]):
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        plt.figure(figsize=(14, 8))
        for i, (k, v) in enumerate(sorted(reason_accs.items())):
            reason_name = k.replace("accuracy_reason_", "").upper()
            color = colors[i % len(colors)]
            plt.plot(epochs[: len(v)], v, alpha=0.3, color=color)
            if len(v) > 3:
                window = min(3, len(v) // 2)
                if window > 1:
                    smoothed = np.convolve(v, np.ones(window) / window, mode="valid")
                    start = (window - 1) // 2
                    end = start + len(smoothed)
                    plt.plot(epochs[start:end], smoothed, label=reason_name, color=color, linewidth=2, marker="o", markersize=4)
                else:
                    plt.plot(epochs[: len(v)], v, label=reason_name, color=color, linewidth=2, marker="o", markersize=4)
            else:
                plt.plot(epochs[: len(v)], v, label=reason_name, color=color, linewidth=2, marker="o", markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Recall (per reason)", fontsize=12)
        plt.title("Per-Reason Accuracy Over Training", fontsize=14, fontweight="bold")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        plt.grid(True, alpha=0.8)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "validation_per_reason_accuracy.png"), dpi=150)
        plt.savefig(os.path.join(self.output_dir, "validation_reasons_correctness.png"), dpi=150)
        plt.close()

    def plot_grpo_metrics(self, metrics: List[Dict[str, Any]]):
        """Rewards from judge_logs (so plots work from first GRPO step); fall back to metrics if no judge_logs."""
        judge_series = self._load_judge_logs_series()
        scores, steps = [], []
        if judge_series:
            max_iter = max(r["iteration"] for r in judge_series) + 1
            for row in judge_series:
                scores.append(row["reward_mean"])
                steps.append(row["epoch"] * max_iter + row["iteration"])
        if not scores and metrics:
            for m in metrics:
                if m.get("type") == "grpo_batch" or (m.get("type") == "train_batch" and "reward" in m):
                    scores.append(m.get("reward", 0))
                    steps.append(m.get("step", m.get("iteration_sort", m.get("iteration", 0))))

        if not scores:
            print("  (no judge_logs and no reward in metrics, skipping GRPO reward plots)")
            return

        # Rewards during training (by iteration)
        plt.figure(figsize=(12, 6))
        plt.plot(steps, scores, alpha=0.4, color="green", label="Raw")
        if len(scores) > 10:
            ma = np.convolve(scores, np.ones(10) / 10, mode="valid")
            plt.plot(steps[9:], ma, color="darkgreen", linewidth=2, label="MA-10")
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.title("GRPO: Rewards During Training (from judge_logs)", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "grpo_rewards_over_time.png"), dpi=150)
        plt.close()

        # Legacy name for compatibility
        plt.figure(figsize=(10, 5))
        plt.plot(steps, scores, alpha=0.4)
        if len(scores) > 10:
            ma = np.convolve(scores, np.ones(10) / 10, mode="valid")
            plt.plot(steps[9:], ma, color="red", label="MA-10")
        plt.xlabel("Iteration")
        plt.ylabel("Judge Score / Reward")
        plt.title("GRPO Judge Score over Time")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "judge_score.png"))
        plt.close()

        # Mean reward per epoch (from judge_series or metrics)
        epoch_rewards: Dict[int, List[float]] = defaultdict(list)
        if judge_series:
            for row in judge_series:
                epoch_rewards[row["epoch"]].append(row["reward_mean"])
        elif metrics:
            for m in metrics:
                if (m.get("type") == "grpo_batch" or (m.get("type") == "train_batch" and "reward" in m)) and m.get("epoch") is not None:
                    epoch_rewards[m["epoch"]].append(m.get("reward", 0))
        if epoch_rewards:
            epochs_sorted = sorted(epoch_rewards.keys())
            mean_rewards = [np.mean(epoch_rewards[e]) for e in epochs_sorted]
            std_rewards = [np.std(epoch_rewards[e]) for e in epochs_sorted]
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                epochs_sorted, mean_rewards, yerr=std_rewards,
                marker="o", capsize=3, color="darkgreen", linewidth=2, markersize=6
            )
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Mean Reward", fontsize=12)
            plt.title("GRPO: Mean Reward per Epoch", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "grpo_reward_per_epoch.png"), dpi=150)
            plt.close()

    def plot_grpo_reasons_overlap(self, metrics: List[Dict[str, Any]]):
        """Plot correct_reasons_overlap from judge_logs (so it works from first GRPO step); fall back to metrics."""
        judge_series = self._load_judge_logs_series()
        overlap_vals, steps = [], []
        if judge_series:
            max_iter = max(r["iteration"] for r in judge_series) + 1
            for row in judge_series:
                if row.get("overlap_mean") is not None:
                    overlap_vals.append(row["overlap_mean"])
                    steps.append(row["epoch"] * max_iter + row["iteration"])
        if not overlap_vals and metrics:
            for m in metrics:
                if m.get("type") == "train_batch" and "correct_reasons_overlap" in m:
                    overlap_vals.append(m["correct_reasons_overlap"])
                    steps.append(m.get("step", m.get("iteration_sort", m.get("iteration", 0))))

        if not overlap_vals:
            print("  (no overlap in judge_logs and no correct_reasons_overlap in metrics, skipping)")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(steps, overlap_vals, alpha=0.5, color="teal", label="Raw")
        if len(overlap_vals) > 10:
            ma = np.convolve(overlap_vals, np.ones(10) / 10, mode="valid")
            plt.plot(steps[9:], ma, color="darkgreen", linewidth=2, label="MA-10")
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Correct Reasons Overlap", fontsize=12)
        plt.title("GRPO: Correct Reasons Overlap Over Time (from judge_logs)", fontsize=14, fontweight="bold")
        plt.ylim([0, 1.05])
        plt.legend()
        plt.grid(True, alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "grpo_correct_reasons_overlap.png"), dpi=150)
        plt.close()
        print(f"Saved GRPO correct_reasons_overlap plot to {self.output_dir}/grpo_correct_reasons_overlap.png")

    def plot_grpo_text_length_per_iteration(self):
        """Plot mean text length (chars) per iteration with Raw and MA-10."""
        judge_series = self._load_judge_logs_series()
        lengths, steps = [], []
        if judge_series:
            max_iter = max(r["iteration"] for r in judge_series) + 1
            for row in judge_series:
                if row.get("text_length_mean") is not None:
                    lengths.append(row["text_length_mean"])
                    steps.append(row["epoch"] * max_iter + row["iteration"])

        if not lengths:
            print("  (no text in judge_logs, skipping grpo_text_length_per_iteration)")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(steps, lengths, alpha=0.4, color="lightblue", linewidth=1, label="Raw")
        if len(lengths) > 10:
            ma = np.convolve(lengths, np.ones(10) / 10, mode="valid")
            plt.plot(steps[9:], ma, color="steelblue", linewidth=2, label="MA-10")
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Mean Text Length (chars)", fontsize=12)
        plt.title("GRPO: Text Length per Iteration", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "grpo_text_length_per_iteration.png"), dpi=150)
        plt.close()
        print(f"Saved GRPO text length per iteration to {self.output_dir}/grpo_text_length_per_iteration.png")

    def plot_grpo_correctness_distribution(self):
        """Plot distribution of how many generations were correct per sample (0 to num_generations)."""
        judge_dir = os.path.join(self.run_dir, "judge_logs")
        if not os.path.isdir(judge_dir):
            return

        json_files = sorted(glob.glob(os.path.join(judge_dir, "*.json")))
        if not json_files:
            return

        correct_counts = []  # for each sample: number of correct generations (0 .. num_generations)
        num_generations = None
        for fpath in json_files:
            if "skeptic_" in os.path.basename(fpath):
                continue  # skeptic_epoch has different format: audio_id, rollouts, has_passed
            try:
                with open(fpath, "r") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(logs, list):
                continue
            for sample in logs:
                gens = sample.get("generations", [])
                if not gens:
                    continue
                if num_generations is None:
                    num_generations = len(gens)
                n_correct = sum(1 for g in gens if g.get("is_correct") is True)
                correct_counts.append(n_correct)

        if not correct_counts or num_generations is None:
            return

        bins = np.arange(-0.5, num_generations + 1.5, 1)
        hist, _ = np.histogram(correct_counts, bins=bins)
        x_labels = [str(k) for k in range(num_generations + 1)]
        x_pos = np.arange(num_generations + 1)

        plt.figure(figsize=(10, 6))
        plt.bar(x_pos, hist, color="steelblue", edgecolor="navy", alpha=0.85)
        plt.xlabel("Number of correct generations per sample", fontsize=12)
        plt.ylabel("Number of samples", fontsize=12)
        plt.title("GRPO: Distribution of correct generations per sample", fontsize=14, fontweight="bold")
        plt.xticks(x_pos, x_labels)
        plt.grid(True, alpha=0.5, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "grpo_correctness_distribution.png"), dpi=150)
        plt.close()
        print(f"Saved GRPO correctness distribution to {self.output_dir}/grpo_correctness_distribution.png")

    def plot_grpo_correctness_mean_per_epoch(self):
        """Plot mean number of correct generations per sample, epoch by epoch (like correctness distribution but over time)."""
        judge_dir = os.path.join(self.run_dir, "judge_logs")
        if not os.path.isdir(judge_dir):
            print("  (no judge_logs/, skipping grpo_correctness_mean_per_epoch)")
            return

        # Parse filenames: epoch_{e}_iter_{i}[_rank{N}].json only (skeptic_epoch has different format)
        pattern = re.compile(r"^epoch_(\d+)_iter_\d+(?:_rank\d+)?\.json$")
        epoch_correct_counts: Dict[int, List[float]] = defaultdict(list)

        for fname in os.listdir(judge_dir):
            if not fname.endswith(".json") or fname.startswith("skeptic_"):
                continue
            m = pattern.search(fname)
            if not m:
                continue
            epoch = int(m.group(1))
            fpath = os.path.join(judge_dir, fname)
            try:
                with open(fpath, "r") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(logs, list):
                continue
            for sample in logs:
                gens = sample.get("generations", [])
                if not gens:
                    continue
                n_correct = sum(1 for g in gens if g.get("is_correct") is True)
                epoch_correct_counts[epoch].append(float(n_correct))

        if not epoch_correct_counts:
            print("  (no valid judge JSON in judge_logs/, skipping grpo_correctness_mean_per_epoch)")
            return

        epochs_sorted = sorted(epoch_correct_counts.keys())
        means = [np.mean(epoch_correct_counts[e]) for e in epochs_sorted]
        stds = [np.std(epoch_correct_counts[e]) for e in epochs_sorted]
        # Max correct = num_generations (from judge log structure)
        num_gens = int(max(max(v) for v in epoch_correct_counts.values())) if epoch_correct_counts else 4

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            epochs_sorted, means, yerr=stds,
            marker="o", capsize=3, color="steelblue", linewidth=2, markersize=6
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Mean correct generations per sample", fontsize=12)
        plt.title("GRPO: Mean Correctness per Epoch (0–{} correct per sample)".format(num_gens), fontsize=14, fontweight="bold")
        plt.ylim(-0.1, num_gens + 0.5)
        plt.grid(True, alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "grpo_correctness_mean_per_epoch.png"), dpi=150)
        plt.close()
        print(f"Saved GRPO correctness mean per epoch to {self.output_dir}/grpo_correctness_mean_per_epoch.png")

    def plot_grpo_advantages_distribution(self):
        """Plot overall distribution of GRPO advantages (from judge_logs)."""
        judge_dir = os.path.join(self.run_dir, "judge_logs")
        if not os.path.isdir(judge_dir):
            return

        advantages = []
        for fname in os.listdir(judge_dir):
            if not fname.endswith(".json") or fname.startswith("skeptic_"):
                continue
            if not re.match(r"^epoch_\d+_iter_\d+(?:_rank\d+)?\.json$", fname):
                continue
            fpath = os.path.join(judge_dir, fname)
            try:
                with open(fpath, "r") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(logs, list):
                continue
            for sample in logs:
                for g in sample.get("generations", []):
                    if "advantage" in g:
                        advantages.append(float(g["advantage"]))

        if not advantages:
            print("  (no advantage in judge_logs, skipping grpo_advantages_distribution)")
            return

        advantages = np.asarray(advantages)
        plt.figure(figsize=(10, 6))
        plt.hist(advantages, bins=50, color="steelblue", edgecolor="navy", alpha=0.85)
        plt.axvline(np.mean(advantages), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(advantages):.4f}")
        plt.axvline(np.median(advantages), color="orange", linestyle=":", linewidth=2, label=f"Median: {np.median(advantages):.4f}")
        plt.xlabel("Advantage", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title("GRPO: Overall Advantages Distribution", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.5, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "grpo_advantages_distribution.png"), dpi=150)
        plt.close()
        print(f"Saved GRPO advantages distribution to {self.output_dir}/grpo_advantages_distribution.png")

    def plot_grpo_advantages_distribution_per_epoch(self):
        """Plot advantages distribution per epoch (violin plot) to see how it changes over training."""
        judge_dir = os.path.join(self.run_dir, "judge_logs")
        if not os.path.isdir(judge_dir):
            return

        pattern = re.compile(r"^epoch_(\d+)_iter_\d+(?:_rank\d+)?\.json$")
        epoch_advantages: Dict[int, List[float]] = defaultdict(list)

        for fname in os.listdir(judge_dir):
            if not fname.endswith(".json") or fname.startswith("skeptic_"):
                continue
            m = pattern.search(fname)
            if not m:
                continue
            epoch = int(m.group(1))
            fpath = os.path.join(judge_dir, fname)
            try:
                with open(fpath, "r") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(logs, list):
                continue
            for sample in logs:
                for g in sample.get("generations", []):
                    if "advantage" in g:
                        epoch_advantages[epoch].append(float(g["advantage"]))

        if not epoch_advantages:
            print("  (no advantage in judge_logs, skipping grpo_advantages_distribution_per_epoch)")
            return

        epochs_sorted = sorted(epoch_advantages.keys())
        data = [epoch_advantages[ep] for ep in epochs_sorted]
        positions = list(range(len(epochs_sorted)))

        plt.figure(figsize=(14, 6))
        parts = plt.violinplot(
            data,
            positions=positions,
            showmeans=True,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("lightsteelblue")
            pc.set_edgecolor("steelblue")
            pc.set_alpha(0.8)

        plt.axhline(0.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        plt.xticks(positions, [str(ep) for ep in epochs_sorted])
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Advantage", fontsize=12)
        plt.title("GRPO: Advantages Distribution by Epoch", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.5, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "grpo_advantages_distribution_per_epoch.png"), dpi=150)
        plt.close()
        print(f"Saved GRPO advantages distribution per epoch to {self.output_dir}/grpo_advantages_distribution_per_epoch.png")

    def plot_grpo_skeptic_filtering(self, metrics: List[Dict[str, Any]]):
        """Plot skeptic mode: samples rejected per GRPO batch iteration, and rejected/accepted ratio per epoch."""
        # Per-iteration: skeptic_rejected_per_check_batch, skeptic_accepted_per_check_batch
        grpo_batches = [
            m for m in metrics
            if (m.get("type") == "grpo_batch" or (m.get("type") == "train_batch" and "reward" in m))
            and ("skeptic_rejected_per_check_batch" in m or "skeptic_accepted_per_check_batch" in m)
        ]
        if grpo_batches:
            iterations = [m.get("iteration", i) for i, m in enumerate(grpo_batches)]
            rejected = [m.get("skeptic_rejected_per_check_batch", 0) for m in grpo_batches]
            accepted = [m.get("skeptic_accepted_per_check_batch", 0) for m in grpo_batches]
            plt.figure(figsize=(12, 6))
            plt.bar([i - 0.2 for i in range(len(iterations))], rejected, width=0.4, label="Rejected", color="coral", alpha=0.8)
            plt.bar([i + 0.2 for i in range(len(iterations))], accepted, width=0.4, label="Accepted", color="seagreen", alpha=0.8)
            plt.xlabel("GRPO Batch Iteration", fontsize=12)
            plt.ylabel("Samples (last 10 check batches)", fontsize=12)
            plt.title("GRPO Skeptic: Rejected vs Accepted Samples per Batch Iteration", fontsize=14, fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.5, axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "grpo_skeptic_rejected_per_iteration.png"), dpi=150)
            plt.close()
            print(f"Saved skeptic rejected per iteration to {self.output_dir}/grpo_skeptic_rejected_per_iteration.png")

        # Per-epoch: skeptic_epoch_rejected, skeptic_epoch_accepted, ratio
        skeptic_epochs = [m for m in metrics if m.get("type") == "skeptic_epoch"]
        if skeptic_epochs:
            epochs = [m["epoch"] for m in skeptic_epochs]
            rejected = [m.get("skeptic_epoch_rejected", 0) for m in skeptic_epochs]
            accepted = [m.get("skeptic_epoch_accepted", 0) for m in skeptic_epochs]
            ratios = [
                r / a if a > 0 else float("nan")
                for r, a in zip(rejected, accepted)
            ]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].bar([e - 0.2 for e in epochs], rejected, width=0.4, label="Rejected", color="coral", alpha=0.8)
            axes[0].bar([e + 0.2 for e in epochs], accepted, width=0.4, label="Accepted", color="seagreen", alpha=0.8)
            axes[0].set_xlabel("Epoch", fontsize=12)
            axes[0].set_ylabel("Samples", fontsize=12)
            axes[0].set_title("Skeptic: Rejected vs Accepted per Epoch", fontsize=14, fontweight="bold")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis="y")

            valid_ratios = [(e, r) for e, r in zip(epochs, ratios) if not (r != r or np.isinf(r))]
            if valid_ratios:
                ep, rat = zip(*valid_ratios)
                axes[1].plot(ep, rat, marker="o", color="darkviolet", linewidth=2, markersize=8)
                axes[1].set_xlabel("Epoch", fontsize=12)
                axes[1].set_ylabel("Rejected / Accepted Ratio", fontsize=12)
                axes[1].set_title("Skeptic: Rejected/Accepted Ratio per Epoch", fontsize=14, fontweight="bold")
                axes[1].grid(True, alpha=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "grpo_skeptic_rejected_accepted_ratio_per_epoch.png"), dpi=150)
            plt.close()
            print(f"Saved skeptic ratio per epoch to {self.output_dir}/grpo_skeptic_rejected_accepted_ratio_per_epoch.png")

    def save_sample_generations(self):
        sample_files = glob.glob(os.path.join(self.run_dir, "samples_validation_epoch_*.jsonl"))
        if not sample_files:
            return

        sample_files.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".")[0]))
        latest_file = sample_files[-1]

        output_html = os.path.join(self.output_dir, "samples.html")
        with open(output_html, "w") as f:
            f.write("<html><body><h1>Latest Validation Samples</h1><table border='1'><tr><th>GT</th><th>Output</th></tr>")
            with open(latest_file, "r") as sf:
                for line in sf:
                    data = json.loads(line)
                    gt = data.get("gt", "").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    out = data.get("output", "").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    f.write(f"<tr><td>{gt}</td><td>{out}</td></tr>")
            f.write("</table></body></html>")
        print(f"Saved sample generations to {output_html}")

    def plot_confidence_metrics(self, metrics: List[Dict[str, Any]]):
        val_metrics_agg = self._aggregate_validation_by_epoch(metrics)
        val_epochs = []

        # Detect format: check if SASV metrics exist
        is_sasv = False
        for m in val_metrics_agg:
            if any(key in m for key in ["confidence_yes_mean", "confidence_no_mean", "confidence_gen_mean"]):
                is_sasv = True
                break

        if is_sasv:
            # SASV format: three classes - show correct/incorrect for each predicted class
            confidence_metrics = {
                "yes_correct": [], "yes_incorrect": [],
                "no_correct": [], "no_incorrect": [],
                "gen_correct": [], "gen_incorrect": []
            }
            for m in val_metrics_agg:
                val_epochs.append(m["epoch"])
                for key in ["yes", "no", "gen"]:
                    confidence_metrics[f"{key}_correct"].append(m.get(f"confidence_pred_{key}_correct_mean"))
                    confidence_metrics[f"{key}_incorrect"].append(m.get(f"confidence_pred_{key}_incorrect_mean"))

            if not any(any(v is not None for v in vals) for vals in confidence_metrics.values()):
                return

            base_colors = {"yes": "#2ecc71", "no": "#e74c3c", "gen": "#f39c12"}
            labels = {
                "yes": "Predicted Yes",
                "no": "Predicted No",
                "gen": "Predicted Gen",
            }

            plt.figure(figsize=(12, 7))
            for key in ["yes", "no", "gen"]:
                # Correct line (darker, solid)
                correct_values = confidence_metrics[f"{key}_correct"]
                valid_points = [(e, v) for e, v in zip(val_epochs, correct_values) if v is not None]
                if valid_points:
                    epochs_filtered, values_filtered = zip(*valid_points)
                    plt.plot(epochs_filtered, values_filtered, marker="o", 
                            label=f"{labels[key]} (Correct)", color=base_colors[key], 
                            linewidth=2.5, markersize=8, linestyle="-")
                
                # Incorrect line (lighter, dashed)
                incorrect_values = confidence_metrics[f"{key}_incorrect"]
                valid_points = [(e, v) for e, v in zip(val_epochs, incorrect_values) if v is not None]
                if valid_points:
                    epochs_filtered, values_filtered = zip(*valid_points)
                    # Lighten color for incorrect
                    light_color = mcolors.to_rgba(base_colors[key], alpha=0.6)
                    plt.plot(epochs_filtered, values_filtered, marker="s", 
                            label=f"{labels[key]} (Incorrect)", color=light_color, 
                            linewidth=2, markersize=6, linestyle="--")
            
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Average Confidence", fontsize=12)
            plt.title("Model Confidence by Predicted Class (SASV)", fontsize=14, fontweight="bold")
            plt.ylim([0, 1.05])
            plt.legend(loc="best", fontsize=10)
            plt.grid(True, alpha=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "confidence_by_type.png"), dpi=150)
            plt.close()
            print(f"Saved confidence plot to {self.output_dir}/confidence_by_type.png")
        else:
            # Antispoofing format: show correct/incorrect for each predicted class
            confidence_metrics = {
                "fake_correct": [], "fake_incorrect": [],
                "real_correct": [], "real_incorrect": []
            }
            for m in val_metrics_agg:
                val_epochs.append(m["epoch"])
                confidence_metrics["fake_correct"].append(m.get("confidence_pred_fake_correct_mean"))
                confidence_metrics["fake_incorrect"].append(m.get("confidence_pred_fake_incorrect_mean"))
                confidence_metrics["real_correct"].append(m.get("confidence_pred_real_correct_mean"))
                confidence_metrics["real_incorrect"].append(m.get("confidence_pred_real_incorrect_mean"))

            if not any(any(v is not None for v in vals) for vals in confidence_metrics.values()):
                return

            base_colors = {"fake": "#e74c3c", "real": "#3498db"}
            labels = {
                "fake": "Predicted Fake",
                "real": "Predicted Real",
            }

            plt.figure(figsize=(12, 7))
            for key in ["fake", "real"]:
                # Correct line (darker, solid)
                correct_values = confidence_metrics[f"{key}_correct"]
                valid_points = [(e, v) for e, v in zip(val_epochs, correct_values) if v is not None]
                if valid_points:
                    epochs_filtered, values_filtered = zip(*valid_points)
                    plt.plot(epochs_filtered, values_filtered, marker="o", 
                            label=f"{labels[key]} (Correct)", color=base_colors[key], 
                            linewidth=2.5, markersize=8, linestyle="-")
                
                # Incorrect line (lighter, dashed)
                incorrect_values = confidence_metrics[f"{key}_incorrect"]
                valid_points = [(e, v) for e, v in zip(val_epochs, incorrect_values) if v is not None]
                if valid_points:
                    epochs_filtered, values_filtered = zip(*valid_points)
                    # Lighten color for incorrect
                    light_color = mcolors.to_rgba(base_colors[key], alpha=0.6)
                    plt.plot(epochs_filtered, values_filtered, marker="s", 
                            label=f"{labels[key]} (Incorrect)", color=light_color, 
                            linewidth=2, markersize=6, linestyle="--")
            
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Average Confidence", fontsize=12)
            plt.title("Model Confidence by Predicted Class", fontsize=14, fontweight="bold")
            plt.ylim([0, 1.05])
            plt.legend(loc="best", fontsize=10)
            plt.grid(True, alpha=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "confidence_by_type.png"), dpi=150)
            plt.close()
            print(f"Saved confidence plot to {self.output_dir}/confidence_by_type.png")


    def plot_sasv_subsystem_accuracies(self, metrics: List[Dict[str, Any]]):
        """Plot ASV accuracy (yes/no) and counter-measure accuracy (bonafide vs spoof)."""
        sample_files = glob.glob(os.path.join(self.run_dir, "samples_validation_epoch_*.jsonl"))
        if not sample_files:
            return

        epochs, asv_accs, cm_accs = [], [], []
        for fpath in sorted(sample_files, key=lambda p: int(p.split("_epoch_")[-1].split(".")[0])):
            epoch = int(fpath.split("_epoch_")[-1].split(".")[0])
            samples = []
            with open(fpath, "r") as f:
                for line in f:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            accs = compute_sasv_subsystem_accuracies(samples)
            if "asv_accuracy" not in accs:
                continue
            epochs.append(epoch)
            asv_accs.append(accs["asv_accuracy"])
            cm_accs.append(accs.get("cm_accuracy"))

        if not epochs:
            return

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, asv_accs, marker="o", linewidth=2, label="ASV accuracy (yes/no)")
        if any(v is not None for v in cm_accs):
            plt.plot(epochs, cm_accs, marker="s", linewidth=2, label="CM accuracy (counter-measure)")
        plt.ylim([0, 1.05])
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("SASV Subsystem Accuracies", fontsize=14, fontweight="bold")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "validation_asv_cm_accuracy.png"), dpi=150)
        plt.close()
        print(f"Saved SASV subsystem accuracy plot to {self.output_dir}/validation_asv_cm_accuracy.png")

        val_metrics_agg = self._aggregate_validation_by_epoch(metrics)
        sv_keys = ["sv_eer", "sasv_sv_eer", "eer_sv"]
        spf_keys = ["spf_eer", "sasv_spf_eer", "eer_spf"]
        sv_epochs, sv_vals, spf_epochs, spf_vals = [], [], [], []
        for m in val_metrics_agg:
            ep = int(m["epoch"])
            for key in sv_keys:
                if key in m:
                    sv_epochs.append(ep)
                    sv_vals.append(m[key])
                    break
            for key in spf_keys:
                if key in m:
                    spf_epochs.append(ep)
                    spf_vals.append(m[key])
                    break
        if sv_vals or spf_vals:
            plt.figure(figsize=(10, 5))
            if sv_vals:
                plt.plot(sv_epochs, sv_vals, marker="o", linewidth=2, label="SV-EER")
            if spf_vals:
                plt.plot(spf_epochs, spf_vals, marker="s", linewidth=2, label="SPF-EER (CM)")
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("EER", fontsize=12)
            plt.title("ASV / Counter-Measure EER", fontsize=14, fontweight="bold")
            plt.legend(loc="best")
            plt.grid(True, alpha=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "validation_asv_cm_eer.png"), dpi=150)
            plt.close()
            print(f"Saved ASV/CM EER plot to {self.output_dir}/validation_asv_cm_eer.png")


    def generate_plots(self):
        """Generate all plots. Metric-based plots need metrics.jsonl; judge-based plots use judge_logs and can run from first GRPO step (no eval epoch required)."""
        metrics = self.load_metrics()
        if not metrics:
            print(f"No metrics.jsonl in {self.run_dir}; drawing only judge-based and file-based plots.")

        if metrics:
            self.plot_learning_rate(metrics)
            self.plot_losses(metrics)
            self.plot_validation_accuracy(metrics)
            self.plot_sasv_subsystem_accuracies(metrics)
            self.plot_confidence_metrics(metrics)
        self.plot_grpo_metrics(metrics if metrics else [])
        self.plot_grpo_reasons_overlap(metrics if metrics else [])
        self.plot_grpo_text_length_per_iteration()
        self.plot_grpo_correctness_distribution()
        self.plot_grpo_correctness_mean_per_epoch()
        self.plot_grpo_advantages_distribution()
        self.plot_grpo_advantages_distribution_per_epoch()
        self.plot_grpo_skeptic_filtering(metrics if metrics else [])
        self.save_sample_generations()
