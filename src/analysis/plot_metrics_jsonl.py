#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plotter_common import compute_sasv_subsystem_accuracies


LOSS_SPECS = [
    ("loss_ce1", "L_CE1"),
    ("loss_ce2", "L_CE2"),
    ("loss_arcface", "L_ARCFACE"),
]


def _get_series_for_key(train_batch, x, key):
    if key == "loss_ce1":
        y = []
        xk = []
        for i, m in enumerate(train_batch):
            if "loss_ce1" in m:
                y.append(m["loss_ce1"])
                xk.append(x[i])
            elif "loss_ce" in m:
                y.append(m["loss_ce"])
                xk.append(x[i])
        return xk, y

    y = [m.get(key) for m in train_batch if key in m]
    xk = [x[i] for i, m in enumerate(train_batch) if key in m]
    return xk, y


def load_metrics(path: Path):
    metrics = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return metrics


def get_train_batch(metrics):
    return [m for m in metrics if m.get("type") == "train_batch" and "iteration" in m]


def get_validation(metrics):
    return [m for m in metrics if m.get("type") == "validation" and "epoch" in m]


def global_iterations(train_batch):
    if not train_batch:
        return []
    max_iter = max(int(m.get("iteration", 0)) for m in train_batch) + 1
    return [int(m.get("epoch", 0)) * max_iter + int(m.get("iteration", 0)) for m in train_batch]


def moving_average(values, window=20):
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return None
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_losses(train_batch, out_dir: Path):
    x = global_iterations(train_batch)

    plt.figure(figsize=(12, 6))
    colors = ["royalblue", "seagreen", "mediumpurple", "darkorange"]
    for (key, label), color in zip(LOSS_SPECS, colors):
        xk, y = _get_series_for_key(train_batch, x, key)
        if not y:
            continue
        plt.plot(xk, y, alpha=0.35, linewidth=1.2, color=color, label=f"{label} (raw)")
        ma = moving_average(y, window=20)
        if ma is not None:
            plt.plot(xk[19:], ma, linewidth=2.0, color=color, label=f"{label} (MA20)")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Losses (all components)")
    plt.grid(alpha=0.3)
    if plt.gca().has_data():
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "losses_all.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(len(LOSS_SPECS), 1, figsize=(12, 15), sharex=True)
    for idx, ((key, label), color) in enumerate(zip(LOSS_SPECS, colors)):
        xk, y = _get_series_for_key(train_batch, x, key)
        if not y:
            axes[idx].set_title(f"{label} (no data)")
            continue
        axes[idx].plot(xk, y, alpha=0.35, linewidth=1.0, color=color, label="Raw")
        ma = moving_average(y, window=20)
        if ma is not None:
            axes[idx].plot(xk[19:], ma, linewidth=2.0, color=color, label="MA20")
        axes[idx].set_ylabel(label)
        axes[idx].grid(alpha=0.3)
        axes[idx].legend()

    axes[-1].set_xlabel("Iteration")
    plt.suptitle("Per-loss Curves", y=0.995)
    plt.tight_layout()
    plt.savefig(out_dir / "losses_split.png", dpi=150)
    plt.close()


def plot_learning_rate(train_batch, out_dir: Path):
    x = global_iterations(train_batch)
    y = [m.get("lr") for m in train_batch if "lr" in m]
    xk = [x[i] for i, m in enumerate(train_batch) if "lr" in m]
    if not y:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(xk, y, color="purple", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_rate.png", dpi=150)
    plt.close()


def plot_validation(validation, out_dir: Path):
    if not validation:
        return

    validation = sorted(validation, key=lambda m: int(m.get("epoch", 0)))
    epochs = [int(m["epoch"]) for m in validation]

    val_loss = [m.get("loss") for m in validation if "loss" in m]
    val_loss_epochs = [int(m["epoch"]) for m in validation if "loss" in m]
    if val_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(val_loss_epochs, val_loss, marker="o", linewidth=2, color="crimson")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "validation_loss.png", dpi=150)
        plt.close()

    acc_keys = [
        ("accuracy", "accuracy"),
        ("accuracy_balanced", "accuracy_balanced"),
        ("accuracy_yes", "accuracy_yes"),
        ("accuracy_no", "accuracy_no"),
        ("accuracy_gen", "accuracy_gen"),
        ("asv_accuracy", "asv_accuracy"),
        ("cm_accuracy", "cm_accuracy"),
        ("token_accuracy", "token_accuracy"),
        ("acc2parse", "accuracy_on_parsed"),
        ("ans_parsed", "parse_rate"),
    ]
    plt.figure(figsize=(12, 6))
    plotted = False
    for key, label in acc_keys:
        y = [m.get(key) for m in validation if key in m]
        xk = [int(m["epoch"]) for m in validation if key in m]
        if y:
            plt.plot(xk, y, marker="o", linewidth=1.8, label=label)
            plotted = True
    if plotted:
        plt.ylim(0.0, 1.05)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy Metrics")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "validation_accuracy_metrics.png", dpi=150)
    plt.close()

    eer_keys = ["sv_eer", "spf_eer", "sasv_sv_eer", "sasv_spf_eer", "eer_sasv", "eer_sv", "eer_spf", "min_a_dcf", "min_t_dcf", "t_eer", "t_eer_pct"]
    plt.figure(figsize=(10, 5))
    plotted = False
    for key in eer_keys:
        y = [m.get(key) for m in validation if key in m]
        xk = [int(m["epoch"]) for m in validation if key in m]
        if y:
            plt.plot(xk, y, marker="s", linewidth=1.8, label=key)
            plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("EER")
        plt.title("Validation EER Metrics")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "validation_eer_metrics.png", dpi=150)
    plt.close()



def _resolve_metric_series(validation, keys):
    for key in keys:
        y = [m.get(key) for m in validation if key in m]
        xk = [int(m["epoch"]) for m in validation if key in m]
        if y:
            return key, xk, y
    return None, [], []


def _load_samples_by_epoch(run_dir: Path):
    samples_dir = run_dir if (run_dir / "samples_validation_epoch_0.jsonl").exists() else run_dir / "logs"
    if not samples_dir.exists():
        return {}

    by_epoch = {}
    for path in sorted(samples_dir.glob("samples_validation_epoch_*.jsonl")):
        try:
            epoch = int(path.stem.split("_epoch_")[-1])
        except ValueError:
            continue
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            by_epoch[epoch] = rows
    return by_epoch


def plot_sasv_subsystem_accuracies(validation, metrics_path: Path, out_dir: Path):
    """Plot ASV accuracy (yes/no) and counter-measure accuracy (bonafide vs spoof)."""
    by_epoch = _load_samples_by_epoch(metrics_path.parent)
    epochs, asv_accs, cm_accs = [], [], []

    for epoch in sorted(by_epoch):
        accs = compute_sasv_subsystem_accuracies(by_epoch[epoch])
        if "asv_accuracy" in accs:
            epochs.append(epoch)
            asv_accs.append(accs["asv_accuracy"])
            cm_accs.append(accs.get("cm_accuracy"))

    if not epochs:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, asv_accs, marker="o", linewidth=2, label="ASV accuracy (yes/no)")
    if any(v is not None for v in cm_accs):
        plt.plot(
            epochs,
            cm_accs,
            marker="s",
            linewidth=2,
            label="CM accuracy (counter-measure)",
        )
    plt.ylim(0.0, 1.05)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SASV Subsystem Accuracies")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "validation_asv_cm_accuracy.png", dpi=150)
    plt.close()

    # Also overlay with logged EER metrics when present.
    validation = sorted(validation, key=lambda m: int(m.get("epoch", 0)))
    _, sv_epochs, sv_eer = _resolve_metric_series(
        validation, ["sv_eer", "sasv_sv_eer", "eer_sv"]
    )
    _, spf_epochs, spf_eer = _resolve_metric_series(
        validation, ["spf_eer", "sasv_spf_eer", "eer_spf"]
    )
    if sv_eer or spf_eer:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        if sv_eer:
            ax1.plot(sv_epochs, sv_eer, marker="o", color="tab:blue", label="SV-EER")
        if spf_eer:
            ax1.plot(spf_epochs, spf_eer, marker="s", color="tab:orange", label="SPF-EER (CM)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("EER")
        ax1.set_title("ASV / Counter-Measure EER")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / "validation_asv_cm_eer.png", dpi=150)
        plt.close(fig)

    conf_keys = ["confidence_mean", "confidence_correct_mean", "confidence_incorrect_mean"]
    plt.figure(figsize=(10, 5))
    plotted = False
    for key in conf_keys:
        y = [m.get(key) for m in validation if key in m]
        xk = [int(m["epoch"]) for m in validation if key in m]
        if y:
            plt.plot(xk, y, marker="^", linewidth=1.8, label=key)
            plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Confidence")
        plt.title("Validation Confidence Metrics")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "validation_confidence_metrics.png", dpi=150)
    plt.close()


def write_summary(metrics, train_batch, validation, out_dir: Path):
    lines = []
    lines.append(f"total_records: {len(metrics)}")
    lines.append(f"train_batch_records: {len(train_batch)}")
    lines.append(f"validation_records: {len(validation)}")

    for key, label in LOSS_SPECS:
        _, vals = _get_series_for_key(train_batch, list(range(len(train_batch))), key)
        vals = [float(v) for v in vals]
        if vals:
            lines.append(
                f"{label}: start={vals[0]:.6f}, end={vals[-1]:.6f}, min={min(vals):.6f}, max={max(vals):.6f}"
            )

    for key in ["accuracy", "accuracy_balanced", "asv_accuracy", "cm_accuracy", "sv_eer", "spf_eer", "min_a_dcf", "min_t_dcf", "t_eer"]:
        vals = [float(m[key]) for m in validation if key in m]
        if vals:
            lines.append(
                f"{key}: start={vals[0]:.6f}, end={vals[-1]:.6f}, best={max(vals):.6f}, worst={min(vals):.6f}"
            )

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from metrics.jsonl")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for plots")
    args = parser.parse_args()

    metrics_path = args.metrics
    output_dir = args.output_dir or (metrics_path.parent / "plots_metrics_jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_path)
    train_batch = get_train_batch(metrics)
    validation = get_validation(metrics)

    if not metrics:
        raise ValueError(f"No valid json records found in {metrics_path}")

    plot_losses(train_batch, output_dir)
    plot_learning_rate(train_batch, output_dir)
    plot_validation(validation, output_dir)
    plot_sasv_subsystem_accuracies(validation, metrics_path, output_dir)
    write_summary(metrics, train_batch, validation, output_dir)

    print(f"Saved plots and summary to: {output_dir}")


if __name__ == "__main__":
    main()
