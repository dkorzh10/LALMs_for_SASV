"""
Test run plotting: metrics summary, prediction distribution, histograms, confidence.
Supports loading from predictions file (or distributed rank files) or metrics.jsonl fallback.
"""
import os
import re
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .plotter_common import extract_answer, extract_answer_from_gt, extract_pred_answer


def _epoch_from_path(p: str) -> int:
    """Extract epoch number from path like predictions_validation_epoch_31_rank0.jsonl."""
    m = re.search(r"epoch_(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 0


def _load_predictions_from_files(log_dir: str) -> List[Dict]:
    """
    Load predictions from file(s).
    Supports: predictions_test_epoch_*.jsonl, samples_test_epoch_*.jsonl,
    predictions_validation_epoch_*.jsonl, samples_validation_epoch_*.jsonl,
    and distributed: *_rank*.jsonl (merged in rank order).
    Uses only the latest epoch's files (not all epochs).
    """
    # Single file pattern - test first, then validation
    pred_files = glob.glob(os.path.join(log_dir, "predictions_test_epoch_*.jsonl"))
    if not pred_files:
        pred_files = glob.glob(os.path.join(log_dir, "samples_test_epoch_*.jsonl"))
    if not pred_files:
        pred_files = glob.glob(os.path.join(log_dir, "predictions_validation_epoch_*.jsonl"))
    if not pred_files:
        pred_files = glob.glob(os.path.join(log_dir, "samples_validation_epoch_*.jsonl"))

    if not pred_files:
        return []

    # Use only the latest epoch (same for validation and test)
    latest_epoch = max(_epoch_from_path(p) for p in pred_files)
    pred_files = [p for p in pred_files if _epoch_from_path(p) == latest_epoch]

    rank_files = [f for f in pred_files if "_rank" in os.path.basename(f)]
    non_rank_files = [f for f in pred_files if "_rank" not in os.path.basename(f)]

    files_to_use = []
    if rank_files:
        def rank_key(p):
            m = re.search(r"rank(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else 0
        rank_files.sort(key=rank_key)
        files_to_use = rank_files
    elif non_rank_files:
        # All are same epoch after filter; take any (e.g. only one exists)
        files_to_use = [non_rank_files[0]]

    if not files_to_use:
        return []

    predictions = []
    for fpath in files_to_use:
        with open(fpath, "r") as f:
            for line in f:
                try:
                    predictions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return predictions


def _load_test_metrics_from_jsonl(log_dir: str) -> Optional[Dict]:
    """Load aggregated test metrics from metrics.jsonl when predictions file is absent."""
    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return None

    test_entries = []
    with open(metrics_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("type") == "test":
                    test_entries.append(data)
            except json.JSONDecodeError:
                pass

    if not test_entries:
        return None
    # Use latest (last) test entry
    return test_entries[-1]


def load_test_data(log_dir: str) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    """
    Load test data. Prefer predictions file; fall back to metrics.jsonl.
    Returns (predictions, metrics_from_jsonl).
    - If predictions available: (list, None) - full per-sample data
    - If only metrics.jsonl: (None, dict) - aggregated metrics only
    - If nothing: (None, None)
    """
    predictions = _load_predictions_from_files(log_dir)
    if predictions:
        return predictions, None

    metrics = _load_test_metrics_from_jsonl(log_dir)
    if metrics:
        return None, metrics
    return None, None


def compute_test_metrics(predictions: List[Dict]) -> Dict[str, Any]:
    """Compute classification metrics from predictions.
    Supports both antispoofing (Real/Fake) and SASV (yes/no/gen) formats.
    """
    y_true = []
    y_pred = []
    y_scores = []
    
    # Track statistics for better error reporting
    total = len(predictions)
    valid_gt_count = 0
    valid_output_count = 0
    invalid_output_samples = []
    
    # Detect format from first prediction
    is_sasv = False
    if predictions:
        first_gt = extract_answer_from_gt(predictions[0].get("gt", "")).lower()
        if first_gt in ["yes", "no", "gen", "verified", "rejected", "spoof"]:
            is_sasv = True

    for p in predictions:
        gt_raw = extract_answer_from_gt(p.get("gt", ""))
        out_raw = extract_pred_answer(p.get("output", ""))
        gt = gt_raw.lower()
        out = out_raw.lower()
        
        # Normalize GT for SASV
        if is_sasv:
            if gt == "verified":
                gt = "yes"
            elif gt == "rejected":
                gt = "no"
            elif gt == "spoof":
                gt = "gen"
            
            if gt not in ["yes", "no", "gen"]:
                continue
            valid_gt_count += 1
            
            # Map to numeric labels
            label_map = {"yes": 0, "no": 1, "gen": 2}
            label_numeric = label_map.get(gt, -1)
            
            if out in ["yes", "no", "gen"]:
                pred_numeric = label_map.get(out, -1)
                valid_output_count += 1
            else:
                # Collect sample of invalid outputs for debugging
                raw_output = p.get("output", "")
                if len(invalid_output_samples) < 3 and raw_output:
                    invalid_output_samples.append(raw_output[:100])
                continue
            
            # Score: use probability of predicted class
            if "yes_prob" in p and out == "yes":
                score = float(p.get("yes_prob", 0.33))
            elif "no_prob" in p and out == "no":
                score = float(p.get("no_prob", 0.33))
            elif "gen_prob" in p and out == "gen":
                score = float(p.get("gen_prob", 0.33))
            elif "confidence" in p and p["confidence"] is not None:
                score = float(p["confidence"])
            else:
                score = 0.33
            
            y_true.append(label_numeric)
            y_pred.append(pred_numeric)
            y_scores.append(float(score))
        else:
            # Antispoofing format
            if not gt or gt not in ["real", "fake"]:
                continue
            valid_gt_count += 1

            label_binary = 1 if gt == "fake" else 0

            if out in ["real", "fake"]:
                pred_binary = 1 if out == "fake" else 0
                valid_output_count += 1
            else:
                # Collect sample of invalid outputs for debugging
                raw_output = p.get("output", "")
                if len(invalid_output_samples) < 3 and raw_output:
                    invalid_output_samples.append(raw_output[:100])
                continue

            # Score for EER/AUC: P(Fake)
            if "fake_prob" in p and p["fake_prob"] is not None:
                score = float(p["fake_prob"])
            elif "confidence" in p and p["confidence"] is not None:
                score = p["confidence"] if pred_binary == 1 else (1 - p["confidence"])
            else:
                score = float(pred_binary)

            y_true.append(label_binary)
            y_pred.append(pred_binary)
            y_scores.append(float(score))

    if len(y_true) == 0 or len(y_pred) == 0:
        error_msg = {
            "error": "No valid predictions",
            "total": total,
            "valid_ground_truth": valid_gt_count,
            "valid_outputs": valid_output_count,
        }
        if invalid_output_samples:
            error_msg["sample_invalid_outputs"] = invalid_output_samples
        return error_msg

    metrics = {"total": len(y_true)}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    if is_sasv:
        # SASV format: multiclass metrics
        cm_func = confusion_matrix
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0, average=None
        )
        metrics["precision_yes"] = float(precision[0])
        metrics["precision_no"] = float(precision[1])
        metrics["precision_gen"] = float(precision[2])
        metrics["recall_yes"] = float(recall[0])
        metrics["recall_no"] = float(recall[1])
        metrics["recall_gen"] = float(recall[2])
        metrics["f1_yes"] = float(f1[0])
        metrics["f1_no"] = float(f1[1])
        metrics["f1_gen"] = float(f1[2])
        
        # Macro-averaged metrics
        metrics["precision"] = float(np.mean(precision))
        metrics["recall"] = float(np.mean(recall))
        metrics["f1"] = float(np.mean(f1))
        
        # Confusion matrix
        cm = cm_func(y_true, y_pred, labels=[0, 1, 2])
        metrics["confusion_matrix"] = cm.tolist()
        
        # Balanced accuracy (average of per-class recall)
        metrics["balanced_accuracy"] = float(np.mean(recall))
    else:
        # Antispoofing format: binary classification metrics
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        # Store confusion matrix for plotting
        metrics["confusion_matrix"] = cm.tolist()
        
        if cm.size >= 4:
            tn, fp, fn, tp = cm.ravel()[:4]
        elif cm.size == 1:
            tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
        else:
            tn = fp = fn = tp = 0

        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2

        # EER/AUC for binary classification
        if y_scores and len(y_scores) == len(y_true):
            try:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                fnr = 1 - tpr
                eer_idx = np.nanargmin(np.absolute(fnr - fpr))
                metrics["eer"] = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
                metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
            except Exception:
                metrics["eer"] = None
                metrics["auc_roc"] = None
        else:
            metrics["eer"] = None
            metrics["auc_roc"] = None

    return metrics


def _compute_prediction_distribution(predictions: List[Dict]) -> Dict[str, Any]:
    """Count prediction categories: supports both antispoofing (fake/bonafide) and SASV (yes/no/gen)."""
    fake_count = 0
    bonafide_count = 0
    yes_count = 0
    no_count = 0
    gen_count = 0
    unknown_count = 0
    total = len(predictions)
    valid_count = 0
    correct_on_valid = 0

    # Detect format from first prediction
    is_sasv = False
    if predictions:
        first_gt = extract_answer_from_gt(predictions[0].get("gt", "")).lower()
        if first_gt in ["yes", "no", "gen", "verified", "rejected", "spoof"]:
            is_sasv = True

    for p in predictions:
        gt_raw = extract_answer_from_gt(p.get("gt", ""))
        out_raw = extract_pred_answer(p.get("output", ""))
        gt = gt_raw.lower()
        out = out_raw.lower()

        if is_sasv:
            # SASV format
            if out == "yes":
                yes_count += 1
            elif out == "no":
                no_count += 1
            elif out == "gen":
                gen_count += 1
            else:
                unknown_count += 1
                continue

            # Check if ground truth is valid
            if gt in ["yes", "no", "gen", "verified", "rejected", "spoof"]:
                # Normalize GT
                if gt == "verified":
                    gt = "yes"
                elif gt == "rejected":
                    gt = "no"
                elif gt == "spoof":
                    gt = "gen"
                
                valid_count += 1
                if out == gt:
                    correct_on_valid += 1
        else:
            # Antispoofing format
            if out == "fake":
                fake_count += 1
            elif out == "real":
                bonafide_count += 1
            else:
                unknown_count += 1
                continue

            # Check if ground truth is valid (case-insensitive)
            if gt in ["real", "fake"]:
                valid_count += 1
                if out == gt:
                    correct_on_valid += 1

    result = {
        "total": total,
        "valid_count": valid_count,
        "correct_on_valid": correct_on_valid,
    }
    
    if is_sasv:
        result.update({
            "yes": yes_count,
            "no": no_count,
            "gen": gen_count,
            "unknown": unknown_count,
        })
    else:
        result.update({
            "fake": fake_count,
            "bonafide": bonafide_count,
            "unknown": unknown_count,
        })
    
    return result


def _print_metrics_from_predictions(predictions: List[Dict], metrics: Dict, dataset_name: str):
    """Print full metrics including prediction distribution."""
    dist = _compute_prediction_distribution(predictions)
    total = dist["total"]
    valid = dist["valid_count"]
    correct_valid = dist["correct_on_valid"]

    print("\n" + "=" * 60)
    print(f"TEST RUN EVALUATION: {dataset_name}")
    print("=" * 60)
    if dataset_name == "validation":
        print("(Using latest validation run only.)")
    print("\nMetrics (on valid predictions):")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")

    is_sasv_metrics = "precision_yes" in metrics
    if is_sasv_metrics:
        print(f"\nPer-class (yes/no/gen):")
        print(f"  Precision: yes={metrics.get('precision_yes', 0):.4f}, no={metrics.get('precision_no', 0):.4f}, gen={metrics.get('precision_gen', 0):.4f}")
        print(f"  Recall:    yes={metrics.get('recall_yes', 0):.4f}, no={metrics.get('recall_no', 0):.4f}, gen={metrics.get('recall_gen', 0):.4f}")
    else:
        print(f"  Specificity:       {metrics.get('specificity', 0):.4f}")

    if metrics.get("eer") is not None:
        print(f"\nAdvanced Metrics:")
        print(f"  EER:     {metrics['eer']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

    print(f"\nConfusion Matrix:")
    if is_sasv_metrics:
        cm = np.array(metrics.get("confusion_matrix", []))
        if cm.size > 0:
            print("  Predicted: Yes  No  Gen")
            for i, row_label in enumerate(["Yes", "No", "Gen"]):
                print(f"  True {row_label}:  {cm[i, 0]:4} {cm[i, 1]:4} {cm[i, 2]:4}")
    else:
        print(f"  True Positives:  {metrics.get('true_positives', 0)}")
        print(f"  True Negatives:  {metrics.get('true_negatives', 0)}")
        print(f"  False Positives: {metrics.get('false_positives', 0)}")
        print(f"  False Negatives: {metrics.get('false_negatives', 0)}")

    print("\nPrediction distribution:")
    is_sasv = "yes" in dist
    if is_sasv:
        for label, key, count in [
            ("yes", "yes", dist["yes"]),
            ("no", "no", dist["no"]),
            ("gen", "gen", dist["gen"]),
            ("unknown", "unknown", dist["unknown"]),
        ]:
            pct = 100 * count / total if total else 0
            print(f"  {label:12} : {count:6} ({pct:5.1f}%)")
    else:
        for label, key, count in [
            ("fake", "fake", dist["fake"]),
            ("bonafide", "bonafide", dist["bonafide"]),
            ("unknown", "unknown", dist["unknown"]),
        ]:
            pct = 100 * count / total if total else 0
            print(f"  {label:12} : {count:6} ({pct:5.1f}%)")

    print(f"\n  Valid predictions (<answer> tag or hard-label): {valid}/{total} ({100 * valid / total:.1f}%)" if total else "")
    if valid:
        acc_valid = correct_valid / valid
        print(f"  Accuracy on valid: {correct_valid}/{valid} = {acc_valid * 100:.2f}%")

    if is_sasv:
        n_yes_gt = sum(1 for p in predictions if extract_answer_from_gt(p.get("gt", "")).lower() in ["yes", "verified"])
        n_no_gt = sum(1 for p in predictions if extract_answer_from_gt(p.get("gt", "")).lower() in ["no", "rejected"])
        n_gen_gt = sum(1 for p in predictions if extract_answer_from_gt(p.get("gt", "")).lower() in ["gen", "spoof"])
        print(f"\nGround truth: {n_yes_gt} yes (verified), {n_no_gt} no (rejected), {n_gen_gt} gen (spoof)")
    else:
        n_fake_gt = sum(1 for p in predictions if extract_answer_from_gt(p.get("gt", "")).lower() == "fake")
        n_real_gt = total - n_fake_gt
        print(f"\nGround truth: {n_real_gt} bonafide, {n_fake_gt} fake")
    print("=" * 60)


def _print_metrics_from_jsonl(metrics: Dict, dataset_name: str):
    """Print metrics when loaded from metrics.jsonl only."""
    print("\n" + "=" * 60)
    print(f"TEST RUN EVALUATION: {dataset_name}")
    print("=" * 60)
    if dataset_name == "validation":
        print("(Using latest validation run only.)")
    print("(Metrics from metrics.jsonl - no per-sample predictions file)")
    print("\nMetrics:")
    if "accuracy" in metrics:
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    if "accuracy_balanced" in metrics:
        print(f"  Balanced Accuracy: {metrics['accuracy_balanced']:.4f}")
    if "accuracy_yes" in metrics:
        print(f"  Accuracy (Yes):   {metrics['accuracy_yes']:.4f}")
    if "accuracy_no" in metrics:
        print(f"  Accuracy (No):    {metrics['accuracy_no']:.4f}")
    if "accuracy_gen" in metrics:
        print(f"  Accuracy (Gen):   {metrics['accuracy_gen']:.4f}")
    if "accuracy_real" in metrics:
        print(f"  Accuracy (Real):   {metrics['accuracy_real']:.4f}")
    if "accuracy_fake" in metrics:
        print(f"  Accuracy (Fake):   {metrics['accuracy_fake']:.4f}")
    if "loss" in metrics:
        print(f"  Loss:              {metrics['loss']:.4f}")
    print("=" * 60)


def _plot_confidence_charts(predictions: List[Dict], output_dir: str):
    """
    Generate confidence histograms and distributions.
    
    Supports both antispoofing (Real/Fake) and SASV (yes/no/gen) formats.
    Note: Confidence is expected to be P(predicted class).
    """
    confidences = []
    labels = []
    preds = []
    tp_conf, fp_conf, tn_conf, fn_conf = [], [], [], []
    yes_conf, no_conf, gen_conf = [], [], []
    
    # Detect format
    is_sasv = False
    if predictions:
        first_gt = extract_answer_from_gt(predictions[0].get("gt", "")).lower()
        if first_gt in ["yes", "no", "gen", "verified", "rejected", "spoof"]:
            is_sasv = True

    for p in predictions:
        gt_raw = extract_answer_from_gt(p.get("gt", ""))
        out_raw = extract_pred_answer(p.get("output", ""))
        gt = gt_raw.lower()
        out = out_raw.lower()
        
        # Normalize GT for SASV
        if is_sasv:
            if gt == "verified":
                gt = "yes"
            elif gt == "rejected":
                gt = "no"
            elif gt == "spoof":
                gt = "gen"
            
            if gt not in ["yes", "no", "gen"]:
                continue
        else:
            if gt not in ["real", "fake"]:
                continue
        
        # Get confidence (should be P(predicted class))
        conf = p.get("confidence")
        if conf is None:
            if is_sasv:
                # SASV format: use yes_prob/no_prob/gen_prob
                yes_prob = p.get("yes_prob", 0.33)
                no_prob = p.get("no_prob", 0.33)
                gen_prob = p.get("gen_prob", 0.33)
                if out == "yes":
                    conf = yes_prob
                elif out == "no":
                    conf = no_prob
                elif out == "gen":
                    conf = gen_prob
                else:
                    conf = max(yes_prob, no_prob, gen_prob)
            else:
                # Antispoofing format: compute from real_prob/fake_prob
                real_prob = p.get("real_prob", 0.5)
                fake_prob = p.get("fake_prob", 0.5)
                if out == "fake":
                    conf = fake_prob
                elif out == "real":
                    conf = real_prob
                else:
                    conf = 0.5
        
        confidences.append(float(conf))
        
        if is_sasv:
            # SASV format: map to numeric labels
            label_map = {"yes": 0, "no": 1, "gen": 2}
            pred_map = {"yes": 0, "no": 1, "gen": 2}
            labels.append(label_map.get(gt, -1))
            preds.append(pred_map.get(out, -1))
            
            # Per-class confidence
            if "yes_prob" in p:
                if gt == "yes":
                    yes_conf.append(float(p.get("yes_prob", 0)))
                if gt == "no":
                    no_conf.append(float(p.get("no_prob", 0)))
                if gt == "gen":
                    gen_conf.append(float(p.get("gen_prob", 0)))
        else:
            labels.append(1 if gt == "fake" else 0)
            preds.append(1 if out == "fake" else 0)
            
            # Store confidence directly (no transformation needed)
            if out == "fake" and gt == "fake":
                tp_conf.append(float(conf))  # P(Fake) for TP
            elif out == "fake" and gt == "real":
                fp_conf.append(float(conf))  # P(Fake) for FP
            elif out == "real" and gt == "real":
                tn_conf.append(float(conf))  # P(Real) for TN
            elif out == "real" and gt == "fake":
                fn_conf.append(float(conf))  # P(Real) for FN

    if confidences:
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=min(30, max(2, len(set(confidences)))), edgecolor="black", alpha=0.7)
        plt.xlabel("Confidence (P(predicted class))")
        plt.ylabel("Count")
        plt.title("Confidence Distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_histogram.png"), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/confidence_histogram.png")

    if is_sasv:
        # SASV format: plot three classes
        if yes_conf or no_conf or gen_conf:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            color_map = {"yes": "#2ecc71", "no": "#e74c3c", "gen": "#f39c12"}
            for ax, (key, name, vals) in zip(axes, [
                ("yes", "Yes (Verified)", yes_conf),
                ("no", "No (Rejected)", no_conf),
                ("gen", "Gen (Spoof)", gen_conf),
            ]):
                if vals:
                    n_bins = min(20, max(2, len(vals) // 3))
                    ax.hist(vals, bins=n_bins, color=color_map[key], alpha=0.7, range=(0, 1))
                ax.set_title(name, fontsize=11, fontweight='bold')
                ax.set_xlabel("Probability", fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.set_xlim(0, 1)
                ax.grid(True, alpha=0.3)
            plt.suptitle("Confidence Distribution by Class (SASV)", fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confidence_by_class_sasv.png"), dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/confidence_by_class_sasv.png")
    elif tp_conf or fp_conf or tn_conf or fn_conf:
            fn_conf.append(float(conf))  # P(Real) for FN

    if confidences:
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=min(30, max(2, len(set(confidences)))), edgecolor="black", alpha=0.7)
        plt.xlabel("Confidence (P(predicted class))")
        plt.ylabel("Count")
        plt.title("Confidence Distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_histogram.png"), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/confidence_histogram.png")

    if is_sasv:
        # SASV format: plot three classes
        if yes_conf or no_conf or gen_conf:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            color_map = {"yes": "#2ecc71", "no": "#e74c3c", "gen": "#f39c12"}
            for ax, (key, name, vals) in zip(axes, [
                ("yes", "Yes (Verified)", yes_conf),
                ("no", "No (Rejected)", no_conf),
                ("gen", "Gen (Spoof)", gen_conf),
            ]):
                if vals:
                    n_bins = min(20, max(2, len(vals) // 3))
                    ax.hist(vals, bins=n_bins, color=color_map[key], alpha=0.7, range=(0, 1))
                ax.set_title(name, fontsize=11, fontweight='bold')
                ax.set_xlabel("Probability", fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.set_xlim(0, 1)
                ax.grid(True, alpha=0.3)
            plt.suptitle("Confidence Distribution by Class (SASV)", fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confidence_by_class_sasv.png"), dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/confidence_by_class_sasv.png")
    elif tp_conf or fp_conf or tn_conf or fn_conf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        color_map = {"tp": "#2ecc71", "tn": "#3498db", "fp": "#e74c3c", "fn": "#f39c12"}
        for ax, (key, name, vals, xlabel) in zip(axes.ravel(), [
            ("tp", "TP (Pred Fake, GT Fake)", tp_conf, "P(Fake)"),
            ("fp", "FP (Pred Fake, GT Real)", fp_conf, "P(Fake)"),
            ("tn", "TN (Pred Real, GT Real)", tn_conf, "P(Real)"),
            ("fn", "FN (Pred Real, GT Fake)", fn_conf, "P(Real)"),
        ]):
            if vals:
                n_bins = min(20, max(2, len(vals) // 3))
                ax.hist(vals, bins=n_bins, color=color_map[key], alpha=0.7, range=(0, 1))
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
        plt.suptitle("Confidence Distribution by Prediction Type", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_by_prediction_type.png"), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/confidence_by_prediction_type.png")

    correct_conf = [c for c, l, p in zip(confidences, labels, preds) if l == p]
    incorrect_conf = [c for c, l, p in zip(confidences, labels, preds) if l != p]
    if correct_conf or incorrect_conf:
        plt.figure(figsize=(10, 6))
        if correct_conf:
            plt.hist(correct_conf, bins=min(25, max(2, len(set(correct_conf)))), alpha=0.6, label="Correct", color="green")
        if incorrect_conf:
            plt.hist(incorrect_conf, bins=min(25, max(2, len(set(incorrect_conf)))), alpha=0.6, label="Incorrect", color="red")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.title("Confidence: Correct vs Incorrect")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_correct_vs_incorrect.png"), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/confidence_correct_vs_incorrect.png")


def _categorize_predictions(predictions: List[Dict]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Categorize predictions by correctness and formatting.
    Returns (all_lengths, correct_lengths, incorrect_formatted_lengths, wrong_formatting_lengths).
    """
    all_lengths = []
    correct_lengths = []
    incorrect_formatted_lengths = []
    wrong_formatting_lengths = []
    
    for p in predictions:
        output_text = str(p.get("output", ""))
        gt_text = str(p.get("gt", ""))
        
        # Get text length
        text_len = len(output_text)
        all_lengths.append(text_len)
        
        # Extract answers
        out_ans_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL | re.IGNORECASE)
        gt_ans_match = re.search(r"<answer>(.*?)</answer>", gt_text, re.DOTALL | re.IGNORECASE)
        
        # Extract ground truth answer
        gt_ans = None
        if gt_ans_match:
            gt_ans = gt_ans_match.group(1).strip().lower()
        else:
            gt_ans = extract_answer_from_gt(gt_text).lower()
        
        # Check if output has valid answer tag
        if out_ans_match:
            out_ans_raw = out_ans_match.group(1).strip().lower()
            # Check if answer is valid (real or fake)
            if out_ans_raw in ["real", "fake"]:
                # Normalize for comparison
                out_ans = "real" if out_ans_raw == "real" else "fake"
                gt_ans_normalized = "real" if gt_ans in ["real", "bonafide"] else ("fake" if gt_ans in ["fake", "spoof"] else None)
                
                # Only check correctness if we have a valid GT
                if gt_ans_normalized:
                    if out_ans == gt_ans_normalized:
                        correct_lengths.append(text_len)
                    else:
                        incorrect_formatted_lengths.append(text_len)
                else:
                    # GT is invalid, but output is formatted correctly
                    incorrect_formatted_lengths.append(text_len)
            else:
                # Has <answer> tag but answer is not real/fake
                wrong_formatting_lengths.append(text_len)
        else:
            # No <answer> tag found
            wrong_formatting_lengths.append(text_len)
    
    return all_lengths, correct_lengths, incorrect_formatted_lengths, wrong_formatting_lengths


def _plot_text_length_distributions(predictions: List[Dict], output_dir: str):
    """
    Plot text length distributions: overall, correct, incorrect but formatted, wrong formatting.
    Shows distribution histogram with mean, median, std annotations.
    """
    all_lengths, correct_lengths, incorrect_formatted_lengths, wrong_formatting_lengths = _categorize_predictions(predictions)
    
    def compute_stats(lengths):
        if not lengths:
            return None, None, None, None
        arr = np.array(lengths)
        return arr, np.mean(arr), np.median(arr), np.std(arr)
    
    def plot_distribution(ax, lengths, title, color):
        """Plot histogram with stats annotation."""
        if not lengths:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        
        arr, mean, median, std = compute_stats(lengths)
        
        # Plot histogram
        n_bins = min(50, max(10, len(set(lengths)) // 2))
        ax.hist(arr, bins=n_bins, color=color, alpha=0.7, edgecolor="black")
        
        # Add vertical lines for mean and median
        ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.1f}")
        ax.axvline(median, color="blue", linestyle="--", linewidth=2, label=f"Median: {median:.1f}")
        
        # Add text box with stats
        stats_text = f"Mean: {mean:.1f}\nMedian: {median:.1f}\nStd: {std:.1f}\nN: {len(lengths)}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=9)
        
        ax.set_xlabel("Text Length (characters)")
        ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    # Plot 1: Overall distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_distribution(ax, all_lengths, "Text Length Distribution (All Predictions)", "skyblue")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution_all.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/text_length_distribution_all.png")
    
    # Plot 2: Correct predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_distribution(ax, correct_lengths, "Text Length Distribution (Correct Predictions)", "green")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution_correct.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/text_length_distribution_correct.png")
    
    # Plot 3: Incorrect but formatted
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_distribution(ax, incorrect_formatted_lengths, "Text Length Distribution (Incorrect but Formatted)", "orange")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution_incorrect_formatted.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/text_length_distribution_incorrect_formatted.png")
    
    # Plot 4: Wrong formatting
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_distribution(ax, wrong_formatting_lengths, "Text Length Distribution (Wrong Formatting)", "red")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution_wrong_formatting.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/text_length_distribution_wrong_formatting.png")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("TEXT LENGTH STATISTICS")
    print("=" * 60)
    
    categories = [
        ("All Predictions", all_lengths),
        ("Correct", correct_lengths),
        ("Incorrect but Formatted", incorrect_formatted_lengths),
        ("Wrong Formatting", wrong_formatting_lengths),
    ]
    
    for name, lengths in categories:
        if lengths:
            arr, mean, median, std = compute_stats(lengths)
            print(f"\n{name}:")
            print(f"  Count: {len(lengths)}")
            print(f"  Mean: {mean:.2f}")
            print(f"  Median: {median:.2f}")
            print(f"  Std: {std:.2f}")
            print(f"  Min: {np.min(arr):.0f}")
            print(f"  Max: {np.max(arr):.0f}")
        else:
            print(f"\n{name}: No data")
    print("=" * 60)


def _plot_confusion_matrix(predictions: List[Dict], metrics: Dict[str, Any], output_dir: str):
    """
    Plot confusion matrix for test predictions.
    Supports both SASV (yes/no/gen) and antispoofing (Real/Fake) formats.
    """
    if "confusion_matrix" not in metrics:
        return
    
    cm = np.array(metrics["confusion_matrix"])
    
    # Detect format from predictions
    is_sasv = False
    if predictions:
        first_gt = extract_answer_from_gt(predictions[0].get("gt", "")).lower()
        if first_gt in ["yes", "no", "gen", "verified", "rejected", "spoof"]:
            is_sasv = True
    
    plt.figure(figsize=(10, 8))
    
    if is_sasv:
        # SASV format: 3x3 confusion matrix
        labels = ["Yes", "No", "Gen"]
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                        cbar_kws={"label": "Count"}, linewidths=0.5, linecolor="gray")
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar(label="Count")
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels)
            plt.yticks(tick_marks, labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        plt.title("Confusion Matrix (SASV)", fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
    else:
        # Antispoofing format: 2x2 confusion matrix
        labels = ["Real", "Fake"]
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                        cbar_kws={"label": "Count"}, linewidths=0.5, linecolor="gray")
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar(label="Count")
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels)
            plt.yticks(tick_marks, labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        plt.title("Confusion Matrix (Antispoofing)", fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/confusion_matrix.png")


def plot_test_run(log_dir: str, output_dir: str, dataset_name: str = "test") -> Optional[Dict]:
    """
    Plot test run: metrics summary, prediction distribution, histograms.
    Loads from predictions file (or distributed rank files) or falls back to metrics.jsonl.
    Returns metrics dict if successful.
    """
    os.makedirs(output_dir, exist_ok=True)
    predictions, metrics_jsonl = load_test_data(log_dir)

    if predictions:
        metrics = compute_test_metrics(predictions)
        if "error" in metrics:
            print(f"Error computing metrics: {metrics}")
            return metrics
        _print_metrics_from_predictions(predictions, metrics, dataset_name)

        if metrics.get("precision_yes") is not None:
            from ..epochs.utils.sasv_metrics import (
                print_sasv_metrics_summary,
                recompute_sasv_metrics_from_predictions,
            )

            sasv_metrics = recompute_sasv_metrics_from_predictions(
                predictions,
                plot_dir=output_dir,
                plot_prefix=f"sasv_{dataset_name}_from_jsonl",
            )
            print_sasv_metrics_summary(sasv_metrics)
            metrics["sasv_metrics"] = sasv_metrics

        _plot_confidence_charts(predictions, output_dir)
        _plot_confusion_matrix(predictions, metrics, output_dir)
        _plot_text_length_distributions(predictions, output_dir)
        return metrics

    if metrics_jsonl:
        _print_metrics_from_jsonl(metrics_jsonl, dataset_name)
        return metrics_jsonl

    print(f"No test data found in {log_dir}")
    return None
