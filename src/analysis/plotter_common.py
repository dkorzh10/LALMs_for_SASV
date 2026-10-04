"""Shared utilities for train and test plotting."""
import json
import os
import re
import glob
import yaml
from typing import Dict, List, Tuple


def extract_answer_from_tag(text: str) -> str:
    """Extract answer strictly from a well-formed <answer>...</answer> tag (no fallback)."""
    if not text:
        return ""
    text_str = str(text).strip()
    text_str = re.sub(r"^<s>\s*", "", text_str)
    text_str = re.sub(r"\s*</s>", "", text_str)
    text_str = re.sub(r"^<\|startoftext\|>\s*", "", text_str, flags=re.IGNORECASE)
    text_str = re.sub(r"\s*<\|endoftext\|>$", "", text_str, flags=re.IGNORECASE)
    text_str = text_str.strip()

    match = re.search(r"<answer>(.*?)</answer>", text_str, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    answer = match.group(1).strip().lower()
    if answer in ("yes", "no", "gen"):
        return answer
    if answer in ("real", "fake"):
        return answer.capitalize()
    return ""


def _looks_like_reasoning_output(text: str) -> bool:
    t = str(text)
    return any(
        marker in t
        for marker in ("<think>", "<answer>", "<reasons>", "<features>")
    )


def extract_pred_answer(text: str) -> str:
    """Extract model prediction: strict <answer> tag for reasoning outputs, else extract_answer."""
    tagged = extract_answer_from_tag(text)
    if tagged:
        return tagged
    if _looks_like_reasoning_output(text):
        return ""
    return extract_answer(text)


def compute_tag_parse_metrics(outputs: List[str], gts: List[str]) -> Dict[str, float]:
    """Fraction with valid <answer> tag and accuracy on those samples only."""
    total = len(outputs)
    parsed = 0
    correct_parsed = 0
    for o, g in zip(outputs, gts):
        o_ans = extract_answer_from_tag(o)
        if not o_ans:
            continue
        parsed += 1
        if o_ans == extract_answer_from_gt(g):
            correct_parsed += 1
    out: Dict[str, float] = {}
    if total:
        out["ans_parsed"] = parsed / total
    if parsed:
        out["acc2parse"] = correct_parsed / parsed
    return out


def _load_prediction_pairs(log_dir: str, epoch_type: str, epoch: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    patterns = [
        os.path.join(log_dir, f"predictions_{epoch_type}_epoch_{epoch}_rank*.jsonl"),
        os.path.join(log_dir, f"predictions_{epoch_type}_epoch_{epoch}.jsonl"),
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pairs.append((rec.get("output", ""), rec.get("gt", "")))
    return pairs


def backfill_parse_metrics(experiment_dir: str) -> int:
    """Recompute ans_parsed/acc2parse in metrics.jsonl from saved predictions (strict tag parse)."""
    logs_dir = os.path.join(experiment_dir, "logs")
    metrics_path = os.path.join(logs_dir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        metrics_path = os.path.join(experiment_dir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        return 0

    log_dir = os.path.dirname(metrics_path)
    with open(metrics_path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    updated = 0
    new_lines = []
    for line in lines:
        rec = json.loads(line)
        epoch_type = rec.get("type")
        if epoch_type not in ("validation", "test"):
            new_lines.append(line)
            continue
        epoch = rec.get("epoch")
        if epoch is None:
            new_lines.append(line)
            continue
        pairs = _load_prediction_pairs(log_dir, epoch_type, int(epoch))
        if not pairs:
            new_lines.append(line)
            continue
        metrics = compute_tag_parse_metrics([p[0] for p in pairs], [p[1] for p in pairs])
        if "ans_parsed" in metrics:
            rec["ans_parsed"] = metrics["ans_parsed"]
            updated += 1
        if "acc2parse" in metrics:
            rec["acc2parse"] = metrics["acc2parse"]
        elif "acc2parse" in rec:
            del rec["acc2parse"]
        new_lines.append(json.dumps(rec))

    if updated:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    return updated


def extract_answer(text: str) -> str:
    """Extract final answer from either reasoning or hard_label format.
    Supports both SASV format (yes/no/gen) and antispoofing format (Real/Fake).
    """
    if not text:
        return ""
    text_str = str(text).strip()
    
    # Strip common special tokens that may appear at the start/end
    # Remove <s>, </s>, and similar tokens
    text_str = re.sub(r'^<s>\s*', '', text_str)
    text_str = re.sub(r'\s*</s>', '', text_str)
    text_str = re.sub(r'^<\|startoftext\|>\s*', '', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\s*<\|endoftext\|>$', '', text_str, flags=re.IGNORECASE)
    text_str = text_str.strip()
    
    # Reasoning format: <answer>yes/no/gen or Real/Fake</answer>
    match = re.search(r"<answer>(.*?)</answer>", text_str, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        # Normalize SASV answers
        if answer in ["yes", "no", "gen"]:
            return answer
        # Normalize antispoofing answers
        if answer in ["real", "fake"]:
            return answer.capitalize()
        return match.group(1).strip()
    
    # Hard-label format: Final Answer: yes/no/gen or Real/Fake
    match = re.search(r"Final\s*Answer:\s*(yes|no|gen|real|fake)", text_str, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        if answer in ["yes", "no", "gen"]:
            return answer
        if answer in ["real", "fake"]:
            return answer.capitalize()
        return match.group(1).strip()
    
    # Fallback: hierarchical / free text — берём самое раннее целое слово yes|no|gen,
    # чтобы «… no … yes» не превращалось в yes из-за порядка проверок.
    text_lower = text_str.lower()
    m_ng = re.search(r"\bno_gen\b", text_lower)
    if m_ng:
        tail = text_lower[m_ng.end() :].strip()
        m_tail = re.match(r"^(yes|no|gen)\b", tail)
        if m_tail:
            return m_tail.group(1)
        m_any = re.search(r"\b(yes|no|gen)\b", tail)
        if m_any:
            return m_any.group(1)
        return ""
    hits = []
    for lab in ("yes", "no", "gen"):
        m = re.search(rf"\b{re.escape(lab)}\b", text_lower)
        if m:
            hits.append((m.start(), lab))
    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[0][1]
    if re.search(r"\b(generated|spoof)\b", text_lower):
        return "gen"
    # Check antispoofing format
    if "fake" in text_lower:
        return "Fake"
    if "real" in text_lower:
        return "Real"
    
    # If no valid answer found, return empty string (not the malformed text)
    # This helps distinguish between "no answer" and "malformed output"
    return ""


def extract_answer_from_gt(text: str) -> str:
    """Extract ground truth answer.
    Supports both SASV format (yes/no/gen or verified/rejected/spoof) and antispoofing format (Real/Fake).
    """
    if not text:
        return ""
    text_str = str(text).strip()
    text_clean = text_str.lower()
    
    # Map SASV GT values: verified -> yes, rejected -> no, spoof -> gen
    if text_clean == "verified":
        return "yes"
    if text_clean == "rejected":
        return "no"
    if text_clean == "spoof":
        return "gen"
    
    # Check if it's already yes/no/gen
    if text_clean in ["yes", "no", "gen"]:
        return text_clean
    
    # Check if it's just "real" or "fake" (antispoofing format)
    if text_clean in ["real", "fake"]:
        return text_clean.capitalize()
    
    # Try reasoning format: <answer>yes/no/gen or Real/Fake</answer>
    match = re.search(r"<answer>(.*?)</answer>", text_str, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        if answer in ["yes", "no", "gen"]:
            return answer
        if answer in ["real", "fake"]:
            return answer.capitalize()
        return match.group(1).strip()
    
    # Try hard-label format: Final Answer: yes/no/gen or Real/Fake
    match = re.search(r"Final\s*Answer:\s*(yes|no|gen|real|fake)", text_str, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        if answer in ["yes", "no", "gen"]:
            return answer
        if answer in ["real", "fake"]:
            return answer.capitalize()
        return match.group(1).strip()
    
    # Fallback: look for yes/no/gen or real/fake anywhere
    if re.search(r'\byes\b', text_clean):
        return "yes"
    if re.search(r'\bno\b', text_clean) and not re.search(r'\b(gen|generated|spoof)\b', text_clean):
        return "no"
    if re.search(r'\bgen\b', text_clean) or re.search(r'\b(generated|spoof)\b', text_clean):
        return "gen"
    if "fake" in text_clean:
        return "Fake"
    if "real" in text_clean:
        return "Real"
    
    return text_str


def detect_run_type(experiment_dir: str) -> Tuple[str, List[str]]:
    """
    Detect whether a run is train, test, or both.
    Uses config_resolved.yaml (Runner.type) as primary source; falls back to logs structure.
    Returns (run_type, test_log_dirs).
    run_type: "train" | "test" | "both"
    test_log_dirs: list of paths to logs/test_* directories (for test runs)
    """
    config_path = os.path.join(experiment_dir, "config_resolved.yaml")
    logs_dir = os.path.join(experiment_dir, "logs")

    # Primary: read Runner.type from config_resolved.yaml
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            runner_type = cfg.get("Runner", {}).get("type", "train")
            if runner_type == "test":
                # Find test log dirs (logs/test_*)
                test_subdirs = []
                if os.path.isdir(logs_dir):
                    for name in os.listdir(logs_dir):
                        if name.startswith("test_") and os.path.isdir(os.path.join(logs_dir, name)):
                            test_subdirs.append(os.path.join(logs_dir, name))
                return "test", test_subdirs
            # train (or unknown type)
            return "train", []
        except (yaml.YAMLError, OSError):
            pass

    # Fallback: infer from logs structure
    if not os.path.isdir(logs_dir):
        if os.path.exists(os.path.join(experiment_dir, "metrics.jsonl")):
            return "train", []
        return "unknown", []

    has_train = (
        os.path.exists(os.path.join(logs_dir, "metrics.jsonl")) or
        os.path.isdir(os.path.join(logs_dir, "judge_logs"))
    )
    test_subdirs = []
    for name in os.listdir(logs_dir):
        if name.startswith("test_") and os.path.isdir(os.path.join(logs_dir, name)):
            test_log_dir = os.path.join(logs_dir, name)
            has_test = (
                os.path.exists(os.path.join(test_log_dir, "metrics.jsonl")) or
                glob.glob(os.path.join(test_log_dir, "samples_test_epoch_*.jsonl")) or
                glob.glob(os.path.join(test_log_dir, "predictions_test_epoch_*.jsonl"))
            )
            if has_test:
                test_subdirs.append(test_log_dir)

    if has_train and test_subdirs:
        return "both", test_subdirs
    if has_train:
        return "train", []
    if test_subdirs:
        return "test", test_subdirs
    return "unknown", []


def normalize_sasv_label(label: str) -> str:
    """Map SASV labels to yes/no/gen."""
    if not label:
        return ""
    label = str(label).strip().lower()
    if label in ("verified",):
        return "yes"
    if label in ("rejected",):
        return "no"
    if label in ("spoof", "generated"):
        return "gen"
    if label in ("yes", "no", "gen"):
        return label
    return label


def compute_sasv_subsystem_accuracies(samples: List[dict]) -> dict:
    """Compute ASV (yes/no) and counter-measure (bonafide vs spoof) accuracies."""
    asv_correct, asv_total = 0, 0
    cm_correct, cm_total = 0, 0

    for sample in samples:
        gt = normalize_sasv_label(extract_answer_from_gt(sample.get("gt", "")))
        pred = normalize_sasv_label(extract_pred_answer(sample.get("output", "")))
        if gt not in ("yes", "no", "gen"):
            continue

        cm_total += 1
        if (gt == "gen") == (pred == "gen"):
            cm_correct += 1

        if gt in ("yes", "no"):
            if pred not in ("yes", "no", "gen"):
                continue
            asv_total += 1
            if pred == gt:
                asv_correct += 1

    out = {}
    if asv_total:
        out["asv_accuracy"] = asv_correct / asv_total
    if cm_total:
        out["cm_accuracy"] = cm_correct / cm_total
    return out
