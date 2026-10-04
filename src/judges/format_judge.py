import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from .base import Judge

_SASV_ANSWERS = frozenset({"yes", "no", "gen"})
_ANTISPOOF_ANSWERS = frozenset({"real", "fake"})


def _normalize_gt_answer(raw: str, answer_labels: str) -> Optional[str]:
    """Normalize ground-truth label to canonical yes/no/gen or real/fake."""
    if raw is None:
        return None
    ans = str(raw).strip().lower()
    match = re.search(r"<answer>(.*?)</answer>", ans, re.DOTALL | re.IGNORECASE)
    if match:
        ans = match.group(1).strip().lower()
    else:
        match = re.search(r"final\s*answer:\s*(yes|no|gen|real|fake)", ans, re.IGNORECASE)
        if match:
            ans = match.group(1).strip().lower()

    if answer_labels == "sasv":
        if ans in ("verified", "bonafide", "same"):
            return "yes"
        if ans in ("rejected", "different"):
            return "no"
        if ans in ("spoof", "generated"):
            return "gen"
        return ans if ans in _SASV_ANSWERS else None

    if ans in ("bonafide", "real", "verified"):
        return "real"
    if ans in ("fake", "spoof", "generated"):
        return "fake"
    return ans if ans in _ANTISPOOF_ANSWERS else None


def _parse_output_answer(raw: str, answer_labels: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", str(raw), re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    ans = match.group(1).strip().lower()
    if answer_labels == "sasv":
        return ans if ans in _SASV_ANSWERS else None
    if ans == "real":
        return "real"
    if ans == "fake":
        return "fake"
    return None


def _normalize_reason_token(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    token = raw.strip().upper()
    return token or None


def _reasons_from_list(parsed: Any) -> Set[str]:
    if not isinstance(parsed, list):
        return set()
    result: Set[str] = set()
    for item in parsed:
        if isinstance(item, str):
            if item.startswith("["):
                try:
                    inner = ast.literal_eval(item)
                except (SyntaxError, ValueError):
                    inner = None
                if isinstance(inner, list):
                    result.update(
                        t for t in (_normalize_reason_token(r) for r in inner) if t
                    )
            else:
                token = _normalize_reason_token(item)
                if token:
                    result.add(token)
    return result


def _parse_reasons(text: str) -> Set[str]:
    """Extract reasons set from <reasons>[...]</reasons> tag."""
    match = re.search(r"<reasons>\s*(\[.*?\])\s*</reasons>", str(text), re.DOTALL)
    if not match:
        return set()
    raw = match.group(1)
    for parser in (ast.literal_eval, eval):
        try:
            return _reasons_from_list(parser(raw))
        except Exception:
            continue
    return set()


def _parse_features(text: str) -> Dict[str, Any]:
    """Extract acoustic features dict from <features>...</features> tag."""
    match = re.search(r"<features>\s*(.*?)\s*</features>", str(text), re.DOTALL)
    if not match:
        return {}
    raw = match.group(1).strip()
    if not raw:
        return {}
    for parser in (ast.literal_eval, eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _normalize_feature_value(raw: Any) -> str:
    return str(raw).strip().lower()


def _flatten_acoustic_features(features: Dict[str, Any]) -> Set[Tuple[str, str, str]]:
    """Flatten nested reference/query acoustic features to comparable tuples."""
    flat: Set[Tuple[str, str, str]] = set()
    for key, val in features.items():
        key_n = str(key).strip().lower()
        if isinstance(val, dict):
            for side, side_val in val.items():
                side_n = str(side).strip().lower()
                flat.add((key_n, side_n, _normalize_feature_value(side_val)))
        else:
            flat.add((key_n, "", _normalize_feature_value(val)))
    return flat


def _compute_set_overlap(gt_items: Set[Any], out_items: Set[Any]) -> float:
    """Recall * precision overlap for comparable token sets."""
    if not gt_items or not out_items:
        return 0.0
    num_correct = len(gt_items & out_items)
    recall = num_correct / len(gt_items)
    precision = num_correct / len(out_items)
    return recall * precision


def _compute_correct_reasons_overlap(gt_reasons: Set[str], out_reasons: Set[str]) -> float:
    """
    Metric: (num_correct / len(gt_reasons)) * (num_correct / len(out_reasons)).
    Returns 0 if either set is empty.
    """
    return _compute_set_overlap(gt_reasons, out_reasons)


def _compute_correct_features_overlap(
    gt_features: Dict[str, Any], out_features: Dict[str, Any]
) -> float:
    return _compute_set_overlap(
        _flatten_acoustic_features(gt_features),
        _flatten_acoustic_features(out_features),
    )


class FormatJudge(Judge):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        weights = config.get("weights", {})
        self.w_format = weights.get("format", 0.5)
        self.w_correctness = weights.get("correctness", 0.5)
        self.w_reasons_correctness = weights.get("reasons_correctness", 0.0)
        self.w_features_correctness = weights.get("features_correctness", 0.0)
        self.answer_labels = str(config.get("answer_labels", "antispoofing")).lower()
        if self.answer_labels not in ("sasv", "antispoofing"):
            self.answer_labels = "antispoofing"
        self.require_reasons = config.get(
            "require_reasons",
            self.answer_labels != "sasv",
        )
        self.require_features = config.get("require_features", False)

    def score(self, inputs, outputs, gt, meta=None):
        results = []

        for out, g in zip(outputs, gt):
            format_ok = False
            is_correct = False
            correct_reasons_overlap = 0.0
            correct_features_overlap = 0.0
            ans = None

            has_features = "<features>" in out and "</features>" in out
            has_think = "<think>" in out and "</think>" in out
            has_reasons = "<reasons>" in out and "</reasons>" in out
            has_answer = "<answer>" in out and "</answer>" in out

            g_text = str(g)
            g_ans = _normalize_gt_answer(g, self.answer_labels)
            gt_features = _parse_features(g_text)
            gt_reasons = _parse_reasons(g_text)
            gt_has_features = bool(gt_features)
            gt_has_reasons = bool(gt_reasons)

            structure_ok = has_think and has_answer
            if self.require_features or gt_has_features:
                structure_ok = structure_ok and has_features
            if self.require_reasons or gt_has_reasons:
                structure_ok = structure_ok and has_reasons

            if structure_ok:
                format_ok = True
                ans = _parse_output_answer(out, self.answer_labels)
                if ans is not None and g_ans is not None and ans == g_ans:
                    is_correct = True

            if gt_reasons and self.w_reasons_correctness > 0:
                if format_ok:
                    out_reasons = _parse_reasons(out)
                    correct_reasons_overlap = _compute_correct_reasons_overlap(
                        gt_reasons, out_reasons
                    )
            elif not gt_reasons and self.w_reasons_correctness > 0 and format_ok and is_correct:
                if g_ans in ("real", "yes"):
                    correct_reasons_overlap = 1.0

            if gt_features and self.w_features_correctness > 0:
                if format_ok:
                    out_features = _parse_features(out)
                    correct_features_overlap = _compute_correct_features_overlap(
                        gt_features, out_features
                    )

            score = 0.0
            score += self.w_format * (1.0 if format_ok else 0.0)
            score += self.w_correctness * (1.0 if is_correct else 0.0)
            score += self.w_reasons_correctness * correct_reasons_overlap
            score += self.w_features_correctness * correct_features_overlap

            results.append({
                "score": score,
                "format_ok": format_ok,
                "is_correct": is_correct,
                "correct_reasons_overlap": correct_reasons_overlap,
                "correct_features_overlap": correct_features_overlap,
                "answer": ans,  # "real" | "fake" | None (for skeptic filtering)
            })

        return {
            "score": sum(r["score"] for r in results) / len(results),
            "meta": {"per_sample": results},
        }

