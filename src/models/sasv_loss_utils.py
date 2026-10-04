"""Token-level and sample-level CE weighting for SASV reasoning SFT.

Supports upweighting answer / format tokens and optional per-example class
weights. Defaults (all weights 1.0, focal_gamma=0) match unweighted CE.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

_TRACE_CLASS_ALIASES = {
    "yes": "yes",
    "true": "yes",
    "target": "yes",
    "verified": "yes",
    "no": "no",
    "false": "no",
    "nontarget": "no",
    "rejected": "no",
    "gen": "gen",
    "spoof": "gen",
    "generated": "gen",
}


def normalize_trace_class_label(label: Union[str, None]) -> Optional[str]:
    """Normalize a label to yes / no / gen."""
    if label is None:
        return None
    text = str(label).strip().lower()
    if not text:
        return None
    return _TRACE_CLASS_ALIASES.get(text)


def parse_trace_class_weights(raw: Optional[dict]) -> Optional[Dict[str, float]]:
    """Parse class weights; return None if all values are 1.0."""
    if not raw or not isinstance(raw, dict):
        return None

    weights = {"yes": 1.0, "no": 1.0, "gen": 1.0}
    any_non_default = False
    for raw_key, raw_value in raw.items():
        key = _TRACE_CLASS_ALIASES.get(str(raw_key).strip().lower())
        if key is None:
            logging.warning(
                "Ignoring unknown trace_class_weights key %r (use yes/no/gen or aliases)",
                raw_key,
            )
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            logging.warning(
                "Ignoring invalid trace_class_weights value for key %r: %r",
                raw_key,
                raw_value,
            )
            continue
        weights[key] = value
        if value != 1.0:
            any_non_default = True
    return weights if any_non_default else None


def sample_trace_class_weights(
    labels: List[Union[str, None]],
    class_weights: Dict[str, float],
    *,
    default_weight: float = 1.0,
) -> torch.Tensor:
    """Per-example weights for a batch of GT / answer labels."""
    sample_weights: List[float] = []
    for label in labels:
        norm = normalize_trace_class_label(label)
        if norm is None:
            sample_weights.append(default_weight)
            continue
        sample_weights.append(float(class_weights.get(norm, default_weight)))
    return torch.tensor(sample_weights, dtype=torch.float32)


def build_full_trace_loss_weights(
    targets: torch.Tensor,
    text_region_start: int,
    *,
    sample_weights: Optional[torch.Tensor] = None,
    text_token_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine optional sample and token weights into a (B, T) CE weight tensor."""
    batch_size, seq_len = targets.shape
    device = targets.device
    weights = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)

    if text_token_weights is not None:
        tw = text_token_weights.to(device=device, dtype=torch.float32)
        if tw.shape != weights.shape:
            raise ValueError(
                f"text_token_weights shape {tuple(tw.shape)} != targets {tuple(weights.shape)}"
            )
        weights[:, text_region_start:] = tw[:, text_region_start:]

    if sample_weights is not None:
        sw = sample_weights.to(device=device, dtype=torch.float32).view(batch_size, 1)
        text_slice = weights[:, text_region_start:]
        weights[:, text_region_start:] = text_slice * sw

    return weights


# Structural tags that wrap the reasoning / answer in the distilled traces.
FORMAT_TAGS = ("<think>", "</think>", "<answer>", "</answer>")
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def _char_to_token_index(tokenizer, text: str, char_idx: int) -> int:
    """Map a character offset in ``text`` to a token index.

    Uses prefix tokenization: ``len(tokenize(text[:char_idx]))``.  This is exact
    at the tag boundaries we care about because the LLaMA/Vicuna SentencePiece
    tokenizer emits ``<`` / ``>`` / ``/`` as standalone pieces, so a span that
    starts right after ``>`` (answer content) or right before ``<`` (closing
    tag) lands on a clean token boundary.
    """
    if char_idx <= 0:
        return 0
    return len(tokenizer(text[:char_idx], add_special_tokens=False).input_ids)


def build_row_token_weights(
    tokenizer,
    full_text: str,
    seq_len: int,
    *,
    base_weight: float = 1.0,
    answer_weight: float = 1.0,
    format_weight: float = 1.0,
    eos_weight: Optional[float] = None,
) -> Tuple[List[float], List[bool]]:
    """Compute per-token loss weights for a single target sequence.

    Args:
        tokenizer: SALMONN's ``llama_tokenizer``.
        full_text: the target text *including* the trailing end symbol, i.e.
            exactly the string that was tokenized into the row's ``input_ids``.
        seq_len: number of token positions in the (padded) row.
        base_weight: weight for ordinary reasoning tokens.
        answer_weight: weight for the answer content tokens (yes/no/gen).
        format_weight: weight for the structural tag tokens.
        eos_weight: weight for the trailing end symbol; defaults to ``answer_weight``.

    Returns:
        (weights, answer_mask) where ``weights`` is a list of length ``seq_len``
        and ``answer_mask[i]`` marks the answer-content positions (used for the
        optional focal modulation).
    """
    if eos_weight is None:
        eos_weight = answer_weight

    weights = [base_weight] * seq_len
    answer_mask = [False] * seq_len

    def boost(c_start: int, c_end: int, w: float, mark_answer: bool = False) -> None:
        t_start = _char_to_token_index(tokenizer, full_text, c_start)
        t_end = _char_to_token_index(tokenizer, full_text, c_end)
        for j in range(max(0, t_start), min(seq_len, t_end)):
            # ``max`` so that a smaller earlier boost never overrides a larger one.
            weights[j] = max(weights[j], w)
            if mark_answer:
                answer_mask[j] = True

    # Structural formatting tags (all occurrences).
    if format_weight != base_weight:
        for tag in FORMAT_TAGS:
            start = 0
            while True:
                pos = full_text.find(tag, start)
                if pos < 0:
                    break
                boost(pos, pos + len(tag), format_weight)
                start = pos + len(tag)

    # Answer content: between the last <answer> and the following </answer>.
    open_pos = full_text.rfind(ANSWER_OPEN)
    if open_pos >= 0:
        content_start = open_pos + len(ANSWER_OPEN)
        close_pos = full_text.find(ANSWER_CLOSE, content_start)
        content_end = close_pos if close_pos >= 0 else len(full_text)
        if content_end > content_start:
            boost(content_start, content_end, answer_weight, mark_answer=True)

    # Trailing end symbol (so the model learns to stop right after the answer).
    if eos_weight != base_weight and seq_len > 0:
        # The EOS is the last real token of the row; mark the final position.
        weights[seq_len - 1] = max(weights[seq_len - 1], eos_weight)

    return weights, answer_mask


def build_text_weight_tensors(
    tokenizer,
    target_texts: List[str],
    end_sym: str,
    input_ids: torch.Tensor,
    *,
    base_weight: float = 1.0,
    answer_weight: float = 1.0,
    format_weight: float = 1.0,
    eos_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (B, L) weight and answer-mask tensors aligned to ``input_ids``.

    ``input_ids`` are the tokenized ``[t + end_sym for t in target_texts]`` (the
    ``to_regress_tokens`` in the model forward).
    """
    batch_size, seq_len = input_ids.shape
    weights = torch.ones(batch_size, seq_len, dtype=torch.float32)
    answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    for b in range(batch_size):
        full_text = target_texts[b] + end_sym
        row_w, row_m = build_row_token_weights(
            tokenizer,
            full_text,
            seq_len,
            base_weight=base_weight,
            answer_weight=answer_weight,
            format_weight=format_weight,
            eos_weight=eos_weight,
        )
        weights[b] = torch.tensor(row_w, dtype=torch.float32)
        answer_mask[b] = torch.tensor(row_m, dtype=torch.bool)

    weights = weights.to(input_ids.device)
    answer_mask = answer_mask.to(input_ids.device)
    return weights, answer_mask


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    answer_mask: Optional[torch.Tensor] = None,
    focal_gamma: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted next-token cross-entropy with optional answer-token focal term.

    Args:
        logits: (B, T, V) LM logits over the full input sequence.
        targets: (B, T) labels with ``-100`` at masked positions; aligned with
            ``logits`` (HF-style, shifting handled here).
        token_weights: (B, T) per-token loss weights aligned with ``targets``.
        answer_mask: (B, T) bool marking answer-content positions for focal.
        focal_gamma: if > 0, multiply answer-token weights by ``(1 - p)^gamma``
            where ``p`` is the model's probability of the gold token (token-level
            hard mining on the decision token).

    Returns:
        Scalar weighted-mean cross-entropy.  Equals the standard mean CE when all
        weights are ``1`` and ``focal_gamma == 0``.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = targets[:, 1:].contiguous()
    shift_weights = token_weights[:, 1:].contiguous()

    batch, seq_minus_1, vocab = shift_logits.shape
    flat_logits = shift_logits.view(-1, vocab).float()
    flat_labels = shift_labels.view(-1)

    valid = flat_labels != -100
    # ``ignore_index`` keeps ignored positions at 0 loss; safe to compute on all.
    per_token = F.cross_entropy(
        flat_logits, flat_labels.clamp_min(0), reduction="none"
    )
    per_token = per_token * valid.float()

    flat_weights = (shift_weights.reshape(-1) * valid.float()).clone()

    if focal_gamma > 0.0 and answer_mask is not None:
        shift_answer = answer_mask[:, 1:].contiguous().reshape(-1) & valid
        if shift_answer.any():
            with torch.no_grad():
                p = torch.exp(-per_token)  # prob of gold token
                focal = (1.0 - p).clamp(min=0.0, max=1.0) ** focal_gamma
            flat_weights = torch.where(shift_answer, flat_weights * focal, flat_weights)

    denom = flat_weights.sum().clamp_min(eps)
    return (per_token * flat_weights).sum() / denom
