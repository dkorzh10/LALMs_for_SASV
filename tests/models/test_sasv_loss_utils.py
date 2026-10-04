"""Unit tests for SASV token-level CE weighting utilities.

Uses a small character-level SentencePiece-like mock tokenizer that reproduces
the property we rely on: ``<``, ``>``, ``/`` are standalone pieces, so tag
boundaries land on clean token positions.  Also includes an optional check
against the real Vicuna tokenizer when transformers + the checkpoint are
available.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from src.models.sasv_loss_utils import (
    build_full_trace_loss_weights,
    build_row_token_weights,
    build_text_weight_tensors,
    normalize_trace_class_label,
    parse_trace_class_weights,
    sample_trace_class_weights,
    weighted_causal_lm_loss,
)


class CharTokenizer:
    """Minimal tokenizer: 1 token per character, plus a single ``</s>`` token.

    This is enough to validate the char->token index math and span boundaries,
    since ``len(encode(prefix)) == len(prefix)`` for tag-delimited spans.
    """

    EOS = "</s>"

    def __init__(self):
        vocab = sorted(set("<>/abcdefghijklmnopqrstuvwxyz ESTANYG.,'"))
        self._stoi = {c: i + 1 for i, c in enumerate(vocab)}  # reserve 0 for eos
        self._stoi[self.EOS] = 0

    def _encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            if text.startswith(self.EOS, i):
                ids.append(self._stoi[self.EOS])
                i += len(self.EOS)
            else:
                ids.append(self._stoi.get(text[i], 1))
                i += 1
        return ids

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self._encode(text))

    def encode(self, text, add_special_tokens=False):
        return self._encode(text)


def _text(answer="gen", reasoning="the voice is robotic"):
    return f"<think>{reasoning}</think><answer>{answer}</answer>"


def test_answer_span_is_weighted():
    tok = CharTokenizer()
    full = _text("gen") + "</s>"
    seq_len = len(tok.encode(full))
    weights, answer_mask = build_row_token_weights(
        tok, full, seq_len,
        base_weight=1.0, answer_weight=8.0, format_weight=2.0,
    )

    # The 3 chars of "gen" inside <answer>gen</answer> must carry answer weight.
    boosted = [i for i, w in enumerate(weights) if w == 8.0]
    assert len(boosted) == len("gen"), (boosted, weights)
    assert all(answer_mask[i] for i in boosted)

    # Those positions decode back to "gen".
    decoded = "".join(
        c for c, idx in zip(full, range(len(full))) if idx in boosted
    )
    assert decoded == "gen"


def test_format_tags_weighted_and_reasoning_is_base():
    tok = CharTokenizer()
    full = _text("no") + "</s>"
    seq_len = len(tok.encode(full))
    weights, _ = build_row_token_weights(
        tok, full, seq_len,
        base_weight=1.0, answer_weight=5.0, format_weight=3.0,
    )
    # Every tag character should be >= format weight.
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        start = full.index(tag)
        for j in range(start, start + len(tag)):
            assert weights[j] >= 3.0
    # A reasoning character keeps the base weight.
    r_idx = full.index("robotic")
    assert weights[r_idx] == 1.0


def test_eos_is_weighted():
    tok = CharTokenizer()
    full = _text("yes") + "</s>"
    seq_len = len(tok.encode(full))
    weights, _ = build_row_token_weights(
        tok, full, seq_len,
        base_weight=1.0, answer_weight=4.0, format_weight=2.0, eos_weight=4.0,
    )
    assert weights[-1] == 4.0  # the appended </s>


def test_uniform_weights_match_mean_cross_entropy():
    torch.manual_seed(0)
    b, t, v = 2, 6, 11
    logits = torch.randn(b, t, v)
    targets = torch.randint(0, v, (b, t))
    targets[:, :2] = -100  # masked prefix

    weights = torch.ones(b, t)
    custom = weighted_causal_lm_loss(logits, targets, weights, focal_gamma=0.0)

    # Reference HF-style mean CE.
    shift_logits = logits[:, :-1, :].reshape(-1, v)
    shift_labels = targets[:, 1:].reshape(-1)
    ref = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    assert torch.allclose(custom, ref, atol=1e-5), (custom, ref)


def test_answer_weight_increases_answer_gradient_share():
    torch.manual_seed(1)
    b, t, v = 1, 8, 11
    logits = torch.randn(b, t, v, requires_grad=False)
    targets = torch.randint(0, v, (b, t))
    targets[:, :2] = -100

    answer_mask = torch.zeros(b, t, dtype=torch.bool)
    answer_mask[0, 6] = True  # one answer token near the end

    w_uniform = torch.ones(b, t)
    w_boosted = torch.ones(b, t)
    w_boosted[0, 6] = 10.0

    lu = weighted_causal_lm_loss(logits, targets, w_uniform)
    lb = weighted_causal_lm_loss(logits, targets, w_boosted)
    # Both finite; boosting changes the loss (answer token now dominates).
    assert torch.isfinite(lu) and torch.isfinite(lb)
    assert not torch.allclose(lu, lb)


def test_build_text_weight_tensors_shapes():
    tok = CharTokenizer()
    texts = [_text("gen"), _text("yes", reasoning="same speaker matching timbre")]
    full = [t + "</s>" for t in texts]
    max_len = max(len(tok.encode(f)) for f in full)
    # Emulate right padding to a common length.
    input_ids = torch.zeros(2, max_len, dtype=torch.long)
    for i, f in enumerate(full):
        ids = tok.encode(f)
        input_ids[i, : len(ids)] = torch.tensor(ids)

    weights, answer_mask = build_text_weight_tensors(
        tok, texts, "</s>", input_ids,
        base_weight=1.0, answer_weight=8.0, format_weight=2.0,
    )
    assert weights.shape == (2, max_len)
    assert answer_mask.shape == (2, max_len)
    assert (weights >= 1.0).all()
    assert answer_mask.any()


@pytest.mark.skipif(
    not os.environ.get("LLAMA_PATH") or not os.path.exists(os.environ["LLAMA_PATH"]),
    reason="LLAMA_PATH not set",
)
def test_real_vicuna_tokenizer_answer_span():
    transformers = pytest.importorskip("transformers")
    tok = transformers.LlamaTokenizer.from_pretrained(
        os.environ["LLAMA_PATH"],
        use_fast=False,
        local_files_only=True,
    )
    full = _text("gen") + "</s>"
    ids = tok(full, add_special_tokens=False).input_ids
    seq_len = len(ids)
    weights, answer_mask = build_row_token_weights(
        tok, full, seq_len,
        base_weight=1.0, answer_weight=8.0, format_weight=2.0,
    )
    boosted = [i for i, w in enumerate(weights) if w == 8.0 and answer_mask[i]]
    assert boosted, "answer span not detected with real tokenizer"
    decoded = tok.decode([ids[i] for i in boosted]).strip()
    assert "gen" in decoded

def test_normalize_trace_class_label_aliases():
    assert normalize_trace_class_label("verified") == "yes"
    assert normalize_trace_class_label("target") == "yes"
    assert normalize_trace_class_label("rejected") == "no"
    assert normalize_trace_class_label("nontarget") == "no"
    assert normalize_trace_class_label("spoof") == "gen"
    assert normalize_trace_class_label("GEN") == "gen"
    assert normalize_trace_class_label("") is None


def test_parse_trace_class_weights_noop_when_all_one():
    assert parse_trace_class_weights({"yes": 1.0, "no": 1.0, "gen": 1.0}) is None
    assert parse_trace_class_weights(None) is None


def test_parse_trace_class_weights_accepts_aliases():
    parsed = parse_trace_class_weights({"target": 1.0, "nontarget": 1.0, "spoof": 2.0})
    assert parsed == {"yes": 1.0, "no": 1.0, "gen": 2.0}


def test_sample_trace_class_weights():
    weights = {"yes": 1.0, "no": 1.5, "gen": 2.0}
    got = sample_trace_class_weights(["yes", "rejected", "spoof", "unknown"], weights)
    assert got.tolist() == [1.0, 1.5, 2.0, 1.0]


def test_build_full_trace_loss_weights_scales_text_region_only():
    targets = torch.tensor([[ -100, -100, 3, 4, 5], [ -100, -100, 6, 7, 8]])
    sample_weights = torch.tensor([2.0, 3.0])
    weights = build_full_trace_loss_weights(targets, text_region_start=2, sample_weights=sample_weights)
    assert weights[0, :2].tolist() == [1.0, 1.0]
    assert weights[0, 2:].tolist() == [2.0, 2.0, 2.0]
    assert weights[1, 2:].tolist() == [3.0, 3.0, 3.0]


def test_trace_sample_weight_changes_loss_vs_uniform():
    torch.manual_seed(0)
    b, t, v = 2, 5, 9
    logits = torch.randn(b, t, v)
    targets = torch.tensor([[-100, -100, 1, 2, 3], [-100, -100, 4, 5, 6]])
    uniform = build_full_trace_loss_weights(targets, text_region_start=2)
    weighted = build_full_trace_loss_weights(
        targets, text_region_start=2, sample_weights=torch.tensor([1.0, 3.0])
    )
    lu = weighted_causal_lm_loss(logits, targets, uniform)
    lw = weighted_causal_lm_loss(logits, targets, weighted)
    assert torch.isfinite(lu) and torch.isfinite(lw)
    assert not torch.allclose(lu, lw)
