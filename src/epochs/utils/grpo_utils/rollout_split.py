"""
Chunked generation and logits for OOM: split by rollouts to reduce peak memory.

Flow:
1. Generate rollouts in chunks (e.g. 2 at a time) -> collect texts (lightweight), completion_ids
2. Score + compute advantages (lightweight, once we have texts)
3. Compute logits and backward in chunks (split by rollout groups, gradient accumulation)

All functions are pure/modular and testable.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def chunk_ranges_for_rollouts(
    batch_size: int, num_gens: int, chunk_size: int
) -> List[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) for each chunk in completion_ids (gen-major order).
    index = g * batch_size + b.
    chunk_size = number of generations per chunk; each chunk has chunk_size * batch_size rollouts.
    """
    if chunk_size >= num_gens:
        return [(0, batch_size * num_gens)]
    ranges = []
    for chunk_start in range(0, num_gens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_gens)
        start_idx = chunk_start * batch_size
        end_idx = chunk_end * batch_size
        ranges.append((start_idx, end_idx))
    return ranges


def chunk_ranges_by_rollout_count(
    total_rollouts: int, max_rollouts_per_chunk: int
) -> List[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) for each chunk, where each chunk has at most
    max_rollouts_per_chunk rollouts. Enables true 1-rollout-at-a-time processing.
    """
    if max_rollouts_per_chunk >= total_rollouts:
        return [(0, total_rollouts)]
    return [
        (i, min(i + max_rollouts_per_chunk, total_rollouts))
        for i in range(0, total_rollouts, max_rollouts_per_chunk)
    ]


def _get_pad_token_id(model) -> int:
    pad_id = 0
    if hasattr(model, "model") and hasattr(model.model, "llama_tokenizer"):
        tok = model.model.llama_tokenizer
        raw = tok.pad_token_id if tok.pad_token_id is not None else 0
        pad_id = int(raw) if isinstance(raw, int) else 0
    elif hasattr(model, "get_pad_token_id"):
        raw = model.get_pad_token_id()
        pad_id = int(raw) if isinstance(raw, int) else 0
    return pad_id


def _reorder_sample_major_to_gen_major(
    texts: List[str],
    completion_ids: torch.Tensor,
    batch_size: int,
    num_gens: int,
) -> Tuple[List[str], List[torch.Tensor]]:
    """Convert sample-major [s0_g0, s0_g1, s1_g0, ...] to gen-major rollout lists."""
    all_texts: List[str] = []
    all_completion_ids_list: List[torch.Tensor] = []
    for g in range(num_gens):
        gen_texts = []
        gen_ids = []
        for b in range(batch_size):
            idx = b * num_gens + g
            gen_texts.append(texts[idx])
            gen_ids.append(completion_ids[idx])
        all_texts.extend(gen_texts)
        all_completion_ids_list.append(torch.stack(gen_ids, dim=0))
    return all_texts, all_completion_ids_list


def generate_rollouts_batched(
    model,
    batch: Dict[str, Any],
    num_gens: int,
    gen_cfg: Dict[str, Any],
    prompts: Optional[List] = None,
    amp: bool = True,
) -> Tuple[List[str], torch.Tensor, int, int]:
    """
    Generate num_gens rollouts per sample in one batched call (encode audio once).
    Returns (all_texts, all_completion_ids, pad_id, batch_size) in gen-major order.
    """
    prompts = prompts or batch.get("prompts")
    batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
    autocast_dtype = torch.float16
    if amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16

    chunk_gen_cfg = dict(gen_cfg)
    chunk_gen_cfg["num_return_sequences"] = num_gens

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp, dtype=autocast_dtype):
            texts, completion_ids, _ = model.generate(
                batch,
                chunk_gen_cfg,
                prompts=prompts,
                return_outputs=True,
                return_logits=False,
            )

    all_texts, all_completion_ids_list = _reorder_sample_major_to_gen_major(
        texts, completion_ids, batch_size, num_gens
    )
    pad_id = _get_pad_token_id(model)
    max_len = max(t.size(1) for t in all_completion_ids_list)
    padded = [
        F.pad(t, (0, max_len - t.size(1)), value=pad_id) for t in all_completion_ids_list
    ]
    all_completion_ids = torch.cat(padded, dim=0)
    return all_texts, all_completion_ids, pad_id, batch_size


def generate_rollouts_in_chunks(
    model,
    batch: Dict[str, Any],
    num_gens: int,
    gen_cfg: Dict[str, Any],
    prompts: Optional[List] = None,
    chunk_size: int = 1,
    amp: bool = True,
) -> Tuple[List[str], torch.Tensor, int, int]:
    """
    Generate num_gens rollouts per sample in chunks of chunk_size to reduce peak memory.
    Returns (all_texts, all_completion_ids, pad_id, batch_size).
    Data layout: gen-major order, index = g * batch_size + b.
    """
    prompts = prompts or batch.get("prompts")
    batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
    autocast_dtype = torch.float16
    if amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16

    all_texts: List[str] = []
    all_completion_ids_list: List[torch.Tensor] = []
    pad_id = _get_pad_token_id(model)

    for chunk_start in range(0, num_gens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_gens)
        n_this_chunk = chunk_end - chunk_start

        chunk_gen_cfg = dict(gen_cfg)
        chunk_gen_cfg["num_return_sequences"] = n_this_chunk

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=amp, dtype=autocast_dtype):
                texts, completion_ids, _ = model.generate(
                    batch,
                    chunk_gen_cfg,
                    prompts=prompts,
                    return_outputs=True,
                    return_logits=False,
                )

        chunk_texts, chunk_ids = _reorder_sample_major_to_gen_major(
            texts, completion_ids, batch_size, n_this_chunk
        )
        all_texts.extend(chunk_texts)
        all_completion_ids_list.extend(chunk_ids)

        if chunk_end < num_gens:
            torch.cuda.empty_cache()

    max_len = max(t.size(1) for t in all_completion_ids_list)
    padded = [
        F.pad(t, (0, max_len - t.size(1)), value=pad_id) for t in all_completion_ids_list
    ]
    all_completion_ids = torch.cat(padded, dim=0)
    return all_texts, all_completion_ids, pad_id, batch_size


def compute_grpo_loss_for_chunk(
    chunk_completion_ids: torch.Tensor,
    chunk_advantages: torch.Tensor,
    ref_logits: torch.Tensor,
    current_logits: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO loss for a chunk of rollouts. Returns (pg_loss, kl_div).
    Handles 2D and 3D logits.
    """
    T = chunk_completion_ids.size(1)
    if ref_logits.dim() == 3:
        ref_logits = ref_logits[:, -T:, :]
        current_logits = current_logits[:, -T:, :]
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    token_log_probs = torch.gather(
        current_log_probs, dim=-1, index=chunk_completion_ids.unsqueeze(-1)
    ).squeeze(-1)
    ref_token_log_probs = torch.gather(
        ref_log_probs, dim=-1, index=chunk_completion_ids.unsqueeze(-1)
    ).squeeze(-1)
    mask = (chunk_completion_ids != pad_id).float()
    total_log_probs = (token_log_probs * mask).sum(dim=-1)
    ref_total_log_probs = (ref_token_log_probs * mask).sum(dim=-1)
    pg_loss = -(chunk_advantages * total_log_probs).mean()
    kl_div = (total_log_probs - ref_total_log_probs).mean()
    return pg_loss, kl_div
