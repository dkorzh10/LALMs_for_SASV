"""Shared SALMON prompt selection and speech embedding wrap helpers."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

PromptValue = Union[str, List[str]]


def _batch_size(samples: Dict[str, Any]) -> int:
    for key in ("text", "gt", "task", "answer", "audio_ids"):
        val = samples.get(key)
        if isinstance(val, list) and len(val) > 0:
            return len(val)
    return 1


def pick_prompt_template(
    prompt_templates: List[str],
    sample_index: int = 0,
    *,
    deterministic: bool = False,
) -> str:
    """Pick a prompt template; eval uses the first template for reproducibility."""
    if not prompt_templates:
        raise ValueError("prompt_templates is empty")
    if deterministic:
        return prompt_templates[0]
    return random.choice(prompt_templates)


def resolve_speech_prompts(
    salmon_model: Any,
    samples: Dict[str, Any],
    *,
    training: bool,
) -> Tuple[Optional[PromptValue], bool]:
    """Resolve prompt text for a batch without wrapping or mutating ``multi_prompt``.

    When ``salmon_model.wrap_collator_prompts`` is false, batch ``prompts`` from the
    collator are ignored (pre-bugfix #1 behavior). Only ``prompt_dict`` from
    ``prompt_path`` is used for speech wrapping.
    """
    wrap_collator_prompts = getattr(salmon_model, "wrap_collator_prompts", True)
    batch_prompts = samples.get("prompts") if wrap_collator_prompts else None
    batch_size = _batch_size(samples)

    if isinstance(batch_prompts, list) and len(batch_prompts) > 0:
        use_multi = len(batch_prompts) > 1 or len(set(batch_prompts)) > 1
        if use_multi:
            return batch_prompts, True
        return batch_prompts[0], False

    if isinstance(batch_prompts, str) and batch_prompts:
        use_multi = batch_size > 1
        if use_multi:
            return [batch_prompts] * batch_size, True
        return batch_prompts, False

    if salmon_model.prompt_dict:
        tasks = samples["task"]
        unique_tasks = list(set(tasks))
        use_multi = len(unique_tasks) > 1 or "QA" in unique_tasks
        if use_multi:
            if training:
                prompt = [random.choice(salmon_model.prompt_dict[t]) for t in tasks]
            else:
                prompt = [salmon_model.prompt_dict[t][0] for t in tasks]
            return prompt, True
        task_key = tasks[0]
        if training:
            return random.choice(salmon_model.prompt_dict[task_key]), False
        return salmon_model.prompt_dict[task_key][0], False

    return None, False


def wrap_speech_with_prompts(
    salmon_model: Any,
    speech_embeds: torch.Tensor,
    speech_atts: torch.Tensor,
    samples: Dict[str, Any],
    *,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrap speech embeddings with batch or fallback prompts.

    Restores ``salmon_model.multi_prompt`` after the call so eval/train flags do not leak.
    """
    prompt, use_multi = resolve_speech_prompts(salmon_model, samples, training=training)
    if prompt is None:
        return speech_embeds, speech_atts

    saved_multi_prompt = salmon_model.multi_prompt
    try:
        salmon_model.multi_prompt = use_multi
        return salmon_model.prompt_wrap(
            speech_embeds, speech_atts, prompt, multi_prompt=use_multi
        )
    finally:
        salmon_model.multi_prompt = saved_multi_prompt
