"""AdamW param groups: base (предобученный бэкбон) vs head (новые слои SASV / классификаторы)."""

from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

# Префиксы имён параметров после снятия ``module.`` (DDP)
_HEAD_PARAM_PREFIXES = (
    "answer_head.",
    "bonafide_spoof_head.",
    "speaker_projector.",
    "arcface_head.",
    "fusion_head.",
    "logit_bias",
)


def is_head_param(param_name: str) -> bool:
    n = param_name
    if n.startswith("module."):
        n = n[len("module.") :]
    return any(n.startswith(p) for p in _HEAD_PARAM_PREFIXES)


def split_trainable_params(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """(base_wd, base_no_wd, head_wd, head_no_wd)."""
    base_wd, base_no_wd = [], []
    head_wd, head_no_wd = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim < 2 or "bias" in name or "ln" in name or "bn" in name
        if is_head_param(name):
            (head_no_wd if no_wd else head_wd).append(p)
        else:
            (base_no_wd if no_wd else base_wd).append(p)
    return base_wd, base_no_wd, head_wd, head_no_wd


def build_adamw_param_groups(
    model: nn.Module,
    base_cfg: Dict[str, Any],
    head_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Группы для ``torch.optim.AdamW`` с полем ``lr_ratio_to_base`` (голова = базовый LR × ratio)."""
    base_wd, base_no_wd, head_wd, head_no_wd = split_trainable_params(model)
    base_lr = float(base_cfg.get("init_lr", base_cfg.get("lr", 1e-4)))
    base_wd_val = float(base_cfg.get("weight_decay", 0.05))

    groups: List[Dict[str, Any]] = []

    def add_group(params: List[nn.Parameter], weight_decay: float, lr_ratio: float) -> None:
        if not params:
            return
        groups.append(
            {
                "params": params,
                "weight_decay": weight_decay,
                "lr_ratio_to_base": lr_ratio,
            }
        )

    if head_cfg:
        head_lr = float(head_cfg.get("init_lr", base_lr))
        ratio = (head_lr / base_lr) if base_lr > 0 else 1.0
        head_wd_val = float(head_cfg.get("weight_decay", base_wd_val))
        add_group(base_wd, base_wd_val, 1.0)
        add_group(base_no_wd, 0.0, 1.0)
        add_group(head_wd, head_wd_val, ratio)
        add_group(head_no_wd, 0.0, ratio)
    else:
        add_group(base_wd, base_wd_val, 1.0)
        add_group(base_no_wd, 0.0, 1.0)
        add_group(head_wd, base_wd_val, 1.0)
        add_group(head_no_wd, 0.0, 1.0)

    if not groups:
        raise ValueError("No trainable parameters in model for optimizer.")

    return groups
