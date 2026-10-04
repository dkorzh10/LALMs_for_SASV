"""
Nonlinear score-level fusion for SASV baselines.

Frozen experts produce scalar scores per pair:
  - **ECAPA / wav2vec2 (ASV)**: cosine similarity between enroll and query embeddings
  - **AASIST (CM)**: P(bonafide), P(spoof) on the query

``SASVScoreFusionHead`` maps these scores to 3-way logits (yes / no / gen) via
expert-specific branches, optional multiplicative interactions, and a fusion MLP.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Layout used by sasv_ecapa_w2v_aasist (column order in feature tensor)
FEATURE_NAMES_ECAPA_W2V_CM: Tuple[str, ...] = (
    "ecapa_cos",
    "w2v_cos",
    "bonafide_prob",
    "spoof_prob",
)
# Layout used by sasv_w2v_aasist
FEATURE_NAMES_W2V_CM: Tuple[str, ...] = ("w2v_cos", "bonafide_prob", "spoof_prob")

_LAYOUT_DIMS: Dict[str, Tuple[int, int, int]] = {
    # layout -> (input_dim, asv_branch_in, cm_branch_in)
    "ecapa_w2v_cm": (4, 2, 2),
    "w2v_cm": (3, 1, 2),
}


def _parse_hidden_dims(raw: Any, default: Sequence[int]) -> Tuple[int, ...]:
    if raw is None:
        return tuple(default)
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    return (int(raw),)


class _ExpertBranch(nn.Module):
    """Small nonlinear transform for one expert group (ASV cosines or CM probs)."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SASVScoreFusionHead(nn.Module):
    """Nonlinear score-level fusion head over expert scalar outputs.

    Args:
        layout: ``ecapa_w2v_cm`` (4 scores) or ``w2v_cm`` (3 scores).
        expert_hidden_dim: Hidden size for ASV / CM expert branches.
        fusion_hidden_dims: Hidden sizes of the fusion MLP (before 3-way logits).
        dropout: Dropout probability.
        use_interactions: Append pairwise products of ASV×CM scores.
        use_simple_fallback: If True and fusion_hidden_dims is empty, use one
            Linear→ReLU block (legacy ``fusion_mode: mlp`` behaviour).
    """

    def __init__(
        self,
        *,
        layout: str = "ecapa_w2v_cm",
        expert_hidden_dim: int = 64,
        fusion_hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        use_interactions: bool = True,
        use_simple_fallback: bool = False,
    ) -> None:
        super().__init__()
        if layout not in _LAYOUT_DIMS:
            raise ValueError(f"Unknown layout {layout!r}; use {list(_LAYOUT_DIMS)}")
        self.layout = layout
        in_dim, asv_in, cm_in = _LAYOUT_DIMS[layout]
        self.input_dim = in_dim
        self.use_interactions = bool(use_interactions)

        hidden_dims = _parse_hidden_dims(fusion_hidden_dims, (32,))
        if use_simple_fallback and len(hidden_dims) == 0:
            self.asv_branch = None
            self.cm_branch = None
            self.fusion = nn.Sequential(
                nn.Linear(in_dim, expert_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, 3),
            )
            self._interaction_dim = 0
            return

        self.asv_branch = _ExpertBranch(asv_in, expert_hidden_dim, dropout)
        self.cm_branch = _ExpertBranch(cm_in, expert_hidden_dim, dropout)
        self._interaction_dim = self._num_interactions(asv_in, cm_in) if use_interactions else 0

        fusion_in = expert_hidden_dim * 2 + self._interaction_dim
        layers: List[nn.Module] = []
        prev = fusion_in
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.fusion = nn.Sequential(*layers)

    @staticmethod
    def _num_interactions(asv_in: int, cm_in: int) -> int:
        return asv_in * cm_in

    def _split_experts(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layout == "ecapa_w2v_cm":
            asv = scores[:, :2]
            cm = scores[:, 2:4]
        else:
            asv = scores[:, :1]
            cm = scores[:, 1:3]
        return asv, cm

    def _interaction_features(self, asv: torch.Tensor, cm: torch.Tensor) -> torch.Tensor:
        """Pairwise products between ASV and CM score groups [B, asv_in * cm_in]."""
        parts: List[torch.Tensor] = []
        for i in range(asv.size(1)):
            for j in range(cm.size(1)):
                parts.append((asv[:, i] * cm[:, j]).unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Map expert scores [B, D] to SASV logits [B, 3]."""
        x = scores.float()
        if self.asv_branch is None:
            return self.fusion(x)

        asv, cm = self._split_experts(x)
        parts: List[torch.Tensor] = [self.asv_branch(asv), self.cm_branch(cm)]
        if self.use_interactions and self._interaction_dim > 0:
            parts.append(self._interaction_features(asv, cm))
        fused = torch.cat(parts, dim=-1)
        return self.fusion(fused)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        if self.layout == "ecapa_w2v_cm":
            return FEATURE_NAMES_ECAPA_W2V_CM
        return FEATURE_NAMES_W2V_CM


def build_score_fusion_head(
    baseline_cfg: Dict[str, Any],
    *,
    layout: str,
) -> nn.Module:
    """Build fusion head from ``additional_kwargs.sasv_baseline`` or ``w2v_aasist``."""
    sf = baseline_cfg.get("score_fusion", {}) or {}
    fusion_mode = str(baseline_cfg.get("fusion_mode", "mlp")).lower()

    expert_hidden = int(
        sf.get("expert_hidden_dim", baseline_cfg.get("fusion_hidden_dim", 64))
    )
    fusion_hidden = _parse_hidden_dims(
        sf.get("fusion_hidden_dims", baseline_cfg.get("fusion_hidden_dims")),
        (32,),
    )
    dropout = float(sf.get("dropout", baseline_cfg.get("fusion_dropout", 0.1)))
    use_interactions = bool(sf.get("use_interactions", True))

    # Legacy 2-layer MLP: fusion_mode=mlp without score_fusion block
    use_simple = fusion_mode == "mlp" and not sf
    if use_simple:
        in_dim = _LAYOUT_DIMS[layout][0]
        return nn.Sequential(
            nn.Linear(in_dim, expert_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, 3),
        )

    return SASVScoreFusionHead(
        layout=layout,
        expert_hidden_dim=expert_hidden,
        fusion_hidden_dims=fusion_hidden,
        dropout=dropout,
        use_interactions=use_interactions,
        use_simple_fallback=False,
    )


def stack_ecapa_w2v_cm_scores(
    ecapa_cos: torch.Tensor,
    w2v_cos: torch.Tensor,
    bonafide_prob: torch.Tensor,
    spoof_prob: torch.Tensor,
) -> torch.Tensor:
    """Stack [B] tensors into feature matrix [B, 4]."""
    return torch.stack([ecapa_cos, w2v_cos, bonafide_prob, spoof_prob], dim=-1)


def stack_w2v_cm_scores(
    w2v_cos: torch.Tensor,
    bonafide_prob: torch.Tensor,
    spoof_prob: torch.Tensor,
) -> torch.Tensor:
    """Stack [B] tensors into feature matrix [B, 3]."""
    return torch.stack([w2v_cos, bonafide_prob, spoof_prob], dim=-1)
