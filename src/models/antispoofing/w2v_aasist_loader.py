"""Load a Wav2Vec2-AASIST countermeasure checkpoint."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class _AttrDict(dict):
    """Minimal EasyDict replacement (attribute + dict access)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, _AttrDict):
                self[key] = _AttrDict(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_w2v_aasist_checkpoint(
    ckpt_path: str,
    antispoofing_repo: str,
    device: torch.device,
    *,
    xlsr_weights_path: str = "",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate ``Wav2vec2AASISTLinearTa`` and load a training checkpoint.

    Args:
        ckpt_path: Path to ``running_checkpoint.pth`` (or similar).
        antispoofing_repo: Repository root added to ``sys.path``.
        device: Target device.
        xlsr_weights_path: Optional override for XLSR weights; when empty, uses
            ``{repo}/weights/xlsr2_300m_ta.pt`` if present.
    """
    repo = os.path.abspath(antispoofing_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from models.wav2vec2_aasist_linear_ta import Wav2vec2AASISTLinearTa

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = _AttrDict(checkpoint.get("model_config", {}))
    if xlsr_weights_path:
        model_cfg.weights_path = xlsr_weights_path
    elif not model_cfg.get("weights_path"):
        default_xlsr = os.path.join(repo, "weights", "xlsr2_300m_ta.pt")
        if os.path.isfile(default_xlsr):
            model_cfg.weights_path = default_xlsr

    model = Wav2vec2AASISTLinearTa(model_cfg)
    state = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, dict(checkpoint.get("audio_config", {}))
