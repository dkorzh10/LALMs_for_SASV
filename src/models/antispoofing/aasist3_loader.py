"""Load AASIST3 CM from the local AASIST3 repository."""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn


def load_aasist3_checkpoint(
    ckpt_path: str,
    aasist3_repo: str,
    device: torch.device,
    *,
    hf_model_id: str = "MTUCI/AASIST3",
    w2v_cache_dir: str = "",
    load_pretrained: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate ``aasist3`` and optionally load fine-tuned weights.

    Args:
        ckpt_path: Path to ``.safetensors`` / ``.pth`` checkpoint, or empty to use
            Hugging Face weights only.
        aasist3_repo: Root of the AASIST3 repository (added to ``sys.path``).
        device: Target device.
        hf_model_id: Hugging Face model id for ``from_pretrained``.
        w2v_cache_dir: Cache dir for Wav2Vec2 weights; defaults to ``{repo}/weights``.
        load_pretrained: Passed to ``aasist3`` when building without HF hub weights.
    """
    repo = os.path.abspath(aasist3_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from model import aasist3

    cache_dir = w2v_cache_dir or os.path.join(repo, "weights")
    os.makedirs(cache_dir, exist_ok=True)

    model = aasist3.from_pretrained(hf_model_id, cache_dir=cache_dir)
    if ckpt_path:
        ckpt_path = os.path.abspath(ckpt_path)
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(ckpt_path)
        else:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(raw, dict):
                state = raw.get("model_state_dict", raw.get("model", raw))
            else:
                state = raw
        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    nb_samp = int(getattr(model, "d_args", {}).get("nb_samp", 64600))
    model.to(device)
    model.eval()
    return model, {"samples": nb_samp}
