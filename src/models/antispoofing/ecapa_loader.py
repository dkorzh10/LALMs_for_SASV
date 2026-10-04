"""Load ECAPA2 speaker embedding model (Jenthe/ECAPA2 on Hugging Face)."""

from __future__ import annotations

import os

import torch


def load_ecapa2(
    device: torch.device,
    *,
    repo_id: str = "Jenthe/ECAPA2",
    filename: str = "ecapa2.pt",
    weights_path: str = "",
) -> torch.jit.ScriptModule:
    """Load TorchScript ECAPA2; uses HF cache when ``weights_path`` is empty."""
    if weights_path and os.path.isfile(weights_path):
        model_file = weights_path
    else:
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(repo_id=repo_id, filename=filename)

    model = torch.jit.load(model_file, map_location=device).to(device)
    model.eval()
    return model
