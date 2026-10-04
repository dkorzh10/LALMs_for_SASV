"""PEFT helpers compatible across peft versions."""
from __future__ import annotations

import inspect
from typing import Any

from peft import LoraConfig


def make_lora_config(**kwargs: Any) -> LoraConfig:
    """Build ``LoraConfig``, dropping kwargs unsupported by the installed peft."""
    allowed = set(inspect.signature(LoraConfig.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return LoraConfig(**filtered)
