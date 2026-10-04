"""
Extract and save LLaMA hidden-state fusion weights (ArcFace multi-layer pooling)
from SASV salmon checkpoints when plotting an experiment run.

The trainable vector is `arcface_layer_weights` on SASVSalmonModel; forward uses
softmax over it with `arcface_embedding_layers` indices into HF `hidden_states`.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore


def _find_arcface_layer_weights_key(state_dict: Dict[str, Any]) -> Optional[str]:
    if not state_dict:
        return None
    if "arcface_layer_weights" in state_dict:
        return "arcface_layer_weights"
    for k in state_dict:
        if k.endswith("arcface_layer_weights"):
            return k
    return None


def _load_arcface_config(experiment_dir: str) -> Tuple[List[int], str]:
    config_path = os.path.join(experiment_dir, "config_resolved.yaml")
    layers: List[int] = [-1]
    pooling = "learned_weighted_sum"
    if not os.path.isfile(config_path):
        return layers, pooling
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        salmon = (cfg.get("Model") or {}).get("additional_kwargs", {}).get("salmon", {}) or {}
        raw = salmon.get("arcface_embedding_layers", [-1])
        if isinstance(raw, list) and raw:
            layers = [int(x) for x in raw]
        pooling = str(salmon.get("arcface_layer_pooling", "learned_weighted_sum"))
    except (yaml.YAMLError, OSError, TypeError, ValueError):
        pass
    return layers, pooling


def _checkpoint_sort_key(path: str) -> Tuple[int, str]:
    """Higher epoch first; then basename for stability."""
    m = re.search(r"epoch_(\d+)", os.path.basename(path))
    epoch = int(m.group(1)) if m else -1
    return (epoch, os.path.basename(path))


def save_arcface_llama_layer_weights_from_run(experiment_dir: str, output_dir: str) -> bool:
    """
    Load all *.pt under experiment_dir/checkpoints, read arcface_layer_weights if present,
    write JSON + a bar chart (softmax weights) for the highest-epoch checkpoint.

    Returns True if at least one checkpoint contained arcface_layer_weights.
    """
    if torch is None:
        print("plotter_arcface_layers: torch not available, skipping layer weights export.")
        return False

    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False

    layers, pooling = _load_arcface_config(experiment_dir)
    pattern = os.path.join(ckpt_dir, "*.pt")
    ckpt_files = sorted(glob.glob(pattern), key=_checkpoint_sort_key, reverse=True)
    if not ckpt_files:
        return False

    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for path in ckpt_files:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"plotter_arcface_layers: could not load {path}: {e}")
            continue
        state = ckpt.get("model")
        if not isinstance(state, dict):
            continue
        key = _find_arcface_layer_weights_key(state)
        if key is None:
            continue
        w = state[key].detach().float().view(-1)
        if w.numel() != len(layers):
            print(
                f"plotter_arcface_layers: {os.path.basename(path)}: "
                f"len(arcface_layer_weights)={w.numel()} != len(arcface_embedding_layers)={len(layers)}; "
                "using generic labels."
            )
        sm = torch.softmax(w, dim=0)
        results.append(
            {
                "checkpoint": os.path.basename(path),
                "checkpoint_path": path,
                "state_dict_key": key,
                "epoch": ckpt.get("epoch"),
                "arcface_embedding_layers": layers,
                "arcface_layer_pooling": pooling,
                "layer_weight_logits": w.tolist(),
                "layer_weight_softmax": sm.tolist(),
            }
        )

    if not results:
        return False

    out_json = os.path.join(output_dir, "llama_arcface_layer_weights.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"plotter_arcface_layers: wrote {out_json}")

    latest = results[0]
    n = len(latest["layer_weight_softmax"])
    labels = [str(layers[i]) if i < len(layers) else str(i) for i in range(n)]
    if plt is not None:
        plt.figure(figsize=(max(10, n * 0.6), 5))
        plt.bar(range(n), latest["layer_weight_softmax"], tick_label=labels, color="steelblue")
        plt.xlabel("hidden_states index (from config arcface_embedding_layers)")
        plt.ylabel("Softmax weight")
        plt.title(
            f"LLaMA ArcFace layer fusion weights\n{latest['checkpoint']} ({pooling})",
            fontsize=11,
        )
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(output_dir, "llama_arcface_layer_weights.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"plotter_arcface_layers: wrote {out_png}")
    else:
        print("plotter_arcface_layers: matplotlib not available, skipped PNG.")

    return True
