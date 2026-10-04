"""Train sklearn decision tree fusion from unified_training config."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..dataloaders.builder import get_dataloader
from ..models.sasv_decision_tree_fusion import (
    CLASS_NAMES,
    FEATURE_NAMES,
    SASVDecisionTreeFusion,
    TreeTrainResult,
    label_to_idx,
)
from ..models.sasv_ecapa_w2v_aasist_fusion import SASVEcapaW2vAasistFusionModel


def _tree_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    ak = config.get("Model", {}).get("additional_kwargs", {}).get("sasv_baseline", {})
    return ak.get("decision_tree", {}) or {}


def default_tree_output_path(config: Dict[str, Any], output_dir: str) -> str:
    ak = config.get("Model", {}).get("additional_kwargs", {}).get("sasv_baseline", {})
    explicit = str(ak.get("fusion_tree_path", "") or "").strip()
    if explicit:
        return explicit
    tree_cfg = _tree_cfg(config)
    name = str(tree_cfg.get("output_name", "fusion_tree.joblib"))
    return os.path.join(output_dir, name)


@torch.inference_mode()
def extract_features(
    model: SASVEcapaW2vAasistFusionModel,
    dataloader,
    device: torch.device,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        enroll = batch["enroll_wav"].to(device)
        query = batch["query_wav"].to(device)
        feats = model.extract_fusion_features(enroll, query)
        labels = batch.get("answer") or batch.get("gt") or batch.get("text")
        if not isinstance(labels, list):
            labels = [labels]

        feats_np = feats.detach().cpu().numpy()
        for i, lab in enumerate(labels):
            try:
                ys.append(label_to_idx(lab))
            except ValueError:
                continue
            xs.append(feats_np[i])

    if not xs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64)


def _build_loader(
    dataset_path: str,
    config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    is_train: bool,
):
    return get_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        max_samples=max_samples,
        task_type="hard_label",
        model_name="sasv_ecapa_w2v_aasist",
        audio_cfg=config.get("Audio", {}),
        is_train=is_train,
        split="train" if is_train else "val",
    )


def train_decision_tree_from_config(
    config: Dict[str, Any],
    output_dir: str,
    device: torch.device,
    *,
    local_rank: int = 0,
) -> str:
    """Extract biometric features and fit a decision tree. Returns path to .joblib."""
    if local_rank != 0:
        return default_tree_output_path(config, output_dir)

    data_cfg = config.get("Datasets", {})
    runner_cfg = config.get("Runner", {})
    tree_cfg = _tree_cfg(config)

    train_path = data_cfg.get("dataset_train_path", "")
    val_path = data_cfg.get("dataset_val_path", "")
    if not train_path:
        raise ValueError("Datasets.dataset_train_path is required for fusion_mode=tree training")

    batch_size = int(
        tree_cfg.get("batch_size", runner_cfg.get("SFT", {}).get("batch_size_train", 16))
    )
    num_workers = int(tree_cfg.get("num_workers", runner_cfg.get("num_workers", 4)))
    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_valid_samples")

    out_path = default_tree_output_path(config, output_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cache_path = str(tree_cfg.get("cache_features", "") or "").strip()
    if cache_path and not os.path.isabs(cache_path):
        cache_path = os.path.join(output_dir, cache_path)

    model_cfg = copy.deepcopy(config.get("Model", {}))
    ak = model_cfg.setdefault("additional_kwargs", {}).setdefault("sasv_baseline", {})
    ak["fusion_ckpt"] = ""
    ak["fusion_mode"] = "mlp"

    print(
        f"[tree] Loading backbones on {device} (cm_backend={ak.get('cm_backend', 'lkb')})...",
        flush=True,
    )
    model = SASVEcapaW2vAasistFusionModel(model_cfg)
    model.eval()
    model.to(device)

    if cache_path and os.path.isfile(cache_path):
        print(f"[tree] Loading cached features from {cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=True)
        x_train, y_train = cached["x_train"], cached["y_train"]
        x_val, y_val = cached["x_val"], cached["y_val"]
    else:
        train_loader = _build_loader(
            train_path, config, batch_size, num_workers, max_train, is_train=True
        )
        print(
            f"[tree] Extracting train features ({len(train_loader.dataset)} pairs)...",
            flush=True,
        )
        x_train, y_train = extract_features(model, train_loader, device, "train")

        if val_path:
            val_loader = _build_loader(
                val_path, config, batch_size, num_workers, max_val, is_train=False
            )
            print(
                f"[tree] Extracting val features ({len(val_loader.dataset)} pairs)...",
                flush=True,
            )
            x_val, y_val = extract_features(model, val_loader, device, "val")
        else:
            x_val = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
            y_val = np.zeros(0, dtype=np.int64)

        if cache_path:
            np.savez_compressed(
                cache_path, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val
            )
            print(f"[tree] Saved feature cache to {cache_path}", flush=True)

    class_weight = tree_cfg.get("class_weight", "balanced")
    class_weight = None if class_weight in (None, "", "none", "None") else class_weight

    tree, result = SASVDecisionTreeFusion.train(
        x_train,
        y_train,
        x_val if len(y_val) else None,
        y_val if len(y_val) else None,
        max_depth=tree_cfg.get("max_depth", 8),
        min_samples_leaf=int(tree_cfg.get("min_samples_leaf", 20)),
        min_samples_split=int(tree_cfg.get("min_samples_split", 40)),
        class_weight=class_weight,
        random_state=int(tree_cfg.get("random_state", 42)),
    )

    _print_result(result)
    print("\n--- Decision tree rules ---\n", flush=True)
    print(tree.export_rules(), flush=True)

    metadata = {
        "train_dataset": train_path,
        "val_dataset": val_path,
        "cm_backend": ak.get("cm_backend", "lkb"),
        "tree_hparams": {
            k: tree_cfg.get(k)
            for k in (
                "max_depth",
                "min_samples_leaf",
                "min_samples_split",
                "class_weight",
                "random_state",
            )
        },
        "metrics": {k: getattr(result, k) for k in result.__dataclass_fields__},
    }
    tree.save(out_path, metadata=metadata)

    metrics_path = os.path.join(output_dir, "tree_train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata["metrics"], f, indent=2)

    print(f"[tree] Saved model to {out_path}", flush=True)
    print(f"[tree] Metrics written to {metrics_path}", flush=True)
    return out_path


def _print_result(result: TreeTrainResult) -> None:
    print("\n=== Decision tree training ===", flush=True)
    for key, val in result.__dict__.items():
        if key == "val_report":
            continue
        print(f"  {key}: {val}", flush=True)
    if result.val_report:
        print("\nValidation report:\n", flush=True)
        print(result.val_report, flush=True)
