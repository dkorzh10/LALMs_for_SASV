"""SASV cascade threshold grid search (AASIST CM + ECAPA ASV scores)."""

from __future__ import annotations

import copy
import csv
import json
import os
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError as exc:
    raise ImportError("sasv_threshold_gridsearch requires soundfile") from exc

from ..epochs.utils.sasv_metrics import compute_all_sasv_metrics, compute_det_curve


def parse_float_list(value: Union[str, float, List, Tuple, None], default: List[float]) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(value)]


def _grid_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    runner = config.get("Runner", {}) or {}
    return runner.get("FusionGridsearch") or runner.get("ThresholdGrid") or {}


def is_fusion_gridsearch_run_type(run_type: str) -> bool:
    return str(run_type).lower() in ("fusion_gridsearch", "threshold_grid")


def _gt_to_label(gt: Any) -> Optional[str]:
    raw = str(gt).strip().lower()
    if raw in ("verified", "yes"):
        return "yes"
    if raw in ("rejected", "no"):
        return "no"
    if raw in ("spoof", "gen"):
        return "gen"
    if raw in ("yes", "no", "gen"):
        return raw
    return None


def _load_pairs_json(path: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("pairs", "data", "samples"):
            if key in data:
                data = data[key]
                break
    if max_samples is not None:
        data = data[:max_samples]
    return data


def _read_wav_mono(path: str, target_sr: int) -> Optional[torch.Tensor]:
    if not path or not os.path.isfile(path):
        return None
    try:
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        wav_np = wav_np.mean(axis=1)
        if sr != target_sr:
            n = max(int(round(len(wav_np) * target_sr / sr)), 1)
            x_old = np.linspace(0.0, 1.0, num=len(wav_np), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
            wav_np = np.interp(x_new, x_old, wav_np).astype(np.float32)
        return torch.from_numpy(np.asarray(wav_np, dtype=np.float32).flatten())
    except Exception:
        return None


def _crop_or_pad(wav: torch.Tensor, max_len: int, *, center: bool) -> torch.Tensor:
    t = wav.numel()
    if t >= max_len:
        start = (t - max_len) // 2 if center else 0
        return wav[start : start + max_len]
    if t == 0:
        return torch.zeros(max_len, dtype=torch.float32)
    reps = int(np.ceil(max_len / max(1, t)))
    return wav.repeat(reps)[:max_len]


class _SasvPairsDataset(Dataset):
    """SASV pairs via soundfile (no torchaudio / VAD)."""

    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        *,
        target_sr: int,
        enroll_max_len: int,
        query_max_len: int,
        center_crop: bool = True,
    ) -> None:
        self.pairs = pairs
        self.target_sr = target_sr
        self.enroll_max_len = enroll_max_len
        self.query_max_len = query_max_len
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.pairs[idx]
        ref = (item.get("reference_audios") or [{}])[0]
        qry = (item.get("query_audios") or [{}])[0]
        enroll = _read_wav_mono(ref.get("original_path") or ref.get("path", ""), self.target_sr)
        query = _read_wav_mono(qry.get("original_path") or qry.get("path", ""), self.target_sr)
        if enroll is None:
            enroll = torch.zeros(self.enroll_max_len)
        if query is None:
            query = torch.zeros(self.query_max_len)
        enroll = _crop_or_pad(enroll, self.enroll_max_len, center=self.center_crop)
        query = _crop_or_pad(query, self.query_max_len, center=self.center_crop)
        gt = _gt_to_label(item.get("gt", "")) or "no"
        return {
            "enroll_wav": enroll,
            "query_wav": query,
            "answer": gt,
            "gt": gt,
            "audio_ids": item.get("task_id") or item.get("pair_id") or f"idx_{idx}",
        }


def _collate_sasv_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    from torch.nn.utils.rnn import pad_sequence

    enroll = pad_sequence([b["enroll_wav"] for b in batch], batch_first=True)
    query = pad_sequence([b["query_wav"] for b in batch], batch_first=True)
    return {
        "enroll_wav": enroll,
        "query_wav": query,
        "answer": [b["answer"] for b in batch],
        "gt": [b["gt"] for b in batch],
        "audio_ids": [b["audio_ids"] for b in batch],
    }


def _audio_limits(audio_cfg: Dict[str, Any], is_train: bool) -> Tuple[int, int, int]:
    target_sr = int(audio_cfg.get("target_sr", 16000))
    max_sec = float(audio_cfg.get("max_len_sec", 6.0))
    if is_train:
        enroll_sec = float(audio_cfg.get("train_enroll_max_len_sec", max_sec))
        query_sec = float(audio_cfg.get("train_query_max_len_sec", max_sec))
    else:
        enroll_sec = float(audio_cfg.get("test_enroll_max_len_sec", max_sec))
        query_sec = float(audio_cfg.get("test_query_max_len_sec", max_sec))
    return (
        target_sr,
        int(round(enroll_sec * target_sr)),
        int(round(query_sec * target_sr)),
    )


def compute_eer_sasv(labels: np.ndarray, yes_probs: np.ndarray) -> Tuple[float, float]:
    labels = np.asarray(labels)
    yes_probs = np.asarray(yes_probs, dtype=np.float64)
    binary = (labels == "yes").astype(np.int64)
    if len(np.unique(binary)) < 2:
        return float("nan"), float("nan")
    pos = yes_probs[binary == 1]
    neg = yes_probs[binary == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")
    frr, far, thr = compute_det_curve(pos, neg)
    idx = int(np.argmin(np.abs(frr - far)))
    return float((frr[idx] + far[idx]) / 2), float(thr[idx])


def cascade_soft_scores(
    ecapa_cos: np.ndarray,
    bonafide_prob: np.ndarray,
    spoof_prob: np.ndarray,
    *,
    cm_threshold: float,
    ecapa_threshold: float,
    q: float,
    w2v_cos: Optional[np.ndarray] = None,
    asv_signal: str = "ecapa",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ecapa_cos = np.asarray(ecapa_cos, dtype=np.float64)
    bonafide_prob = np.asarray(bonafide_prob, dtype=np.float64)
    spoof_prob = np.asarray(spoof_prob, dtype=np.float64)

    if asv_signal == "ecapa":
        asv = ecapa_cos
    elif asv_signal == "w2v" and w2v_cos is not None:
        asv = np.asarray(w2v_cos, dtype=np.float64)
    elif asv_signal == "mean" and w2v_cos is not None:
        asv = 0.5 * (ecapa_cos + np.asarray(w2v_cos, dtype=np.float64))
    else:
        asv = ecapa_cos

    cm_safe = np.clip(bonafide_prob, 1e-8, 1.0)
    fused = asv * np.power(cm_safe, q)
    is_spoof = bonafide_prob < cm_threshold

    yes_prob = np.where(is_spoof, 0.0, np.clip(fused, 0.0, 1.0))
    gen_prob = np.clip(spoof_prob, 0.0, 1.0)
    return yes_prob, gen_prob, asv


@torch.inference_mode()
def extract_scores(
    config: Dict[str, Any],
    dataset_path: str,
    *,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    device: torch.device,
    cache_path: str,
    force_recompute: bool,
) -> Dict[str, np.ndarray]:
    if cache_path and os.path.isfile(cache_path) and not force_recompute:
        print(f"[FusionGridsearch] Loading cached scores from {cache_path}", flush=True)
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    from ..models.sasv_ecapa_w2v_aasist_fusion import SASVEcapaW2vAasistFusionModel

    model_cfg = copy.deepcopy(config.get("Model", {}))
    ak = model_cfg.setdefault("additional_kwargs", {}).setdefault("sasv_baseline", {})
    ak["fusion_mode"] = "mlp"
    ak["fusion_ckpt"] = ""

    print(f"[FusionGridsearch] Loading backbones on {device}...", flush=True)
    model = SASVEcapaW2vAasistFusionModel(model_cfg)
    model.eval()
    model.to(device)

    audio_cfg = config.get("Audio", {}) or {}
    target_sr, enroll_max_len, query_max_len = _audio_limits(audio_cfg, is_train=False)
    pairs = _load_pairs_json(dataset_path, max_samples)
    dataset = _SasvPairsDataset(
        pairs,
        target_sr=target_sr,
        enroll_max_len=enroll_max_len,
        query_max_len=query_max_len,
        center_crop=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_sasv_pairs,
        pin_memory=device.type == "cuda",
    )
    print(
        f"[FusionGridsearch] {len(dataset)} pairs (soundfile loader); "
        f"enroll_len={enroll_max_len}, query_len={query_max_len}",
        flush=True,
    )

    labels: List[str] = []
    ecapa: List[float] = []
    w2v: List[float] = []
    bonafide: List[float] = []
    spoof: List[float] = []
    task_ids: List[str] = []

    for batch in tqdm(loader, desc="extract scores"):
        enroll = batch["enroll_wav"].to(device)
        query = batch["query_wav"].to(device)
        feats, ecapa_cos_t, w2v_cos_t, bonafide_t = model._fusion_features(enroll, query)

        ecapa_cos = ecapa_cos_t.detach().cpu().numpy()
        w2v_cos = w2v_cos_t.detach().cpu().numpy()
        bon = bonafide_t.detach().cpu().numpy()
        spoof_prob = feats[:, 3].detach().cpu().numpy()

        gt_list = batch.get("answer") or batch.get("gt")
        if not isinstance(gt_list, list):
            gt_list = [gt_list]
        ids = batch.get("audio_ids") or [""] * len(gt_list)

        for i, gt in enumerate(gt_list):
            lab = _gt_to_label(gt)
            if lab is None:
                continue
            labels.append(lab)
            ecapa.append(float(ecapa_cos[i]))
            w2v.append(float(w2v_cos[i]))
            bonafide.append(float(bon[i]))
            spoof.append(float(spoof_prob[i]))
            task_ids.append(str(ids[i] if i < len(ids) else ""))

    out = {
        "labels": np.array(labels, dtype=object),
        "ecapa_cos": np.array(ecapa, dtype=np.float64),
        "w2v_cos": np.array(w2v, dtype=np.float64),
        "bonafide_prob": np.array(bonafide, dtype=np.float64),
        "spoof_prob": np.array(spoof, dtype=np.float64),
        "task_ids": np.array(task_ids, dtype=object),
    }
    print(f"[FusionGridsearch] Extracted {len(labels)} trials", flush=True)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        np.savez_compressed(cache_path, **out)
        print(f"[FusionGridsearch] Saved cache to {cache_path}", flush=True)

    return out


def evaluate_thresholds(
    scores: Dict[str, np.ndarray],
    *,
    cm_threshold: float,
    ecapa_threshold: float,
    q: float,
    asv_signal: str,
    metric: str,
) -> Dict[str, float]:
    labels = np.asarray(scores["labels"])
    yes_prob, gen_prob, asv_used = cascade_soft_scores(
        scores["ecapa_cos"],
        scores["bonafide_prob"],
        scores["spoof_prob"],
        cm_threshold=cm_threshold,
        ecapa_threshold=ecapa_threshold,
        q=q,
        w2v_cos=scores.get("w2v_cos"),
        asv_signal=asv_signal,
    )

    result: Dict[str, float] = {
        "cm_threshold": cm_threshold,
        "ecapa_threshold": ecapa_threshold,
        "q": q,
    }

    eer_sasv, eer_thr = compute_eer_sasv(labels, yes_prob)
    result["eer_sasv"] = eer_sasv
    result["eer_sasv_threshold"] = eer_thr

    sasv = compute_all_sasv_metrics(
        labels,
        yes_prob,
        gen_prob,
        asv_scores=asv_used,
        cm_scores=scores["bonafide_prob"],
    )
    result["t_eer"] = float(sasv.get("t_eer", float("nan")))
    result["t_eer_pct"] = float(sasv.get("t_eer_pct", result["t_eer"] * 100))
    result["min_a_dcf"] = float(sasv.get("min_a_dcf", float("nan")))
    result["min_t_dcf"] = float(sasv.get("min_t_dcf", float("nan")))

    preds = np.where(
        scores["bonafide_prob"] < cm_threshold,
        "gen",
        np.where(
            (asv_used * np.power(np.clip(scores["bonafide_prob"], 1e-8, 1), q)) >= ecapa_threshold,
            "yes",
            "no",
        ),
    )
    result["accuracy"] = float(np.mean(preds == labels))

    if metric == "eer_sasv":
        result["objective"] = eer_sasv
    elif metric == "t_eer":
        result["objective"] = result["t_eer"]
    elif metric == "min_a_dcf":
        result["objective"] = result["min_a_dcf"]
    elif metric == "accuracy":
        result["objective"] = -result["accuracy"]
    else:
        raise ValueError(f"Unknown metric {metric!r}")

    return result


def run_grid(
    scores: Dict[str, np.ndarray],
    *,
    cm_thresholds: List[float],
    ecapa_thresholds: List[float],
    q_values: List[float],
    asv_signal: str,
    metric: str,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    best: Optional[Dict[str, float]] = None

    for cm_t, ecapa_t, q in product(cm_thresholds, ecapa_thresholds, q_values):
        row = evaluate_thresholds(
            scores,
            cm_threshold=cm_t,
            ecapa_threshold=ecapa_t,
            q=q,
            asv_signal=asv_signal,
            metric=metric,
        )
        rows.append(row)
        if best is None or row["objective"] < best["objective"]:
            best = row

    assert best is not None
    return best, rows


def _resolve_dataset_path(config: Dict[str, Any], grid: Dict[str, Any]) -> str:
    explicit = str(grid.get("dataset_path", "") or "").strip()
    if explicit:
        return explicit
    split = str(grid.get("dataset", "val")).lower()
    data_cfg = config.get("Datasets", {})
    key = {
        "train": "dataset_train_path",
        "val": "dataset_val_path",
        "test": "dataset_test_path",
    }.get(split, "dataset_val_path")
    path = data_cfg.get(key, "")
    if not path:
        raise ValueError(f"No dataset path for split {split!r} (Datasets.{key})")
    return path


def _resolve_max_samples(config: Dict[str, Any], grid: Dict[str, Any]) -> Optional[int]:
    if grid.get("max_samples") is not None:
        return int(grid["max_samples"])
    split = str(grid.get("dataset", "val")).lower()
    data_cfg = config.get("Datasets", {})
    if split == "train":
        v = data_cfg.get("max_train_samples")
    elif split == "test":
        v = data_cfg.get("max_test_samples")
    else:
        v = data_cfg.get("max_valid_samples")
    return int(v) if v is not None else None


def run_threshold_gridsearch_from_config(
    config: Dict[str, Any],
    output_dir: str,
    device: torch.device,
    *,
    local_rank: int = 0,
) -> Dict[str, str]:
    """Run threshold grid search. Returns paths to artifacts (rank 0 only)."""
    grid = _grid_cfg(config)
    if local_rank != 0:
        return {}

    runner_cfg = config.get("Runner", {})
    dataset_path = _resolve_dataset_path(config, grid)
    max_samples = _resolve_max_samples(config, grid)

    batch_size = int(grid.get("batch_size", runner_cfg.get("SFT", {}).get("batch_size_eval", 4)))
    num_workers = int(grid.get("num_workers", runner_cfg.get("num_workers", 2)))
    grid_only = bool(grid.get("grid_only", False))
    force_recompute = bool(grid.get("force_recompute", False))

    asv_signal = str(grid.get("asv_signal", "ecapa"))
    metric = str(grid.get("metric", "eer_sasv"))

    apply_from = str(grid.get("apply_from", "") or "").strip()
    if apply_from:
        with open(apply_from, encoding="utf-8") as f:
            prev = json.load(f)
        best_prev = prev.get("best", prev)
        cm_grid = [float(best_prev["cm_threshold"])]
        ecapa_grid = [float(best_prev["ecapa_threshold"])]
        q_grid = [float(best_prev.get("q", 0.5))]
        if "asv_signal" in prev:
            asv_signal = str(prev["asv_signal"])
        print(
            f"[FusionGridsearch] apply_from={apply_from} -> "
            f"cm={cm_grid[0]}, ecapa={ecapa_grid[0]}, q={q_grid[0]}",
            flush=True,
        )
    else:
        cm_grid = parse_float_list(
            grid.get("cm_thresholds"), [0.3, 0.4, 0.5, 0.6, 0.7]
        )
        ecapa_grid = parse_float_list(
            grid.get("ecapa_thresholds"), [0.25, 0.35, 0.45, 0.55, 0.65]
        )
        q_grid = parse_float_list(grid.get("q_values"), [0.5])

    cache_name = str(grid.get("cache_features", "") or "").strip()
    if cache_name:
        cache_path = (
            cache_name
            if os.path.isabs(cache_name)
            else os.path.join(output_dir, cache_name)
        )
    else:
        cache_path = os.path.join(
            output_dir,
            f"scores_{grid.get('dataset', 'val')}_{max_samples or 'all'}.npz",
        )

    if grid_only and not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"FusionGridsearch.grid_only=true but cache not found: {cache_path}"
        )

    extract_device = torch.device("cpu") if grid_only else device
    scores = extract_scores(
        config,
        dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        device=extract_device,
        cache_path=cache_path,
        force_recompute=force_recompute and not grid_only,
    )

    n_combo = len(cm_grid) * len(ecapa_grid) * len(q_grid)
    print(
        f"[FusionGridsearch] Grid {len(cm_grid)}×{len(ecapa_grid)}×{len(q_grid)} = "
        f"{n_combo}; metric={metric}",
        flush=True,
    )

    best, rows = run_grid(
        scores,
        cm_thresholds=cm_grid,
        ecapa_thresholds=ecapa_grid,
        q_values=q_grid,
        asv_signal=asv_signal,
        metric=metric,
    )

    csv_path = os.path.join(output_dir, "grid_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_path = os.path.join(output_dir, "best_thresholds.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": metric,
                "dataset": dataset_path,
                "n_trials": int(len(scores["labels"])),
                "asv_signal": asv_signal,
                "best": best,
            },
            f,
            indent=2,
        )

    metrics_path = os.path.join(output_dir, "fusion_gridsearch_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\n=== Best thresholds ===", flush=True)
    for k in (
        "cm_threshold",
        "ecapa_threshold",
        "q",
        "objective",
        "eer_sasv",
        "t_eer_pct",
        "min_a_dcf",
        "accuracy",
    ):
        if k in best:
            print(f"  {k}: {best[k]}", flush=True)
    print(f"[FusionGridsearch] grid_results.csv -> {csv_path}", flush=True)
    print(f"[FusionGridsearch] best_thresholds.json -> {best_path}", flush=True)

    return {
        "csv": csv_path,
        "best_json": best_path,
        "metrics_json": metrics_path,
        "cache": cache_path,
    }
