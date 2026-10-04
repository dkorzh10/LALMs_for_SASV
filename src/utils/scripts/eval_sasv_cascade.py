#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download


def _bootstrap_paths() -> None:
    this_file = Path(__file__).resolve()
    repo_src = this_file.parents[2]
    cascade_root = Path(os.environ.get("SASV_CASCADE_ROOT", ""))
    for path in (repo_src, cascade_root):
        if not path:
            continue
        path_str = str(path)
        if path_str and path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_paths()

from dataset import AudioDataset  # noqa: E402
from sasv_cascade import ASVSystem, CMSystem, WavLM, WavLMConfig  # noqa: E402
from epochs.utils.sasv_metrics import compute_all_sasv_metrics  # noqa: E402


class ECAPA2ASVWrapper(torch.nn.Module):
    """Wrap pretrained ECAPA2 with ASVSystem-compatible score_pair API."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        model_file = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
        self.ecapa = torch.jit.load(model_file, map_location=device)
        self.ecapa.eval()

    def extract_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        emb = self.ecapa(wav)
        return F.normalize(emb, dim=-1)

    def score_pair(self, enroll_wav: torch.Tensor, test_wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        enr = self.extract_embedding(enroll_wav)
        tst = self.extract_embedding(test_wav)
        score = F.cosine_similarity(enr, tst, dim=-1)
        return {"score1": score, "score": score}


class CascadeEvaluator:
    """Run SASV_CASCADE pairwise inference and collect metrics."""

    def __init__(
        self,
        cm_model_path: str,
        asv_model_path: str,
        q: float,
        cm_threshold: float,
        fusion_threshold: float,
        device: str,
        asv_type: str,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.q = float(q)
        self.cm_threshold = float(cm_threshold)
        self.fusion_threshold = float(fusion_threshold)
        self.asv_type = asv_type

        self.cm_model = self._load_cm_model(cm_model_path).to(self.device).eval()
        if asv_type == "ecapa2":
            self.asv_model = ECAPA2ASVWrapper(device=str(self.device)).to(self.device).eval()
        else:
            self.asv_model = self._load_asv_model(asv_model_path).to(self.device).eval()

    @staticmethod
    def _load_wavlm() -> WavLM:
        wavlm_checkpoint = torch.load("WavLM-Base.pt", map_location="cpu")
        cfg = WavLMConfig(wavlm_checkpoint["cfg"])
        wavlm = WavLM(cfg)
        wavlm.load_state_dict(wavlm_checkpoint["model"])
        return wavlm

    def _load_cm_model(self, cm_model_path: str) -> CMSystem:
        checkpoint = torch.load(cm_model_path, map_location="cpu")
        wavlm = self._load_wavlm()
        model = CMSystem(
            wavlm=wavlm,
            hidden_size=768,
            num_classes=2,
            dropout=0.2,
            freeze_feature_extractor=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @staticmethod
    def _load_asv_model(asv_model_path: str) -> ASVSystem:
        checkpoint = torch.load(asv_model_path, map_location="cpu")
        model = ASVSystem(emb_dim=256, use_campp=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @staticmethod
    def compute_power_fusion(asv_score: torch.Tensor, cm_score: torch.Tensor, q: float) -> torch.Tensor:
        cm_safe = torch.clamp(cm_score, min=1e-8)
        return asv_score * torch.pow(cm_safe, q)

    @torch.no_grad()
    def predict_single_with_scores(self, enroll_audio: torch.Tensor, query_audio: torch.Tensor) -> Dict[str, float]:
        if enroll_audio.dim() == 1:
            enroll_audio = enroll_audio.unsqueeze(0)
        if query_audio.dim() == 1:
            query_audio = query_audio.unsqueeze(0)

        enroll_audio = enroll_audio.to(self.device, dtype=torch.float32)
        query_audio = query_audio.to(self.device, dtype=torch.float32)

        cm_out = self.cm_model(query_audio)
        asv_out = self.asv_model.score_pair(enroll_audio, query_audio)

        cm_bonafide_score = cm_out["probs"][:, 0]
        cm_spoof_score = cm_out["probs"][:, 1]
        asv_score = asv_out["score"]
        fused_score = self.compute_power_fusion(asv_score, cm_bonafide_score, self.q)

        is_spoof = cm_bonafide_score < self.cm_threshold
        is_verified = fused_score >= self.fusion_threshold
        pred_idx = torch.where(is_spoof, 2, torch.where(is_verified, 0, 1))

        return {
            "prediction_idx": int(pred_idx.item()),
            "asv_score": float(asv_score.item()),
            "cm_score": float(cm_bonafide_score.item()),
            "fused_score": float(fused_score.item()),
            "yes_prob": float(torch.sigmoid(fused_score).item()),
            "gen_prob": float(cm_spoof_score.item()),
        }

    def evaluate_dataset(self, dataset_path: str, config_path: str, max_samples: int) -> Dict[str, float]:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        audio_cfg = config["Audio"]
        dataset = AudioDataset(
            data_path=dataset_path,
            max_samples=max_samples,
            target_sample_rate=audio_cfg["target_sr"],
            task_type="hard_label",
            samples_offset=0,
            audio_cfg=audio_cfg,
            is_train=False,
            split="test",
        )

        gt_map = {"yes": 0, "no": 1, "gen": 2}
        class_name_by_idx = {0: "yes", 1: "no", 2: "gen"}

        total = 0
        correct = 0
        class_total = {0: 0, 1: 0, 2: 0}
        class_correct = {0: 0, 1: 0, 2: 0}

        labels: List[str] = []
        yes_probs: List[float] = []
        gen_probs: List[float] = []
        asv_scores: List[float] = []
        cm_scores: List[float] = []

        for i in range(len(dataset)):
            try:
                item = dataset[i]
                ref_audios = item.get("reference_audios", [])
                qry_audios = item.get("query_audios", [])
                if not ref_audios or not qry_audios:
                    continue

                gt_label = str(item.get("gt", "")).lower()
                if gt_label not in gt_map:
                    continue

                gt_idx = gt_map[gt_label]
                pred = self.predict_single_with_scores(ref_audios[0].squeeze(), qry_audios[0].squeeze())
                pred_idx = int(pred["prediction_idx"])

                total += 1
                class_total[gt_idx] += 1
                if pred_idx == gt_idx:
                    correct += 1
                    class_correct[gt_idx] += 1

                labels.append(gt_label)
                yes_probs.append(pred["yes_prob"])
                gen_probs.append(pred["gen_prob"])
                asv_scores.append(pred["asv_score"])
                cm_scores.append(pred["cm_score"])

            except Exception as exc:
                print(f"Warning: failed on sample {i}: {exc}", flush=True)
                continue

        accuracy = (correct / total) if total > 0 else 0.0
        class_acc = {
            class_name_by_idx[idx]: (
                class_correct[idx] / class_total[idx] if class_total[idx] > 0 else 0.0
            )
            for idx in (0, 1, 2)
        }
        accuracy_balanced = float(np.mean([class_acc["yes"], class_acc["no"], class_acc["gen"]]))

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy),
            "accuracy_balanced": float(accuracy_balanced),
            "accuracy_yes": float(class_acc["yes"]),
            "accuracy_no": float(class_acc["no"]),
            "accuracy_gen": float(class_acc["gen"]),
        }

        if labels:
            sasv_metrics = compute_all_sasv_metrics(
                np.array(labels),
                np.array(yes_probs),
                np.array(gen_probs),
                asv_scores=np.array(asv_scores),
                cm_scores=np.array(cm_scores),
            )
            for key in ("t_eer", "min_a_dcf", "min_t_dcf"):
                if key in sasv_metrics:
                    metrics[key] = float(sasv_metrics[key])

        return metrics


def _outputs_root_dir(default_root: str) -> str:
    root = os.path.abspath(os.path.expanduser(default_root))
    if os.path.basename(root) == "outputs":
        return root
    return os.path.dirname(root)


def _checkpoint_summary_identity(ckpt_path: str, fallback_experiment: str, fallback_run: str) -> Dict[str, str]:
    best_ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))
    parts = os.path.normpath(best_ckpt_path).split(os.sep)
    for i, part in enumerate(parts):
        if part == "outputs" and i + 2 < len(parts):
            maybe_run = parts[i + 2]
            if maybe_run.startswith("run_"):
                return {
                    "experiment": parts[i + 1],
                    "run": maybe_run,
                    "best_ckpt_path": best_ckpt_path,
                }

    return {
        "experiment": fallback_experiment,
        "run": fallback_run,
        "best_ckpt_path": best_ckpt_path,
    }


def _csv_metric_value(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def upsert_test_summary_csv(
    outputs_root: str,
    row: Dict[str, Any],
) -> str:
    os.makedirs(outputs_root, exist_ok=True)
    csv_path = os.path.join(outputs_root, "checkpoint_test_summary.csv")
    id_fields = ["experiment", "run", "best_ckpt_path", "meta_path"]
    metric_fields = [
        "accuracy",
        "accuracy_balanced",
        "accuracy_yes",
        "accuracy_no",
        "accuracy_gen",
        "t_eer",
        "min_a_dcf",
        "min_t_dcf",
    ]
    fieldnames = id_fields + metric_fields

    rows: List[Dict[str, Any]] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    key = (row["experiment"], row["run"], row["best_ckpt_path"])
    updated = False
    for idx, existing in enumerate(rows):
        existing_key = (
            existing.get("experiment", ""),
            existing.get("run", ""),
            existing.get("best_ckpt_path", ""),
        )
        if existing_key == key:
            rows[idx] = {**existing, **row}
            updated = True
            break
    if not updated:
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SASV_CASCADE with unified SASV metrics and save to checkpoint_test_summary.csv."
    )
    parser.add_argument("--cm_model", type=str, required=True, help="Path to CM checkpoint (.pt)")
    parser.add_argument("--asv_model", type=str, default="", help="Path to ASV checkpoint (.pt)")
    parser.add_argument("--asv_type", type=str, default="ecapa2", choices=["custom", "ecapa2"])
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config with Audio section")
    parser.add_argument("--dataset", type=str, required=True, help="Path to test dataset JSON/JSONL")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--q", type=float, default=0.5, help="Power exponent in fused score")
    parser.add_argument("--cm_threshold", type=float, default=0.5, help="CM spoof threshold")
    parser.add_argument("--fusion_threshold", type=float, default=0.3, help="Fusion accept threshold")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "./outputs"),
        help="Root outputs dir that contains checkpoint_test_summary.csv",
    )
    parser.add_argument("--experiment", type=str, default="sasv_cascade", help="Fallback experiment name")
    parser.add_argument("--run", type=str, default="manual_eval", help="Fallback run name")
    parser.add_argument(
        "--best_ckpt_path",
        type=str,
        default="",
        help="Optional path to identify run in checkpoint_test_summary.csv; defaults to --cm_model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_paths = [
        (args.cm_model, "CM checkpoint"),
        (args.config, "Config file"),
        (args.dataset, "Dataset file"),
    ]
    if args.asv_type == "custom":
        if not args.asv_model:
            raise ValueError("--asv_model is required when --asv_type=custom")
        check_paths.append((args.asv_model, "ASV checkpoint"))
    for path, name in check_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    evaluator = CascadeEvaluator(
        cm_model_path=args.cm_model,
        asv_model_path=args.asv_model,
        q=args.q,
        cm_threshold=args.cm_threshold,
        fusion_threshold=args.fusion_threshold,
        device=args.device,
        asv_type=args.asv_type,
    )
    max_samples = None if args.max_samples <= 0 else args.max_samples
    metrics = evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        config_path=args.config,
        max_samples=max_samples,
    )

    id_row = _checkpoint_summary_identity(
        ckpt_path=args.best_ckpt_path or args.cm_model,
        fallback_experiment=args.experiment,
        fallback_run=args.run,
    )
    row = {**id_row}
    row["meta_path"] = os.path.abspath(os.path.expanduser(args.dataset))
    for key, value in metrics.items():
        row[key] = _csv_metric_value(value)

    csv_path = upsert_test_summary_csv(_outputs_root_dir(args.outputs_root), row)
    print("SASV_CASCADE metrics:", flush=True)
    for key in (
        "accuracy",
        "accuracy_balanced",
        "accuracy_yes",
        "accuracy_no",
        "accuracy_gen",
        "t_eer",
        "min_a_dcf",
        "min_t_dcf",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}", flush=True)
    print(f"Checkpoint test summary saved to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
