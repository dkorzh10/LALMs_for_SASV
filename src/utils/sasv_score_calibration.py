"""Calibrate ECAPA (ASV) and AASIST (CM) scores with sklearn CalibratedClassifierCV."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sasv_threshold_gridsearch import (
    _resolve_dataset_path,
    _resolve_max_samples,
    compute_eer_sasv,
    extract_scores,
    parse_float_list,
    run_grid,
)


@dataclass
class CalibratorMetrics:
    expert: str
    n_train: int
    pos_rate: float
    brier: float
    roc_auc: float
    method: str
    cv: int
    target: str


def _calibration_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    runner = config.get("Runner", {}) or {}
    return runner.get("ScoreCalibration") or runner.get("score_calibration") or {}



def _require_sklearn():
    try:
        import joblib
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sasv_score_calibration requires scikit-learn and joblib (see requirements.txt)."
        ) from exc
    return joblib, CalibratedClassifierCV, calibration_curve, LogisticRegression, brier_score_loss, roc_auc_score


def is_score_calibration_run_type(run_type: str) -> bool:
    return str(run_type).lower() in ("score_calibration", "calibrate_scores")


def _label_array(labels: np.ndarray) -> np.ndarray:
    return np.asarray([str(x).strip().lower() for x in labels], dtype=object)


def _mask_for_target(labels: np.ndarray, target: str, expert: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (row_mask, binary_y) for the requested calibration target."""
    labels = _label_array(labels)

    if expert == "ecapa":
        if target == "yes_vs_no":
            mask = np.isin(labels, ("yes", "no"))
            y = (labels[mask] == "yes").astype(np.int64)
        elif target == "yes_vs_rest":
            mask = np.isin(labels, ("yes", "no", "gen"))
            y = (labels[mask] == "yes").astype(np.int64)
        else:
            raise ValueError(f"Unknown ecapa calibration target {target!r}")
        return mask, y

    if expert == "aasist":
        if target == "bonafide_vs_spoof":
            mask = np.isin(labels, ("yes", "no", "gen"))
            y = np.isin(labels[mask], ("yes", "no")).astype(np.int64)
        elif target == "spoof_vs_bonafide":
            mask = np.isin(labels, ("yes", "no", "gen"))
            y = (labels[mask] == "gen").astype(np.int64)
        else:
            raise ValueError(f"Unknown aasist calibration target {target!r}")
        return mask, y

    raise ValueError(f"Unknown expert {expert!r}; use 'ecapa' or 'aasist'")


class BinaryScoreCalibrator:
    """Map a scalar score to calibrated P(positive) via CalibratedClassifierCV."""

    def __init__(
        self,
        *,
        expert: str,
        target: str,
        method: str = "sigmoid",
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.expert = expert
        self.target = target
        self.method = method
        self.cv = int(cv)
        self.random_state = int(random_state)
        self.calibrator = None  # sklearn CalibratedClassifierCV when fitted
        self.metrics: Optional[CalibratorMetrics] = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> CalibratorMetrics:
        (
            _joblib,
            CalibratedClassifierCV,
            _calibration_curve,
            LogisticRegression,
            brier_score_loss,
            roc_auc_score,
        ) = _require_sklearn()
        scores = np.asarray(scores, dtype=np.float64)
        mask, y = _mask_for_target(labels, self.target, self.expert)
        x = scores[mask].reshape(-1, 1)

        if x.shape[0] < max(2, self.cv):
            raise ValueError(
                f"{self.expert}: only {x.shape[0]} samples for target={self.target!r}; "
                f"need at least {max(2, self.cv)}"
            )
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"{self.expert}: target={self.target!r} has a single class after filtering"
            )

        base = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=self.random_state,
        )
        self.calibrator = CalibratedClassifierCV(
            base,
            method=self.method,
            cv=self.cv,
        )
        self.calibrator.fit(x, y)

        probs = self.predict_proba(scores[mask])
        brier = float(brier_score_loss(y, probs))
        roc = float(roc_auc_score(y, probs))

        self.metrics = CalibratorMetrics(
            expert=self.expert,
            n_train=int(x.shape[0]),
            pos_rate=float(y.mean()),
            brier=brier,
            roc_auc=roc,
            method=self.method,
            cv=self.cv,
            target=self.target,
        )
        return self.metrics

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("BinaryScoreCalibrator is not fitted")
        x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return self.calibrator.predict_proba(x)[:, 1]

    def calibration_curve(
        self, scores: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        (
            _joblib,
            _CalibratedClassifierCV,
            calibration_curve,
            _LogisticRegression,
            _brier_score_loss,
            _roc_auc_score,
        ) = _require_sklearn()
        mask, y = _mask_for_target(labels, self.target, self.expert)
        probs = self.predict_proba(scores[mask])
        return calibration_curve(y, probs, n_bins=n_bins, strategy="quantile")

    def save(self, path: str) -> None:
        joblib, *_rest = _require_sklearn()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "expert": self.expert,
            "target": self.target,
            "method": self.method,
            "cv": self.cv,
            "random_state": self.random_state,
            "calibrator": self.calibrator,
            "metrics": asdict(self.metrics) if self.metrics else None,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "BinaryScoreCalibrator":
        joblib, *_rest = _require_sklearn()
        payload = joblib.load(path)
        obj = cls(
            expert=str(payload["expert"]),
            target=str(payload["target"]),
            method=str(payload.get("method", "sigmoid")),
            cv=int(payload.get("cv", 5)),
            random_state=int(payload.get("random_state", 42)),
        )
        obj.calibrator = payload["calibrator"]
        metrics = payload.get("metrics")
        if metrics:
            obj.metrics = CalibratorMetrics(**metrics)
        return obj


class SASVExpertCalibrators:
    """ECAPA + AASIST calibrators for downstream SASV fusion."""

    def __init__(self, ecapa: BinaryScoreCalibrator, aasist: BinaryScoreCalibrator) -> None:
        self.ecapa = ecapa
        self.aasist = aasist

    @classmethod
    def from_config(cls, cal_cfg: Dict[str, Any]) -> "SASVExpertCalibrators":
        ecapa_cfg = cal_cfg.get("ecapa", {}) or {}
        aasist_cfg = cal_cfg.get("aasist", {}) or {}
        rs = int(cal_cfg.get("random_state", 42))
        return cls(
            ecapa=BinaryScoreCalibrator(
                expert="ecapa",
                target=str(ecapa_cfg.get("target", "yes_vs_no")),
                method=str(ecapa_cfg.get("method", cal_cfg.get("method", "sigmoid"))),
                cv=int(ecapa_cfg.get("cv", cal_cfg.get("cv", 5))),
                random_state=rs,
            ),
            aasist=BinaryScoreCalibrator(
                expert="aasist",
                target=str(aasist_cfg.get("target", "bonafide_vs_spoof")),
                method=str(aasist_cfg.get("method", cal_cfg.get("method", "sigmoid"))),
                cv=int(aasist_cfg.get("cv", cal_cfg.get("cv", 5))),
                random_state=rs,
            ),
        )

    def fit(self, scores: Dict[str, np.ndarray]) -> Dict[str, CalibratorMetrics]:
        labels = scores["labels"]
        ecapa_metrics = self.ecapa.fit(scores["ecapa_cos"], labels)
        aasist_metrics = self.aasist.fit(scores["bonafide_prob"], labels)
        return {"ecapa": ecapa_metrics, "aasist": aasist_metrics}

    def transform(self, scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in scores.items()}
        out["ecapa_cos_raw"] = np.asarray(scores["ecapa_cos"], dtype=np.float64)
        out["bonafide_prob_raw"] = np.asarray(scores["bonafide_prob"], dtype=np.float64)
        out["ecapa_cos"] = self.ecapa.predict_proba(scores["ecapa_cos"])
        out["bonafide_prob"] = self.aasist.predict_proba(scores["bonafide_prob"])
        spoof_raw = np.asarray(scores.get("spoof_prob", 1.0 - out["bonafide_prob_raw"]))
        out["spoof_prob"] = np.clip(1.0 - out["bonafide_prob"], 0.0, 1.0)
        out["spoof_prob_raw"] = spoof_raw
        return out

    def save(self, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        ecapa_path = os.path.join(output_dir, "calibrator_ecapa.joblib")
        aasist_path = os.path.join(output_dir, "calibrator_aasist.joblib")
        self.ecapa.save(ecapa_path)
        self.aasist.save(aasist_path)
        return {"ecapa": ecapa_path, "aasist": aasist_path}

    @classmethod
    def load(cls, output_dir: str) -> "SASVExpertCalibrators":
        return cls(
            ecapa=BinaryScoreCalibrator.load(os.path.join(output_dir, "calibrator_ecapa.joblib")),
            aasist=BinaryScoreCalibrator.load(os.path.join(output_dir, "calibrator_aasist.joblib")),
        )


def default_calibrator_output_dir(config: Dict[str, Any], output_dir: str) -> str:
    cal_cfg = _calibration_cfg(config)
    explicit = str(cal_cfg.get("output_dir", "") or "").strip()
    if explicit:
        return explicit
    name = str(cal_cfg.get("output_name", "score_calibrators"))
    return os.path.join(output_dir, name)


def _evaluate_on_split(
    calibrators: SASVExpertCalibrators,
    scores: Dict[str, np.ndarray],
    *,
    split_name: str,
) -> Dict[str, Any]:
    calibrated = calibrators.transform(scores)
    labels = np.asarray(calibrated["labels"])

    ecapa_mask, ecapa_y = _mask_for_target(labels, calibrators.ecapa.target, "ecapa")
    aasist_mask, aasist_y = _mask_for_target(labels, calibrators.aasist.target, "aasist")

    ecapa_probs = calibrated["ecapa_cos"][ecapa_mask]
    aasist_probs = calibrated["bonafide_prob"][aasist_mask]

    result: Dict[str, Any] = {
        "split": split_name,
        "n_trials": int(len(labels)),
        "ecapa_brier": float(brier_score_loss(ecapa_y, ecapa_probs)),
        "ecapa_roc_auc": float(roc_auc_score(ecapa_y, ecapa_probs)),
        "aasist_brier": float(brier_score_loss(aasist_y, aasist_probs)),
        "aasist_roc_auc": float(roc_auc_score(aasist_y, aasist_probs)),
    }

    yes_prob_raw, _, _ = _yes_prob_from_scores(scores)
    yes_prob_cal, _, _ = _yes_prob_from_scores(calibrated)
    eer_raw, thr_raw = compute_eer_sasv(labels, yes_prob_raw)
    eer_cal, thr_cal = compute_eer_sasv(labels, yes_prob_cal)
    result["eer_sasv_raw"] = eer_raw
    result["eer_sasv_raw_threshold"] = thr_raw
    result["eer_sasv_calibrated"] = eer_cal
    result["eer_sasv_calibrated_threshold"] = thr_cal
    return result


def _yes_prob_from_scores(scores: Dict[str, np.ndarray], *, q: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .sasv_threshold_gridsearch import cascade_soft_scores

    yes_prob, gen_prob, asv_used = cascade_soft_scores(
        scores["ecapa_cos"],
        scores["bonafide_prob"],
        scores["spoof_prob"],
        cm_threshold=0.0,
        ecapa_threshold=0.0,
        q=q,
        w2v_cos=scores.get("w2v_cos"),
        asv_signal="ecapa",
    )
    return yes_prob, gen_prob, asv_used


def run_score_calibration_from_config(
    config: Dict[str, Any],
    output_dir: str,
    device,
    *,
    local_rank: int = 0,
) -> Dict[str, str]:
    """Fit ECAPA/AASIST calibrators and optionally tune fusion thresholds on val."""
    if local_rank != 0:
        return {}

    import torch

    cal_cfg = _calibration_cfg(config)
    runner_cfg = config.get("Runner", {})

    train_split = str(cal_cfg.get("train_split", "train")).lower()
    eval_split = str(cal_cfg.get("eval_split", "val")).lower()

    train_path = _resolve_dataset_path(
        config,
        {
            "dataset": train_split,
            "dataset_path": cal_cfg.get("train_dataset_path", ""),
        },
    )
    eval_path = _resolve_dataset_path(
        config,
        {
            "dataset": eval_split,
            "dataset_path": cal_cfg.get("eval_dataset_path", ""),
        },
    )

    train_max = cal_cfg.get("max_train_samples")
    if train_max is None:
        train_max = _resolve_max_samples(
            config, {"dataset": train_split, "max_samples": None}
        )
    eval_max = cal_cfg.get("max_eval_samples")
    if eval_max is None:
        eval_max = _resolve_max_samples(
            config, {"dataset": eval_split, "max_samples": None}
        )

    batch_size = int(cal_cfg.get("batch_size", runner_cfg.get("SFT", {}).get("batch_size_eval", 4)))
    num_workers = int(cal_cfg.get("num_workers", runner_cfg.get("num_workers", 2)))
    force_recompute = bool(cal_cfg.get("force_recompute", False))
    scores_only = bool(cal_cfg.get("scores_only", False))

    cache_train = str(cal_cfg.get("cache_train", "") or "").strip()
    cache_eval = str(cal_cfg.get("cache_eval", "") or "").strip()
    if cache_train and not os.path.isabs(cache_train):
        cache_train = os.path.join(output_dir, cache_train)
    if cache_eval and not os.path.isabs(cache_eval):
        cache_eval = os.path.join(output_dir, cache_eval)
    if not cache_train:
        cache_train = os.path.join(output_dir, f"scores_{train_split}_{train_max or 'all'}.npz")
    if not cache_eval:
        cache_eval = os.path.join(output_dir, f"scores_{eval_split}_{eval_max or 'all'}.npz")

    extract_device = torch.device("cpu") if scores_only else device

    print(f"[ScoreCalibration] Extracting train scores: {train_path}", flush=True)
    train_scores = extract_scores(
        config,
        train_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=int(train_max) if train_max is not None else None,
        device=extract_device,
        cache_path=cache_train,
        force_recompute=force_recompute,
    )

    print(f"[ScoreCalibration] Extracting eval scores: {eval_path}", flush=True)
    eval_scores = extract_scores(
        config,
        eval_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=int(eval_max) if eval_max is not None else None,
        device=extract_device,
        cache_path=cache_eval,
        force_recompute=force_recompute,
    )

    calibrators = SASVExpertCalibrators.from_config(cal_cfg)
    fit_metrics = calibrators.fit(train_scores)

    cal_dir = default_calibrator_output_dir(config, output_dir)
    cal_paths = calibrators.save(cal_dir)

    eval_raw = _evaluate_on_split(calibrators, eval_scores, split_name=eval_split)
    eval_calibrated_scores = calibrators.transform(eval_scores)

    artifacts: Dict[str, str] = {
        "calibrator_dir": cal_dir,
        "calibrator_ecapa": cal_paths["ecapa"],
        "calibrator_aasist": cal_paths["aasist"],
        "cache_train": cache_train,
        "cache_eval": cache_eval,
    }

    metrics_doc: Dict[str, Any] = {
        "train_dataset": train_path,
        "eval_dataset": eval_path,
        "fit_metrics": {k: asdict(v) for k, v in fit_metrics.items()},
        "eval_metrics": eval_raw,
        "calibrator_paths": cal_paths,
    }

    if bool(cal_cfg.get("grid_search", True)):
        asv_signal = str(cal_cfg.get("asv_signal", "ecapa"))
        metric = str(cal_cfg.get("metric", "eer_sasv"))
        cm_grid = parse_float_list(
            cal_cfg.get("cm_thresholds"), [0.3, 0.4, 0.5, 0.6, 0.7]
        )
        ecapa_grid = parse_float_list(
            cal_cfg.get("ecapa_thresholds"), [0.25, 0.35, 0.45, 0.55, 0.65]
        )
        q_grid = parse_float_list(cal_cfg.get("q_values"), [0.5])

        best_raw, _ = run_grid(
            eval_scores,
            cm_thresholds=cm_grid,
            ecapa_thresholds=ecapa_grid,
            q_values=q_grid,
            asv_signal=asv_signal,
            metric=metric,
        )
        best_cal, rows_cal = run_grid(
            eval_calibrated_scores,
            cm_thresholds=cm_grid,
            ecapa_thresholds=ecapa_grid,
            q_values=q_grid,
            asv_signal=asv_signal,
            metric=metric,
        )
        metrics_doc["fusion_grid_raw"] = best_raw
        metrics_doc["fusion_grid_calibrated"] = best_cal

        grid_csv = os.path.join(output_dir, "fusion_grid_calibrated.csv")
        import csv

        with open(grid_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_cal[0].keys()))
            writer.writeheader()
            writer.writerows(rows_cal)
        artifacts["fusion_grid_csv"] = grid_csv

        best_path = os.path.join(output_dir, "best_thresholds_calibrated.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metric": metric,
                    "asv_signal": asv_signal,
                    "eval_dataset": eval_path,
                    "best_raw": best_raw,
                    "best_calibrated": best_cal,
                },
                f,
                indent=2,
            )
        artifacts["best_thresholds"] = best_path

    metrics_path = os.path.join(output_dir, "score_calibration_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_doc, f, indent=2)
    artifacts["metrics_json"] = metrics_path

    print("\n=== Score calibration ===", flush=True)
    for expert, m in fit_metrics.items():
        print(
            f"  {expert}: n={m.n_train} pos_rate={m.pos_rate:.4f} "
            f"brier={m.brier:.4f} auc={m.roc_auc:.4f} ({m.method}, cv={m.cv})",
            flush=True,
        )
    print(
        f"  eval EER raw={eval_raw['eer_sasv_raw']:.4f} "
        f"calibrated={eval_raw['eer_sasv_calibrated']:.4f}",
        flush=True,
    )
    if "fusion_grid_calibrated" in metrics_doc:
        print(
            f"  fusion grid best raw objective={metrics_doc['fusion_grid_raw']['objective']:.4f}",
            flush=True,
        )
        print(
            f"  fusion grid best calibrated objective="
            f"{metrics_doc['fusion_grid_calibrated']['objective']:.4f}",
            flush=True,
        )
    print(f"[ScoreCalibration] Calibrators saved to {cal_dir}", flush=True)
    print(f"[ScoreCalibration] Metrics -> {metrics_path}", flush=True)

    return artifacts
