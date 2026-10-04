import numpy as np

from src.epochs.utils.sasv_metrics import (
    arcface_pipeline_class,
    compute_cosine_sasv_metrics,
    cosine_spoof_pipeline_class,
)


def test_cosine_spoof_pipeline_uses_spoof_gate_before_cosine():
    assert cosine_spoof_pipeline_class(0.95, 0.80, tau_sv=0.50, tau_spf=0.70) == "gen"
    assert cosine_spoof_pipeline_class(0.95, 0.20, tau_sv=0.50, tau_spf=0.70) == "yes"
    assert cosine_spoof_pipeline_class(0.20, 0.20, tau_sv=0.50, tau_spf=0.70) == "no"


def test_arcface_pipeline_class_remains_backward_compatible_alias():
    assert arcface_pipeline_class(0.95, 0.80, tau_sv=0.50, tau_spf=0.70) == "gen"
    assert arcface_pipeline_class(0.95, 0.20, tau_sv=0.50, tau_spf=0.70) == "yes"


def test_compute_cosine_metrics_accepts_backend_spoof_probabilities_fixed_thresholds():
    labels = np.array(["yes", "no", "gen", "gen"])
    cos_scores = np.array([0.80, 0.20, 0.90, 0.10])
    spoof_probs = np.array([0.10, 0.10, 0.90, 0.80])

    metrics = compute_cosine_sasv_metrics(
        labels,
        cos_scores,
        spoof_probs,
        tau_sv=0.50,
        tau_spf=0.50,
        threshold_mode="fixed",
    )

    assert metrics["tau_sv_selected"] == 0.50
    assert metrics["tau_spf_selected"] == 0.50
    assert metrics["t_eer"] == 0.0
    assert metrics["min_a_dcf"] == 0.0


def test_compute_cosine_metrics_auto_tunes_spoof_probability_threshold():
    labels = np.array(["yes", "no", "gen", "gen"])
    cos_scores = np.array([0.80, 0.20, 0.90, 0.10])
    ce2_spoof_probs = np.array([0.10, 0.10, 0.90, 0.80])

    metrics = compute_cosine_sasv_metrics(
        labels,
        cos_scores,
        ce2_spoof_probs,
        threshold_mode="auto",
        threshold_objective="min_a_dcf",
    )

    assert metrics["min_a_dcf"] == 0.0
    assert metrics["t_eer"] == 0.0
    assert metrics["tau_spf_selected"] in ce2_spoof_probs
