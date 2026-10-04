"""
SASV metrics per ASVspoof5 Track 2 (evaluation-package / ASVSpoof5.pdf):

- min a-DCF: architecture-agnostic DCF on SASV scores
- min t-DCF: ASV-constrained tandem DCF on CM scores
- t-EER: concurrent tandem EER from ASV (cosine_sim) + CM (bonafide_prob)
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

# ASVspoof 5 Track 2 priors/costs
PSPOOF = 0.05
PTAR = (1.0 - PSPOOF) * 0.99  # 0.9405
PNON = (1.0 - PSPOOF) * 0.01  # 0.0095
CMISS = 1.0
CFA = 10.0
CFA_SPOOF = 10.0
ALPHA = CMISS * PTAR / (CFA * PNON + CFA_SPOOF * PSPOOF)
GAMMA = CFA_SPOOF * PSPOOF / (CFA * PNON + CFA_SPOOF * PSPOOF)
PFA_NON_ASV_ORG = 0.01881016557566423
PMISS_ASV_ORG = 0.01880141010575793
PFA_SPF_ASV_ORG = 0.4607082907604729


def _labels_to_arrays(
    labels: np.ndarray,
    sasv_scores: np.ndarray,
    asv_scores: Optional[np.ndarray] = None,
    cm_scores: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels)
    sasv_scores = np.asarray(sasv_scores, dtype=np.float64)
    out = {
        "tar_sasv": sasv_scores[labels == "yes"],
        "non_sasv": sasv_scores[labels == "no"],
        "spoof_sasv": sasv_scores[labels == "gen"],
    }
    if asv_scores is not None:
        asv_scores = np.asarray(asv_scores, dtype=np.float64)
        out["tar_asv"] = asv_scores[labels == "yes"]
        out["non_asv"] = asv_scores[labels == "no"]
        out["spoof_asv"] = asv_scores[labels == "gen"]
    if cm_scores is not None:
        cm_scores = np.asarray(cm_scores, dtype=np.float64)
        bon_mask = (labels == "yes") | (labels == "no")
        out["bona_cm"] = cm_scores[bon_mask]
        out["spoof_cm"] = cm_scores[labels == "gen"]
    return out


def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DET curve (target=1, nontarget=0). Returns (frr, far, thresholds)."""
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    if target_scores.size == 0 or nontarget_scores.size == 0:
        return np.array([0.0]), np.array([1.0]), np.array([0.0])

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0.0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1.0), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def compute_Pmiss_Pfa_Pspoof_curves(
    tar_scores: np.ndarray,
    non_scores: np.ndarray,
    spf_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tar_scores = np.asarray(tar_scores, dtype=np.float64)
    non_scores = np.asarray(non_scores, dtype=np.float64)
    spf_scores = np.asarray(spf_scores, dtype=np.float64)

    all_scores = np.concatenate((tar_scores, non_scores, spf_scores))
    labels = np.concatenate(
        (
            np.ones(tar_scores.size),
            np.zeros(non_scores.size),
            -np.ones(spf_scores.size),
        )
    )
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_sums = np.cumsum(labels == 1)
    non_sums = np.cumsum(labels == 0)
    spoof_sums = np.cumsum(labels == -1)

    p_miss = np.concatenate((np.atleast_1d(0.0), tar_sums / max(tar_scores.size, 1)))
    p_fa_non = np.concatenate(
        (np.atleast_1d(1.0), 1.0 - (non_sums / max(non_scores.size, 1)))
    )
    p_fa_spoof = np.concatenate(
        (np.atleast_1d(1.0), 1.0 - (spoof_sums / max(spf_scores.size, 1)))
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return p_miss, p_fa_non, p_fa_spoof, thresholds


def compute_a_det_curve(
    trg_scores: np.ndarray,
    nontrg_scores: np.ndarray,
    spf_scores: np.ndarray,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    a-DCF DET (ASVspoof5 a_dcf.py): sweep SASV threshold, track
    FAR on non-target, FAR on spoof, FRR on target.
    """
    trg_scores = np.asarray(trg_scores, dtype=np.float64)
    nontrg_scores = np.asarray(nontrg_scores, dtype=np.float64)
    spf_scores = np.asarray(spf_scores, dtype=np.float64)

    if trg_scores.size == 0:
        return [1.0], [1.0], [0.0], [0.0]

    all_scores = np.concatenate((trg_scores, nontrg_scores, spf_scores))
    labels = np.concatenate(
        (
            np.ones_like(trg_scores),
            np.zeros_like(nontrg_scores),
            np.ones_like(spf_scores) + 1,
        )
    )
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    scores_sorted = all_scores[indices]

    fp_nontrg, fp_spf, fn = len(nontrg_scores), len(spf_scores), 0
    far_asvs, far_cms, frrs, thresh = [1.0], [1.0], [0.0], [float(scores_sorted.min()) - 1e-8]

    n_non = max(len(nontrg_scores), 1)
    n_spf = max(len(spf_scores), 1)
    n_trg = max(len(trg_scores), 1)

    for sco, lab in zip(scores_sorted, labels):
        if lab == 0:
            fp_nontrg -= 1
        elif lab == 1:
            fn += 1
        elif lab == 2:
            fp_spf -= 1
        far_asvs.append(fp_nontrg / n_non)
        far_cms.append(fp_spf / n_spf)
        frrs.append(fn / n_trg)
        thresh.append(float(sco))

    return far_asvs, far_cms, frrs, thresh


def compute_min_a_dcf(
    labels,
    sasv_scores,
    *,
    ptrg: float = PTAR,
    pnon: float = PNON,
    pspf: float = PSPOOF,
    cmiss: float = CMISS,
    cfa: float = CFA,
    cfa_spoof: float = CFA_SPOOF,
) -> Dict[str, float]:
    """
    Minimum architecture-agnostic DCF on SASV scores (higher = more target-like).
    Uses official a-DCF DET + ASVspoof5 cost model normalization.
    """
    arrs = _labels_to_arrays(np.asarray(labels), sasv_scores)
    tar, non, spf = arrs["tar_sasv"], arrs["non_sasv"], arrs["spoof_sasv"]

    if tar.size == 0:
        return {"min_a_dcf": 1.0, "min_a_dcf_thr": 0.0}

    far_asvs, far_cms, frrs, a_dcf_thresh = compute_a_det_curve(tar, non, spf)

    a_dcfs = (
        cmiss * ptrg * np.array(frrs)
        + cfa * pnon * np.array(far_asvs)
        + cfa_spoof * pspf * np.array(far_cms)
    )
    a_dcf_all_accept = cfa * pnon + cfa_spoof * pspf
    a_dcf_all_reject = cmiss * ptrg
    norm = min(a_dcf_all_accept, a_dcf_all_reject)
    a_dcfs_normed = a_dcfs / norm

    idx = int(np.argmin(a_dcfs_normed))
    return {
        "min_a_dcf": float(a_dcfs_normed[idx]),
        "min_a_dcf_thr": float(a_dcf_thresh[idx]),
    }


def compute_tdcf(
    bona_cm: np.ndarray,
    spoof_cm: np.ndarray,
    pfa_non_asv: float,
    pmiss_asv: float,
    pfa_spoof_asv: float,
    *,
    ptar: float = PTAR,
    pnon: float = PNON,
    pspoof: float = PSPOOF,
    cmiss: float = CMISS,
    cfa: float = CFA,
    cfa_spoof: float = CFA_SPOOF,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized ASV-constrained t-DCF curve (compute_tDCF in calculate_modules.py)."""
    bona_cm = np.asarray(bona_cm, dtype=np.float64)
    spoof_cm = np.asarray(spoof_cm, dtype=np.float64)
    if bona_cm.size == 0 or spoof_cm.size == 0:
        return np.array([1.0]), np.array([0.0])

    pmiss_cm, pfa_cm, cm_thresholds = compute_det_curve(bona_cm, spoof_cm)

    c0 = ptar * cmiss * pmiss_asv + pnon * cfa * pfa_non_asv
    c1 = ptar * cmiss - (ptar * cmiss * pmiss_asv + pnon * cfa * pfa_non_asv)
    c2 = pspoof * cfa_spoof * pfa_spoof_asv

    tdcf = c0 + c1 * pmiss_cm + c2 * pfa_cm
    tdcf_default = c0 + min(c1, c2)
    if tdcf_default <= 0:
        return np.array([1.0]), cm_thresholds
    return tdcf / tdcf_default, cm_thresholds


def compute_min_t_dcf(
    labels,
    cm_scores,
    *,
    pfa_non_asv: float = PFA_NON_ASV_ORG,
    pmiss_asv: float = PMISS_ASV_ORG,
    pfa_spoof_asv: float = PFA_SPF_ASV_ORG,
) -> Dict[str, float]:
    """Minimum ASV-constrained t-DCF on CM scores (bonafide_prob)."""
    arrs = _labels_to_arrays(np.asarray(labels), np.zeros(len(labels)), cm_scores=cm_scores)
    bona, spf = arrs.get("bona_cm", np.array([])), arrs.get("spoof_cm", np.array([]))

    if bona.size == 0 or spf.size == 0:
        return {"min_t_dcf": 1.0, "min_t_dcf_thr": 0.0}

    tdcf_curve, cm_thresholds = compute_tdcf(
        bona, spf, pfa_non_asv, pmiss_asv, pfa_spoof_asv
    )
    idx = int(np.argmin(tdcf_curve))
    return {
        "min_t_dcf": float(tdcf_curve[idx]),
        "min_t_dcf_thr": float(cm_thresholds[idx]),
    }


def compute_teer(
    pmiss_cm: np.ndarray,
    pfa_cm: np.ndarray,
    tau_cm: np.ndarray,
    pmiss_asv: np.ndarray,
    pfa_non_asv: np.ndarray,
    pfa_spf_asv: np.ndarray,
    tau_asv: np.ndarray,
) -> Tuple[float, float]:
    """
    Concurrent t-EER× (ASVspoof5 calculate_modules.compute_teer).

    Paper: at tandem thresholds τ× = (τ_asv, τ_cm) the tandem miss rate equals
    the tandem false-alarm rate (operating point on the t-EER path).

    Returns:
        (t_eer_pct, t_eer_xpoint_pct):
        - t_eer_pct: min over ASV thresholds of mean(P_miss^tdm, P_fa^tdm) at the
          CM threshold where they cross — interpretable concurrent EER (%).
        - t_eer_xpoint_pct: legacy ASVspoof5 ``Pfa_spf_ASV * Pfa_CM`` at the
          intersection point (%); often ~0 for strong CM and misleading alone.
    """
    rho_spf = 0.5
    min_teER = np.inf
    xpoint_crit_best = np.inf
    xpoint_teer = 0.0

    for tau_asv_idx, _ in enumerate(tau_asv):
        pmiss_tdm = pmiss_cm + (1.0 - pmiss_cm) * pmiss_asv[tau_asv_idx]
        pfa_tdm = (1.0 - rho_spf) * (1.0 - pmiss_cm) * pfa_non_asv[tau_asv_idx] + (
            rho_spf * pfa_cm * pfa_spf_asv[tau_asv_idx]
        )

        tmp = int(np.argmin(np.abs(pmiss_tdm - pfa_tdm)))

        cond = pmiss_asv[tau_asv_idx] < (
            (1.0 - rho_spf) * pfa_non_asv[tau_asv_idx] + rho_spf * pfa_spf_asv[tau_asv_idx]
        )
        if cond:
            teer_val = float(np.mean([pfa_tdm[tmp], pmiss_tdm[tmp]]))
            if teer_val < min_teER:
                min_teER = teer_val

        lhs = pfa_non_asv[tau_asv_idx] / max(pfa_spf_asv[tau_asv_idx], 1e-12)
        rhs = pfa_cm[tmp] / max(1.0 - pmiss_cm[tmp], 1e-12)
        crit = abs(lhs - rhs)
        if crit < xpoint_crit_best:
            xpoint_crit_best = crit
            xpoint_teer = float(pfa_spf_asv[tau_asv_idx] * pfa_cm[tmp])

    if min_teER < np.inf:
        teer_pct = float(min_teER * 100.0)
    elif xpoint_crit_best < np.inf:
        teer_pct = float(xpoint_teer * 100.0)
    else:
        teer_pct = 100.0

    xpoint_pct = float(xpoint_teer * 100.0) if xpoint_crit_best < np.inf else teer_pct
    return teer_pct, xpoint_pct


def compute_teer_accelerated(
    pmiss_cm: np.ndarray,
    pfa_cm: np.ndarray,
    tau_cm: np.ndarray,
    pmiss_asv: np.ndarray,
    pfa_non_asv: np.ndarray,
    pfa_spf_asv: np.ndarray,
    tau_asv: np.ndarray,
    size_decimated: int = 3000,
    bin_width: int = 1600,
) -> Tuple[float, float]:
    """Coarse-to-fine t-EER× (official accelerated path)."""
    ds_asv = max(tau_asv.shape[0] // size_decimated, 0)
    ds_cm = max(tau_cm.shape[0] // size_decimated, 0)

    approx_index: List[int] = []
    if ds_asv > 0 and ds_cm > 0:
        tmp_asv_idx = np.arange(tau_asv.shape[0])[::ds_asv]
        tmp_cm_idx = np.arange(tau_cm.shape[0])[::ds_cm]
        (_, _), approx_index = _compute_teer_with_index(
            pmiss_cm[tmp_cm_idx],
            pfa_cm[tmp_cm_idx],
            tau_cm[tmp_cm_idx],
            pmiss_asv[tmp_asv_idx],
            pfa_non_asv[tmp_asv_idx],
            pfa_spf_asv[tmp_asv_idx],
            tau_asv[tmp_asv_idx],
        )

    if approx_index:
        cen_asv = approx_index[0] * ds_asv
        cen_cm = approx_index[1] * ds_cm
        asv_1, asv_2 = max(cen_asv - bin_width, 0), min(cen_asv + bin_width, len(tau_asv))
        cm_1, cm_2 = max(cen_cm - bin_width, 0), min(cen_cm + bin_width, len(tau_cm))
        return compute_teer(
            pmiss_cm[cm_1:cm_2],
            pfa_cm[cm_1:cm_2],
            tau_cm[cm_1:cm_2],
            pmiss_asv[asv_1:asv_2],
            pfa_non_asv[asv_1:asv_2],
            pfa_spf_asv[asv_1:asv_2],
            tau_asv[asv_1:asv_2],
        )

    return compute_teer(
        pmiss_cm, pfa_cm, tau_cm, pmiss_asv, pfa_non_asv, pfa_spf_asv, tau_asv
    )


def _compute_teer_with_index(
    pmiss_cm, pfa_cm, tau_cm, pmiss_asv, pfa_non_asv, pfa_spf_asv, tau_asv
) -> Tuple[Tuple[float, float], List[int]]:
    teer_pct, xpoint_pct = compute_teer(
        pmiss_cm, pfa_cm, tau_cm, pmiss_asv, pfa_non_asv, pfa_spf_asv, tau_asv
    )
    rho_spf = 0.5
    xpoint_crit_best = np.inf
    xpoint_index: List[int] = []
    for tau_asv_idx, _ in enumerate(tau_asv):
        pmiss_tdm = pmiss_cm + (1.0 - pmiss_cm) * pmiss_asv[tau_asv_idx]
        pfa_tdm = (1.0 - rho_spf) * (1.0 - pmiss_cm) * pfa_non_asv[tau_asv_idx] + (
            rho_spf * pfa_cm * pfa_spf_asv[tau_asv_idx]
        )
        tmp = int(np.argmin(np.abs(pmiss_tdm - pfa_tdm)))
        lhs = pfa_non_asv[tau_asv_idx] / max(pfa_spf_asv[tau_asv_idx], 1e-12)
        rhs = pfa_cm[tmp] / max(1.0 - pmiss_cm[tmp], 1e-12)
        crit = abs(lhs - rhs)
        if crit < xpoint_crit_best:
            xpoint_crit_best = crit
            xpoint_index = [tau_asv_idx, tmp]
    return (teer_pct, xpoint_pct), xpoint_index


def compute_t_eer_x(
    labels,
    asv_scores,
    cm_scores,
) -> Dict[str, float]:
    """t-EER× from tandem ASV + CM subsystem scores."""
    arrs = _labels_to_arrays(
        np.asarray(labels), np.zeros(len(labels)), asv_scores=asv_scores, cm_scores=cm_scores
    )
    tar_asv, non_asv, spf_asv = arrs["tar_asv"], arrs["non_asv"], arrs["spoof_asv"]
    bona_cm, spoof_cm = arrs.get("bona_cm", np.array([])), arrs.get("spoof_cm", np.array([]))

    if (
        tar_asv.size == 0
        or non_asv.size == 0
        or spf_asv.size == 0
        or bona_cm.size == 0
        or spoof_cm.size == 0
    ):
        return {"t_eer": 1.0}

    pmiss_asv, pfa_non_asv, pfa_spf_asv, tau_asv = compute_Pmiss_Pfa_Pspoof_curves(
        tar_asv, non_asv, spf_asv
    )
    pmiss_cm, pfa_cm, tau_cm = compute_det_curve(bona_cm, spoof_cm)

    teer_pct, xpoint_pct = compute_teer_accelerated(
        pmiss_cm, pfa_cm, tau_cm, pmiss_asv, pfa_non_asv, pfa_spf_asv, tau_asv
    )
    return {
        "t_eer": teer_pct / 100.0,
        "t_eer_pct": teer_pct,
        "t_eer_xpoint_pct": xpoint_pct,
    }


def compute_sv_spf_eer(labels, yes_probs, gen_probs) -> Dict[str, float]:
    """Auxiliary SV-EER and SPF-EER (not primary ASVspoof5 metrics)."""
    labels = np.asarray(labels)
    yes_probs = np.asarray(yes_probs, dtype=np.float64)
    gen_probs = np.asarray(gen_probs, dtype=np.float64)
    results: Dict[str, float] = {}

    sv_mask = (labels == "yes") | (labels == "no")
    if sv_mask.sum() >= 2:
        sv_tar = labels[sv_mask] == "yes"
        if sv_tar.sum() > 0 and (~sv_tar).sum() > 0:
            frr, far, thr = compute_det_curve(
                yes_probs[sv_mask][sv_tar], yes_probs[sv_mask][~sv_tar]
            )
            idx = int(np.argmin(np.abs(frr - far)))
            results["sv_eer"] = float(np.mean((frr[idx], far[idx])))
            results["sv_eer_thr"] = float(thr[idx])

    bf_mask = (labels == "yes") | (labels == "no")
    sp_mask = labels == "gen"
    if bf_mask.sum() > 0 and sp_mask.sum() > 0:
        spf_scores = 1.0 - gen_probs
        frr, far, thr = compute_det_curve(spf_scores[bf_mask], spf_scores[sp_mask])
        idx = int(np.argmin(np.abs(frr - far)))
        results["spf_eer"] = float(np.mean((frr[idx], far[idx])))
        results["spf_eer_thr"] = float(thr[idx])

    return results


def _save_sasv_plots(
    labels: np.ndarray,
    yes_probs: np.ndarray,
    gen_probs: np.ndarray,
    asv_scores: Optional[np.ndarray],
    cm_scores: Optional[np.ndarray],
    plot_dir: str,
    plot_prefix: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: failed to import matplotlib, skip SASV plots: {exc}", flush=True)
        return

    os.makedirs(plot_dir, exist_ok=True)
    labels = np.asarray(labels)
    yes_probs = np.asarray(yes_probs, dtype=np.float64)
    gen_probs = np.asarray(gen_probs, dtype=np.float64)

    # 1) Class-conditional SASV score histograms.
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, color in (("yes", "tab:green"), ("no", "tab:orange"), ("gen", "tab:red")):
        cls_scores = yes_probs[labels == cls]
        if cls_scores.size > 0:
            ax.hist(cls_scores, bins=40, alpha=0.45, label=f"{cls} ({cls_scores.size})", color=color)
    ax.set_title("SASV score distribution (P(yes))")
    ax.set_xlabel("P(yes)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{plot_prefix}_yes_score_hist.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, color in (("yes", "tab:blue"), ("no", "tab:purple"), ("gen", "tab:red")):
        cls_scores = gen_probs[labels == cls]
        if cls_scores.size > 0:
            ax.hist(cls_scores, bins=40, alpha=0.45, label=f"{cls} ({cls_scores.size})", color=color)
    ax.set_title("SASV score distribution (P(gen))")
    ax.set_xlabel("P(gen)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{plot_prefix}_gen_score_hist.png"), dpi=160)
    plt.close(fig)

    arrs = _labels_to_arrays(labels, yes_probs, asv_scores=asv_scores, cm_scores=cm_scores)
    tar, non, spf = arrs["tar_sasv"], arrs["non_sasv"], arrs["spoof_sasv"]

    # 2) a-DCF curve vs threshold.
    if tar.size > 0:
        far_asvs, far_cms, frrs, a_dcf_thresh = compute_a_det_curve(tar, non, spf)
        a_dcfs = (
            CMISS * PTAR * np.array(frrs)
            + CFA * PNON * np.array(far_asvs)
            + CFA_SPOOF * PSPOOF * np.array(far_cms)
        )
        norm = min(CFA * PNON + CFA_SPOOF * PSPOOF, CMISS * PTAR)
        a_dcfs_normed = a_dcfs / max(norm, 1e-12)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(a_dcf_thresh, a_dcfs_normed, color="tab:blue", lw=2, label="normalized a-DCF")
        idx = int(np.argmin(a_dcfs_normed))
        ax.scatter(
            [a_dcf_thresh[idx]],
            [a_dcfs_normed[idx]],
            color="tab:red",
            s=36,
            label=f"min={a_dcfs_normed[idx]:.4f}",
        )
        ax.set_title("a-DCF vs threshold")
        ax.set_xlabel("SASV threshold")
        ax.set_ylabel("normalized a-DCF")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{plot_prefix}_a_dcf_curve.png"), dpi=160)
        plt.close(fig)

    # 3) t-DCF curve vs CM threshold.
    bona, spf_cm = arrs.get("bona_cm", np.array([])), arrs.get("spoof_cm", np.array([]))
    if bona.size > 0 and spf_cm.size > 0:
        tdcf_curve, cm_thresholds = compute_tdcf(
            bona, spf_cm, PFA_NON_ASV_ORG, PMISS_ASV_ORG, PFA_SPF_ASV_ORG
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cm_thresholds, tdcf_curve, color="tab:green", lw=2, label="normalized t-DCF")
        idx = int(np.argmin(tdcf_curve))
        ax.scatter(
            [cm_thresholds[idx]],
            [tdcf_curve[idx]],
            color="tab:red",
            s=36,
            label=f"min={tdcf_curve[idx]:.4f}",
        )
        ax.set_title("t-DCF vs CM threshold")
        ax.set_xlabel("CM threshold")
        ax.set_ylabel("normalized t-DCF")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{plot_prefix}_t_dcf_curve.png"), dpi=160)
        plt.close(fig)

    print(f"SASV plots saved to {plot_dir}", flush=True)


def _threshold_candidates(values: np.ndarray, max_candidates: int = 400) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([0.0], dtype=np.float64)
    uniq = np.unique(values)
    if uniq.size <= max_candidates:
        return uniq
    return np.quantile(uniq, np.linspace(0.0, 1.0, max_candidates))


def _eer_from_scores(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[float, float]:
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    idx = int(np.nanargmin(np.abs(frr - far)))
    return float(np.mean((frr[idx], far[idx]))), float(thresholds[idx])


def _pipeline_stats(
    labels: np.ndarray,
    cos_scores: np.ndarray,
    spoof_probs: np.ndarray,
    tau_sv: float,
    tau_spf: float,
) -> Tuple[float, float, float, float]:
    labels = np.asarray(labels)
    cos_scores = np.asarray(cos_scores, dtype=np.float64)
    spoof_probs = np.asarray(spoof_probs, dtype=np.float64)
    pred_gen = spoof_probs > tau_spf
    pred_yes = (~pred_gen) & (cos_scores >= tau_sv)

    tar_mask = labels == "yes"
    non_mask = labels == "no"
    spoof_mask = labels == "gen"
    nontar_mask = non_mask | spoof_mask

    n_tar = max(float(tar_mask.sum()), 1.0)
    n_non = max(float(non_mask.sum()), 1.0)
    n_spoof = max(float(spoof_mask.sum()), 1.0)
    n_nontar = max(float(nontar_mask.sum()), 1.0)

    p_miss = float(np.sum(~pred_yes[tar_mask]) / n_tar) if tar_mask.any() else 1.0
    p_fa_non = float(np.sum(pred_yes[non_mask]) / n_non) if non_mask.any() else 0.0
    p_fa_spoof = float(np.sum(pred_yes[spoof_mask]) / n_spoof) if spoof_mask.any() else 0.0
    p_fa_total = float(np.sum(pred_yes[nontar_mask]) / n_nontar) if nontar_mask.any() else 0.0
    t_eer_like = 0.5 * (p_miss + p_fa_total)
    return p_miss, p_fa_non, p_fa_spoof, t_eer_like


def cosine_spoof_pipeline_class(
    cos_score: float,
    spoof_prob: float,
    tau_sv: float,
    tau_spf: float,
) -> str:
    """Three-class cascade: spoof gate first, speaker cosine second."""
    if spoof_prob > tau_spf:
        return "gen"
    if cos_score >= tau_sv:
        return "yes"
    return "no"


def arcface_pipeline_class(cos_score: float, gen_prob: float, tau_sv: float, tau_spf: float) -> str:
    """Backward-compatible alias for the incoming cosine decision backend."""
    return cosine_spoof_pipeline_class(cos_score, gen_prob, tau_sv, tau_spf)


def compute_cosine_sasv_metrics(
    labels,
    cos_scores,
    spoof_probs,
    tau_sv: Optional[float] = None,
    tau_spf: Optional[float] = None,
    threshold_mode: str = "fixed",
    threshold_objective: str = "min_a_dcf",
    p_tar: float = 0.05,
    p_non: float = 0.45,
    p_spoof: float = 0.50,
    c_miss: float = 1.0,
    c_fa_non: float = 10.0,
    c_fa_spoof: float = 10.0,
) -> Dict[str, float]:
    """Metrics for a cosine + spoof-probability SASV decision pipeline."""
    labels = np.asarray(labels)
    cos_scores = np.asarray(cos_scores, dtype=np.float64)
    spoof_probs = np.asarray(spoof_probs, dtype=np.float64)

    results: Dict[str, float] = {}
    sv_mask = (labels == "yes") | (labels == "no")
    if sv_mask.sum() >= 2:
        sv_tar = labels[sv_mask] == "yes"
        if sv_tar.sum() > 0 and (~sv_tar).sum() > 0:
            sv_eer, sv_thr = _eer_from_scores(cos_scores[sv_mask][sv_tar], cos_scores[sv_mask][~sv_tar])
            results["sv_eer"] = sv_eer
            results["sv_eer_thr"] = sv_thr

    bona_mask = (labels == "yes") | (labels == "no")
    spoof_mask = labels == "gen"
    if bona_mask.sum() > 0 and spoof_mask.sum() > 0:
        spf_scores = 1.0 - spoof_probs
        spf_eer, spf_thr = _eer_from_scores(spf_scores[bona_mask], spf_scores[spoof_mask])
        results["spf_eer"] = spf_eer
        results["spf_eer_thr"] = spf_thr

    tau_sv_sel = float(tau_sv) if tau_sv is not None else (float(np.median(cos_scores)) if cos_scores.size else 0.0)
    tau_spf_sel = float(tau_spf) if tau_spf is not None else (float(np.median(spoof_probs)) if spoof_probs.size else 0.5)

    if threshold_mode in {"auto", "tune"} and labels.size > 0:
        best_obj = float("inf")
        for sv_t in _threshold_candidates(cos_scores):
            for spf_t in _threshold_candidates(spoof_probs):
                p_miss, p_fa_non, p_fa_spoof, t_eer_like = _pipeline_stats(
                    labels, cos_scores, spoof_probs, float(sv_t), float(spf_t)
                )
                dcf = c_miss * p_tar * p_miss + c_fa_non * p_non * p_fa_non + c_fa_spoof * p_spoof * p_fa_spoof
                obj = t_eer_like if threshold_objective == "t_eer" else dcf
                if obj < best_obj:
                    best_obj = obj
                    tau_sv_sel = float(sv_t)
                    tau_spf_sel = float(spf_t)

    p_miss, p_fa_non, p_fa_spoof, t_eer_like = _pipeline_stats(
        labels, cos_scores, spoof_probs, tau_sv_sel, tau_spf_sel
    )
    min_a_dcf = c_miss * p_tar * p_miss + c_fa_non * p_non * p_fa_non + c_fa_spoof * p_spoof * p_fa_spoof
    results.update({
        "tau_sv_selected": float(tau_sv_sel),
        "tau_spf_selected": float(tau_spf_sel),
        "pipeline_p_miss": float(p_miss),
        "pipeline_p_fa_non": float(p_fa_non),
        "pipeline_p_fa_spoof": float(p_fa_spoof),
        "t_eer": float(t_eer_like),
        "min_a_dcf": float(min_a_dcf),
    })
    return results



def _spoof_prob_from_prediction(p: Dict) -> float:
    if "gen_prob" in p:
        return float(p["gen_prob"])
    if "prob_spoof" in p:
        return float(p["prob_spoof"])
    return 0.0


def arrays_from_predictions(
    predictions: List[Dict],
    *,
    proxy_subsystem_scores: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Build SASV metric arrays from saved prediction JSONL rows.

    Mirrors TestEpoch accumulation for labels / yes_prob / gen_prob. When
    ``proxy_subsystem_scores`` is True, also derives subsystem proxies from LLM
    class probabilities:

    - asv_scores: yes_prob / (yes_prob + no_prob)
    - cm_scores: gen_prob (prob_spoof)
    """
    try:
        from ...analysis.plotter_common import extract_answer_from_gt
    except ImportError:
        from analysis.plotter_common import extract_answer_from_gt

    labels: List[str] = []
    yes_probs: List[float] = []
    gen_probs: List[float] = []
    asv_scores: List[float] = []
    cm_scores: List[float] = []

    for p in predictions:
        g = extract_answer_from_gt(p.get("gt", "")).lower()
        if g not in ("yes", "no", "gen"):
            continue

        yes = float(p.get("yes_prob", 0.0))
        no = float(p.get("no_prob", 0.0))
        gen = _spoof_prob_from_prediction(p)

        labels.append(g)
        yes_probs.append(yes)
        gen_probs.append(gen)

        if proxy_subsystem_scores:
            denom = yes + no
            asv_scores.append(yes / denom if denom > eps else 0.5)
            cm_scores.append(1.0 - gen)

    labels_arr = np.asarray(labels)
    yes_arr = np.asarray(yes_probs, dtype=np.float64)
    gen_arr = np.asarray(gen_probs, dtype=np.float64)
    if not proxy_subsystem_scores:
        return labels_arr, yes_arr, gen_arr, None, None
    return (
        labels_arr,
        yes_arr,
        gen_arr,
        np.asarray(asv_scores, dtype=np.float64),
        np.asarray(cm_scores, dtype=np.float64),
    )


def resolve_run_plots_dir(log_dir: str, subdir: str = "recomputed") -> str:
    """Resolve <run_dir>/plots/<subdir> from a test/validation log directory."""
    log_dir = os.path.abspath(log_dir)
    log_base = os.path.basename(log_dir)
    if log_base.startswith("test_") or log_base.startswith("validation_"):
        run_dir = os.path.dirname(os.path.dirname(log_dir))
    else:
        run_dir = os.path.dirname(log_dir)
    return os.path.join(run_dir, "plots", subdir)


def recompute_sasv_metrics_from_predictions(
    predictions: List[Dict],
    *,
    plot_dir: Optional[str] = None,
    plot_prefix: str = "sasv_from_jsonl",
    proxy_subsystem_scores: bool = True,
) -> Dict[str, float]:
    """Recompute ASVspoof5 SASV metrics from saved per-sample predictions."""
    labels, yes_probs, gen_probs, asv_scores, cm_scores = arrays_from_predictions(
        predictions,
        proxy_subsystem_scores=proxy_subsystem_scores,
    )
    if labels.size == 0:
        raise ValueError("No valid SASV predictions (expected gt in yes/no/gen)")

    kwargs: Dict[str, object] = {}
    if proxy_subsystem_scores and asv_scores is not None and cm_scores is not None:
        kwargs["asv_scores"] = asv_scores
        kwargs["cm_scores"] = cm_scores

    return compute_all_sasv_metrics(
        labels,
        yes_probs,
        gen_probs,
        plot_dir=plot_dir,
        plot_prefix=plot_prefix,
        **kwargs,
    )


def compute_all_sasv_metrics(
    labels,
    yes_probs,
    gen_probs,
    asv_scores: Optional[np.ndarray] = None,
    cm_scores: Optional[np.ndarray] = None,
    plot_dir: Optional[str] = None,
    plot_prefix: str = "sasv",
    **kwargs,
) -> Dict[str, float]:
    """
    ASVspoof5 Track 2 metrics for validation logging.

    Args:
        labels: "yes" / "no" / "gen"
        yes_probs: SASV P(yes) — used for min a-DCF
        gen_probs: SASV P(gen)
        asv_scores: optional cosine similarity (ASV subsystem); enables t-EER×
        cm_scores: optional P(bonafide) from CE2 head; enables min t-DCF and t-EER×
    """
    labels = np.asarray(labels)
    yes_probs = np.asarray(yes_probs, dtype=np.float64)
    gen_probs = np.asarray(gen_probs, dtype=np.float64)

    results: Dict[str, float] = {}
    results.update(compute_min_a_dcf(labels, yes_probs))

    if cm_scores is not None:
        results.update(compute_min_t_dcf(labels, cm_scores))

    if asv_scores is not None and cm_scores is not None:
        results.update(compute_t_eer_x(labels, asv_scores, cm_scores))

    results.update(compute_sv_spf_eer(labels, yes_probs, gen_probs))

    if plot_dir:
        _save_sasv_plots(
            labels=labels,
            yes_probs=yes_probs,
            gen_probs=np.asarray(gen_probs, dtype=np.float64),
            asv_scores=asv_scores,
            cm_scores=cm_scores,
            plot_dir=plot_dir,
            plot_prefix=plot_prefix,
        )
    return results


def print_sasv_metrics_summary(m: Dict[str, float]) -> None:
    """Stdout report for SASVEvalEpoch."""
    print("\n" + "=" * 50, flush=True)
    print("ASVspoof5 SASV Metrics:", flush=True)
    if "min_a_dcf" in m:
        print(
            f"  min a-DCF: {m['min_a_dcf']:.6f}  (thr={m.get('min_a_dcf_thr', 0):.4f})",
            flush=True,
        )
    if "min_t_dcf" in m:
        print(
            f"  min t-DCF: {m['min_t_dcf']:.6f}  (thr={m.get('min_t_dcf_thr', 0):.4f})",
            flush=True,
        )
    if "t_eer" in m:
        pct = m.get("t_eer_pct", m["t_eer"] * 100.0)
        print(f"  t-EER×:    {pct:.4f}%", flush=True)
        if "t_eer_xpoint_pct" in m and abs(m["t_eer_xpoint_pct"] - pct) > 0.01:
            print(
                f"             (xpoint legacy: {m['t_eer_xpoint_pct']:.4f}%)",
                flush=True,
            )
    if "sv_eer" in m:
        print(f"  SV-EER:    {m['sv_eer']:.4f}  (aux)", flush=True)
    if "spf_eer" in m:
        print(f"  SPF-EER:   {m['spf_eer']:.4f}  (aux)", flush=True)
    print("=" * 50 + "\n", flush=True)
