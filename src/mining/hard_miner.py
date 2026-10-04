"""
Offline hard pair / trial mining for SASV training.

Scores every sample in the training set with the current model, then selects
the hardest examples from three categories:

1. **Hard rejected** — non-target pairs with high ASV cosine similarity.
2. **Hard spoof**    — spoof trials with high bonafide probability (bypass CM).
3. **Borderline verified** — target pairs with low ASV cosine similarity.

Usage:
    miner = HardPairMiner(cfg)
    scores = miner.score_dataset(model, dataloader, device)
    hard_indices = miner.mine(scores)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class HardMiningConfig:
    enabled: bool = True
    top_k_rejected: float = 0.30       # keep top 30 % hardest rejected
    top_k_spoof: float = 0.30          # keep top 30 % hardest spoof
    top_k_verified: float = 0.30       # keep bottom 30 % verified (lowest sim)
    random_fraction: float = 0.30      # fraction of *easy* samples to keep
    scoring_batch_size: int = 8        # batch size for the offline scoring pass
    log_scores: bool = True            # print score distributions
    mine_every_n_epochs: int = 1       # mine every N epochs (1 = every epoch)
    max_scoring_samples: Optional[int] = None
    scoring_samples_offset: int = 0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "HardMiningConfig":
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Per-sample score record
# ---------------------------------------------------------------------------

@dataclass
class SampleScore:
    index: int                  # global dataset index
    gt_label: str               # "verified" / "rejected" / "spoof"
    cosine_sim: float = 0.0     # cosine(enroll_embed, query_embed)
    bonafide_prob: float = 0.0  # P(bonafide) from bonafide_spoof_head on query
    ce1_loss: float = 0.0       # per-sample token-prediction loss


# ---------------------------------------------------------------------------
# Miner
# ---------------------------------------------------------------------------

class HardPairMiner:
    """Offline hard pair miner for SASV training."""

    def __init__(self, cfg: HardMiningConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_dataset(
        self,
        model: torch.nn.Module,
        dataloader: Any,
        device: torch.device,
        amp: bool = True,
    ) -> List[SampleScore]:
        """Run the model over *dataloader* and return per-sample scores.

        The model must expose a ``score_pairs(samples)`` method (see
        ``SASVSalmonModel.score_pairs``).
        """
        model.eval()

        autocast_dtype = torch.bfloat16
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            autocast_dtype = torch.float16

        unwrapped = model.module if hasattr(model, "module") else model

        _is_main = not dist.is_initialized() or dist.get_rank() == 0

        total_samples = len(dataloader.dataset)

        scores: List[SampleScore] = []
        global_idx = 0
        counts = {"verified": 0, "rejected": 0, "spoof": 0, "other": 0}
        t_start = time.time()

        # Only show progress bar on main rank
        pbar = None
        if _is_main:
            pbar = tqdm(total=total_samples, desc="[HardMining] scoring", unit="samples")

        for batch_idx, batch in enumerate(dataloader):
            batch = _move_to_device(batch, device)
            batch_size = len(batch.get("audio_ids", [None]))

            try:
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda", dtype=autocast_dtype):
                    out = unwrapped.score_pairs(batch)
            except torch.cuda.OutOfMemoryError:
                if _is_main:
                    print(
                        f"[HardMining] OOM at batch {batch_idx} (bs={batch_size}), skipping",
                        flush=True,
                    )
                torch.cuda.empty_cache()
                global_idx += batch_size
                continue

            cosine_sims = out.get("cosine_sim")          # (B,) or None
            bonafide_probs = out.get("bonafide_prob")     # (B,) or None
            ce1_losses = out.get("ce1_loss_per_sample")   # (B,) or None

            gt_labels = batch.get("gt", batch.get("answer", [""] * batch_size))

            for i in range(batch_size):
                gt = _normalise_gt(str(gt_labels[i]) if i < len(gt_labels) else "")
                cs = cosine_sims[i].item() if cosine_sims is not None else 0.0
                bp = bonafide_probs[i].item() if bonafide_probs is not None else 0.0
                cl = ce1_losses[i].item() if ce1_losses is not None else 0.0

                scores.append(SampleScore(
                    index=global_idx + i,
                    gt_label=gt,
                    cosine_sim=cs,
                    bonafide_prob=bp,
                    ce1_loss=cl,
                ))
                counts[gt if gt in counts else "other"] += 1

            global_idx += batch_size

            # Update progress bar on main rank
            if pbar is not None:
                pbar.update(batch_size)
                pbar.set_postfix(
                    V=counts["verified"], R=counts["rejected"], S=counts["spoof"],
                    refresh=True
                )

            # Free GPU memory eagerly
            _free_batch(batch)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if pbar is not None:
            pbar.close()

        if _is_main:
            elapsed = time.time() - t_start
            print(
                f"[HardMining] scoring done: {global_idx} samples in {elapsed:.1f}s "
                f"({global_idx / max(elapsed, 1e-6):.0f} samples/s)",
                flush=True,
            )

        # Distributed: gather across ranks
        if dist.is_initialized():
            dist.barrier()
            all_scores_lists = [None] * dist.get_world_size()
            dist.all_gather_object(all_scores_lists, scores)
            # Deduplicate by index (each rank may see overlapping samples)
            merged: Dict[int, SampleScore] = {}
            for rank_scores in all_scores_lists:
                if rank_scores is None:
                    continue
                for s in rank_scores:
                    merged[s.index] = s
            scores = [merged[k] for k in sorted(merged.keys())]

        model.train()
        return scores

    # ------------------------------------------------------------------
    # Mining (index selection)
    # ------------------------------------------------------------------

    def mine(self, scores: List[SampleScore]) -> List[int]:
        """Select hard sample indices from *scores*.

        Returns a sorted list of unique dataset indices.
        """
        _is_main = not dist.is_initialized() or dist.get_rank() == 0

        rejected = [s for s in scores if s.gt_label == "rejected"]
        spoof = [s for s in scores if s.gt_label == "spoof"]
        verified = [s for s in scores if s.gt_label == "verified"]
        other = [s for s in scores if s.gt_label not in ("rejected", "spoof", "verified")]

        if _is_main:
            print(
                f"[HardMining] category split: "
                f"verified={len(verified)} rejected={len(rejected)} "
                f"spoof={len(spoof)} other={len(other)}",
                flush=True,
            )

        hard_indices: List[int] = []
        easy_indices: List[int] = []

        # 1. Hard rejected — highest cosine similarity
        hard_r, easy_r, thresh_r = self._topk(
            rejected,
            key=lambda s: s.cosine_sim,
            k_frac=self.cfg.top_k_rejected,
            reverse=True,  # descending → highest first
            score_attr="cosine_sim",
        )
        hard_indices.extend(hard_r)
        easy_indices.extend(easy_r)
        if _is_main and rejected:
            print(
                f"[HardMining] rejected: {len(rejected)} total -> "
                f"{len(hard_r)} hard (top {self.cfg.top_k_rejected*100:.0f}%, "
                f"cosine_sim >= {thresh_r:.4f})",
                flush=True,
            )

        # 2. Hard spoof — highest bonafide probability
        hard_s, easy_s, thresh_s = self._topk(
            spoof,
            key=lambda s: s.bonafide_prob,
            k_frac=self.cfg.top_k_spoof,
            reverse=True,
            score_attr="bonafide_prob",
        )
        hard_indices.extend(hard_s)
        easy_indices.extend(easy_s)
        if _is_main and spoof:
            print(
                f"[HardMining] spoof:    {len(spoof)} total -> "
                f"{len(hard_s)} hard (top {self.cfg.top_k_spoof*100:.0f}%, "
                f"bonafide_prob >= {thresh_s:.4f})",
                flush=True,
            )

        # 3. Borderline verified — lowest cosine similarity
        hard_v, easy_v, thresh_v = self._topk(
            verified,
            key=lambda s: s.cosine_sim,
            k_frac=self.cfg.top_k_verified,
            reverse=False,  # ascending → lowest first
            score_attr="cosine_sim",
        )
        hard_indices.extend(hard_v)
        easy_indices.extend(easy_v)
        if _is_main and verified:
            print(
                f"[HardMining] verified: {len(verified)} total -> "
                f"{len(hard_v)} hard (bottom {self.cfg.top_k_verified*100:.0f}%, "
                f"cosine_sim <= {thresh_v:.4f})",
                flush=True,
            )

        # Add random fraction of easy samples to prevent catastrophic forgetting
        n_random = 0
        if self.cfg.random_fraction > 0 and easy_indices:
            import random
            n_random = max(1, int(len(easy_indices) * self.cfg.random_fraction))
            random.shuffle(easy_indices)
            hard_indices.extend(easy_indices[:n_random])

        selected = sorted(set(hard_indices))

        if _is_main:
            n_hard_only = len(selected) - n_random
            print(
                f"[HardMining] selection: {len(hard_r)}R + {len(hard_s)}S + {len(hard_v)}V hard "
                f"+ {n_random} random easy = {len(selected)} unique "
                f"({100 * len(selected) / max(len(scores), 1):.1f}% of {len(scores)})",
                flush=True,
            )

        if self.cfg.log_scores:
            self._log_stats(rejected, spoof, verified, selected, scores)

        return selected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topk(
        samples: List[SampleScore],
        key,
        k_frac: float,
        reverse: bool,
        score_attr: str = "cosine_sim",
    ) -> Tuple[List[int], List[int], float]:
        """Return (hard_indices, easy_indices, threshold_score) after top-K% selection."""
        if not samples:
            return [], [], 0.0
        sorted_samples = sorted(samples, key=key, reverse=reverse)
        k = max(1, int(math.ceil(len(sorted_samples) * k_frac)))
        hard = [s.index for s in sorted_samples[:k]]
        easy = [s.index for s in sorted_samples[k:]]
        threshold = getattr(sorted_samples[k - 1], score_attr, 0.0)
        return hard, easy, threshold

    @staticmethod
    def _log_stats(
        rejected: List[SampleScore],
        spoof: List[SampleScore],
        verified: List[SampleScore],
        selected: List[int],
        all_scores: List[SampleScore],
    ) -> None:
        def _summarise(label: str, items: List[SampleScore], score_key: str) -> str:
            if not items:
                return f"  {label}: 0 samples"
            vals = [getattr(s, score_key) for s in items]
            mn, mx, avg = min(vals), max(vals), sum(vals) / len(vals)
            return (
                f"  {label}: {len(items)} samples, "
                f"{score_key} min={mn:.4f} max={mx:.4f} avg={avg:.4f}"
            )

        print("[HardMining] Score statistics:", flush=True)
        print(_summarise("rejected", rejected, "cosine_sim"), flush=True)
        print(_summarise("spoof", spoof, "bonafide_prob"), flush=True)
        print(_summarise("verified", verified, "cosine_sim"), flush=True)
        print(
            f"[HardMining] Total scored: {len(all_scores)}, "
            f"selected: {len(selected)} "
            f"({100 * len(selected) / max(len(all_scores), 1):.1f}%)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalise_gt(gt: str) -> str:
    """Map answer tokens to canonical category names."""
    gt = gt.strip().lower()
    if gt in ("yes", "verified"):
        return "verified"
    if gt in ("no", "rejected"):
        return "rejected"
    if gt in ("gen", "spoof"):
        return "spoof"
    return gt


def _move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(v, device) for v in batch]
    return batch


def _free_batch(batch: Dict[str, Any]) -> None:
    heavy_keys = [
        "audio", "input_ids", "attention_mask", "pixel_values", "image",
        "input_features", "feature_attention_mask", "spectrogram", "raw_wav",
    ]
    for k in heavy_keys:
        if k in batch:
            del batch[k]
