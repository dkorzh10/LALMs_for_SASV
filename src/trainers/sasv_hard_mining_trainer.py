"""SASV Trainer with offline hard-pair mining every N epochs.

Before training epochs (every N as configured) the current model scores the
full training set (no grad, eval mode) and selects the hardest samples from
3 categories:

1. Hard rejected  — non-target pairs with highest ASV cosine similarity
2. Hard spoof     — spoof trials with highest bonafide probability
3. Borderline verified — target pairs with lowest ASV cosine similarity

A random fraction of easy samples is mixed in to prevent catastrophic
forgetting.  The mined subset is wrapped in a ``Subset`` DataLoader and
passed to the regular ``SASVSFTTrainEpoch``.  On epochs without mining, the
previous mined subset is reused.

Config (``Runner.SFT.hard_mining``):
    enabled: true
    top_k_rejected: 0.30
    top_k_spoof: 0.30
    top_k_verified: 0.30
    random_fraction: 0.30
    scoring_batch_size: 8
    log_scores: true
    mine_every_n_epochs: 1    # set to N to mine every N epochs
"""

import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .base import Trainer
from ..epochs.sasv_sft_epoch import SASVSFTTrainEpoch
from ..epochs.sasv_eval_epoch import SASVEvalEpoch
from ..mining.hard_miner import HardPairMiner, HardMiningConfig
from ..dataloaders.builder import get_subset_dataloader, get_scoring_dataloader


class SASVHardMiningTrainer(Trainer):
    """SASV trainer that mines hard pairs before each epoch."""

    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        mining_cfg_dict = config.get("hard_mining", {})
        self.mining_cfg = HardMiningConfig.from_dict(mining_cfg_dict)
        self.mining_enabled = self.mining_cfg.enabled

        if self.mining_enabled:
            self.miner = HardPairMiner(self.mining_cfg)
        else:
            self.miner = None

        self._is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        self._is_main = not dist.is_initialized() or dist.get_rank() == 0

        # Cache for the mined loader to reuse between mining epochs
        self._cached_mined_loader = None

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(self):
        for epoch in range(self.num_epochs):
            if self._is_main:
                print(f"Starting SASV Hard-Mining Epoch {epoch}", flush=True)

            # --- Mine hard subset (every N epochs) ---
            if self.mining_enabled and self.miner is not None:
                # Check if we should mine this epoch
                mine_this_epoch = epoch % self.mining_cfg.mine_every_n_epochs == 0
                if mine_this_epoch or self._cached_mined_loader is None:
                    self._cached_mined_loader = self._mine_and_build_loader(epoch)
                else:
                    # Update the sampler epoch if using DistributedSampler
                    if hasattr(self._cached_mined_loader, "sampler") and \
                       hasattr(self._cached_mined_loader.sampler, "set_epoch"):
                        self._cached_mined_loader.sampler.set_epoch(epoch)
                    if self._is_main:
                        print(
                            f"[HardMining] Epoch {epoch}: reusing cached mined subset "
                            f"(next mining at epoch {epoch + self.mining_cfg.mine_every_n_epochs - (epoch % self.mining_cfg.mine_every_n_epochs)})",
                            flush=True,
                        )
                epoch_loader = self._cached_mined_loader
            else:
                epoch_loader = self.train_loader

            # --- Train ---
            train_epoch = SASVSFTTrainEpoch(
                self.model, epoch_loader, self.logger,
                self.optimizer, self.scheduler, self.scaler,
                device=self.device,
                amp=self.amp,
                iters_per_epoch=self.config.get("iters_per_epoch"),
                accum_grad_iters=self.config.get("accum_grad_iters", 1),
                max_grad_norm=self.max_grad_norm,
            )
            train_epoch.run(epoch_num=epoch)

            # --- Validate ---
            val_accuracy = 0.0
            if self.val_loader:
                val_accuracy = self.validate(epoch)

            self.save_checkpoint(epoch, val_accuracy, name="sasv_hm")

    # ------------------------------------------------------------------
    # Validation (same as SASVTrainer)
    # ------------------------------------------------------------------

    def validate(self, epoch_num: int) -> float:
        gen_cfg = self.config.get("generation", {})
        if not gen_cfg:
            gen_cfg = {"max_new_tokens": 1, "num_beams": 1, "do_sample": False}

        eval_epoch = SASVEvalEpoch(
            self.model, self.val_loader, self.logger,
            device=self.device, gen_cfg=gen_cfg,
        )
        eval_epoch.run(epoch_num=epoch_num)
        accuracy = getattr(self.logger, 'last_accuracy', 0.0)
        return accuracy

    # ------------------------------------------------------------------
    # Mining + subset loader
    # ------------------------------------------------------------------

    def _mine_and_build_loader(self, epoch: int):
        """Score a configured training window, mine hard indices, return a subset DataLoader."""
        t0 = time.time()
        scoring_indices = self._build_scoring_indices(epoch)

        scoring_loader = get_scoring_dataloader(
            self.train_loader,
            batch_size=self.mining_cfg.scoring_batch_size,
            indices=scoring_indices,
        )

        if self._is_main:
            total = len(scoring_loader.dataset)
            base_total = len(self.train_loader.dataset)
            window_msg = (
                f"window={total}/{base_total}"
                if scoring_indices is not None else f"full={base_total}"
            )
            print(
                f"[HardMining] Epoch {epoch}: scoring {window_msg} samples "
                f"(bs={self.mining_cfg.scoring_batch_size})...",
                flush=True,
            )

        scores = self.miner.score_dataset(
            self.model, scoring_loader, self.device,
            amp=self.amp,
        )
        if scoring_indices is not None:
            for score in scores:
                if 0 <= score.index < len(scoring_indices):
                    score.index = scoring_indices[score.index]

        hard_indices = self.miner.mine(scores)

        elapsed = time.time() - t0
        if self._is_main:
            print(
                f"[HardMining] Epoch {epoch}: mined {len(hard_indices)} samples "
                f"in {elapsed:.1f}s",
                flush=True,
            )

        # Build subset dataloader
        num_replicas = dist.get_world_size() if self._is_distributed else None
        rank = dist.get_rank() if self._is_distributed else None

        subset_loader = get_subset_dataloader(
            self.train_loader,
            hard_indices,
            shuffle=True,
            distributed=self._is_distributed,
            num_replicas=num_replicas,
            rank=rank,
        )

        # Set epoch on sampler if distributed
        if hasattr(subset_loader, "sampler") and hasattr(subset_loader.sampler, "set_epoch"):
            subset_loader.sampler.set_epoch(epoch)

        return subset_loader

    def _build_scoring_indices(self, epoch: int):
        max_samples = self.mining_cfg.max_scoring_samples
        if max_samples is None or max_samples <= 0:
            return None

        dataset_len = len(self.train_loader.dataset)
        if dataset_len == 0:
            return []

        max_samples = min(int(max_samples), dataset_len)
        base_offset = int(self.mining_cfg.scoring_samples_offset or 0)
        offset = (base_offset + epoch * max_samples) % dataset_len
        if offset + max_samples <= dataset_len:
            indices = list(range(offset, offset + max_samples))
        else:
            end_count = (offset + max_samples) - dataset_len
            indices = list(range(offset, dataset_len)) + list(range(0, end_count))

        if self._is_main:
            print(
                f"[HardMining] small scoring window: offset={offset}, "
                f"max_scoring_samples={max_samples}",
                flush=True,
            )
        return indices
