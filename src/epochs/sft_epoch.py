from typing import Any, Optional, Tuple
import torch
import numpy as np
from .base import TrainEpoch
from .utils.batch_utils import split_batch
from ..dataloaders.samplers import normalize_iters_per_epoch
from tqdm.auto import tqdm

class SFTTrainEpoch(TrainEpoch):
    def __init__(self, *args, iters_per_epoch: Optional[int] = None, accum_grad_iters: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.iters_per_epoch = normalize_iters_per_epoch(iters_per_epoch)
        self.accum_grad_iters = accum_grad_iters

    def _get_autocast_dtype(self) -> torch.dtype:
        if self.scaler is None and self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _process_batch(
        self, batch: Any, backward: bool
    ) -> Tuple[
        torch.Tensor,
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """Run SFT forward for one batch; optionally run backward.

        Returns (loss, loss_val, ce1_loss_val, ce2_loss_val, arcface_loss_val).
        """
        autocast_dtype = self._get_autocast_dtype()
        with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
            # outputs = self.model.forward(batch)
            outputs = self.model(batch)
            loss = outputs["loss"] / self.accum_grad_iters
        loss_val = loss.item() * self.accum_grad_iters

        ce1_val = outputs.get("ce1_loss")
        if ce1_val is None:
            ce1_val = outputs.get("ce_loss")
        ce1_val = ce1_val.item() if ce1_val is not None else None

        ce2_val = outputs.get("ce2_loss")
        ce2_val = ce2_val.item() if ce2_val is not None else None

        arcface_val = outputs.get("arcface_loss")
        arcface_val = arcface_val.item() if arcface_val is not None else None

        if np.isnan(loss_val) or np.isinf(loss_val):
            return loss, None, None, None, None
        if backward:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss, loss_val, ce1_val, ce2_val, arcface_val

    def _handle_out_of_memory(
        self, batch: Any, batch_size: int, iteration: int
    ) -> Optional[float]:
        """On OOM: split batch and retry. Returns loss_val for logging or None to skip batch."""
        print(
            f"[OOM] CUDA out of memory at iteration {iteration} (batch_size={batch_size}). "
            "Splitting batch and retrying.",
            flush=True,
        )
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        autocast_dtype = self._get_autocast_dtype()
        n_splits = min(2, batch_size)
        sub_size = (batch_size + n_splits - 1) // n_splits
        loss_val = None
        for start in range(0, batch_size, sub_size):
            end = min(start + sub_size, batch_size)
            if start >= end:
                continue
            sub_batch = split_batch(batch, start, end)
            try:
                _, lv, _, _, _ = self._process_batch(sub_batch, backward=True)
                if lv is not None:
                    loss_val = lv
            except torch.cuda.OutOfMemoryError:
                print(
                    f"[OOM] Sub-batch ({start}:{end}) still OOM; skipping batch.",
                    flush=True,
                )
                torch.cuda.empty_cache()
                loss_val = None
        if loss_val is None:
            self.optimizer.zero_grad()
        return loss_val

    def run(self, epoch_num: int):
        self.logger.set_epoch(epoch_num, "train")
        self.model.train()

        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch_num)

        self.optimizer.zero_grad()
        optimizer_step_idx = 0

        for i, batch in tqdm(enumerate(self.dataloader), total = self.iters_per_epoch, desc="Train"):
            if self.iters_per_epoch is not None and i >= self.iters_per_epoch:
                break

            batch = self._move_to_device(batch, self.device)
            prompts = batch.get("prompts")
            audio_ids = batch.get("audio_ids")
            batch_size = len(prompts) if prompts else (len(audio_ids) if audio_ids else 1)

            loss_val = None
            ce1_val = None
            ce2_val = None
            arcface_val = None
            try:
                _, loss_val, ce1_val, ce2_val, arcface_val = self._process_batch(batch, backward=True)
                if loss_val is None:
                    print(f"Warning: NaN/Inf loss detected at iteration {i}, skipping backward!", flush=True)
                    self.optimizer.zero_grad()
                    self.logger.log(loss=0.0, lr=self.optimizer.param_groups[0]['lr'])
                    continue
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                loss_val = self._handle_out_of_memory(batch, batch_size, i)
                if loss_val is None:
                    continue

            if (i + 1) % self.accum_grad_iters == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                optimizer_step_idx += 1
                if self.scheduler:
                    if hasattr(self.scheduler, "step") and (self.scheduler.__class__.__name__ == "LinearWarmupCosineLRScheduler"):
                        self.scheduler.step(epoch_num, optimizer_step_idx)
                    else:
                        self.scheduler.step()

            if loss_val is not None:
                self.logger.log(
                    loss=loss_val,
                    lr=self.optimizer.param_groups[0]['lr'],
                    ce1_loss=ce1_val,
                    ce2_loss=ce2_val,
                    arcface_loss=arcface_val,
                )

        self.logger.log_epoch()

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch





