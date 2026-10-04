"""
SASV Distillation dataset forming: run inference, collect samples where model makes errors.
No reasoning, no judge, no skeptic — just error-based filtering.
"""
import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .base import Epoch
from .utils.text_utils import texts_for_log


class SASVDistillationDatasetFormingEpoch(Epoch):
    """
    Run inference on the dataset, compare predictions to ground truth (yes/no/gen),
    and collect indices where the model is wrong. Returns error indices for SFT.
    """

    def __init__(
        self,
        model,
        dataloader: DataLoader,
        logger,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        amp: bool = True,
    ):
        super().__init__(model, dataloader, logger, device=device)
        self.config = config
        self.amp = amp

        forming_cfg = config.get("Filtering", {})
        self.gen_cfg = forming_cfg.get(
            "generation",
            {"max_new_tokens": 256, "num_beams": 1, "do_sample": False},
        )
        self.save_intermediate_dataset = forming_cfg.get("save_intermediate_dataset", True)
        self.save_tmp_freq = forming_cfg.get("save_intermediate_dataset_tmp_freq", 10)
        self.min_intermediate_size = forming_cfg.get("min_intermediate_dataset_size", 50)

    def run(
        self,
        distillation_iter: int,
    ) -> List[int]:
        """
        Run inference on the full dataset, collect error indices.
        Returns list of global dataset indices where the model predicted incorrectly.
        """
        from ..analysis.plotter_common import extract_answer

        _is_main = not dist.is_initialized() or dist.get_rank() == 0

        self.logger.set_distillation_iter(distillation_iter, "dataset_forming")

        base_ds = self.dataloader.dataset
        while hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset
        total_size = len(base_ds)

        if _is_main:
            print(
                f"[SASVDistillation] Dataset forming iter {distillation_iter}: "
                f"device={self.device}, total_samples={total_size}",
                flush=True,
            )

        self.model.eval()

        error_indices: List[int] = []
        error_audio_ids: List[Any] = []
        error_gt_labels: List[str] = []
        error_pred_labels: List[str] = []
        samples_processed = 0

        use_amp = self.amp and self.device.type == "cuda"
        autocast_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                batch = self._move_to_device(batch, self.device)

                prompts = batch.get("prompts")
                batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))

                # Ground truth labels
                if batch.get("answer"):
                    gt_labels = batch["answer"]
                else:
                    gt_labels = batch.get("text", [""] * batch_size)
                audio_ids = batch.get("audio_ids", [None] * batch_size)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=use_amp,
                    dtype=autocast_dtype,
                ):
                    out = self.unwrapped_model.generate(
                        batch, self.gen_cfg, prompts=prompts, return_outputs=True,
                    )
                    pred_texts = out[0] if isinstance(out, tuple) else out

                pred_texts = texts_for_log(self.unwrapped_model, pred_texts)

                for i in range(batch_size):
                    gt_ans = extract_answer(str(gt_labels[i])).lower()
                    pred_ans = extract_answer(str(pred_texts[i])).lower()

                    if pred_ans != gt_ans:
                        global_idx = samples_processed + i
                        error_indices.append(global_idx)
                        error_audio_ids.append(audio_ids[i] if i < len(audio_ids) else None)
                        error_gt_labels.append(gt_ans)
                        error_pred_labels.append(pred_ans)

                samples_processed += batch_size

                if _is_main:
                    n_errors = len(error_indices)
                    print(
                        f"[SASVDistillation] batch {batch_idx}: processed={samples_processed}, "
                        f"errors={n_errors} ({100 * n_errors / max(samples_processed, 1):.1f}%)",
                        flush=True,
                    )
                    if hasattr(self.logger, "save_dataset_forming_progress"):
                        self.logger.save_dataset_forming_progress(
                            distillation_iter,
                            "error_collection",
                            batch_idx=batch_idx,
                            samples_processed=samples_processed,
                            n_candidates=samples_processed,
                            n_accumulated=n_errors,
                            attempt=0,
                        )
                    if (
                        self.save_intermediate_dataset
                        and self.save_tmp_freq > 0
                        and (batch_idx + 1) % self.save_tmp_freq == 0
                        and hasattr(self.logger, "save_intermediate_dataset")
                    ):
                        path = self.logger.save_intermediate_dataset(
                            distillation_iter,
                            error_indices,
                            lengths_chars=[0] * len(error_indices),
                            lengths_tokens=[0] * len(error_indices),
                            n_raw=samples_processed,
                            audio_ids=error_audio_ids,
                        )
                        print(f"[SASVDistillation] Saved tmp progress to {path}", flush=True)

                self._free_batch(batch)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Gather across ranks in distributed mode
        if dist.is_initialized():
            dist.barrier()
            all_lists = [None] * dist.get_world_size()
            dist.all_gather_object(all_lists, error_indices)
            error_indices = sorted(set(idx for lst in all_lists for idx in (lst or [])))

            all_aid_lists = [None] * dist.get_world_size()
            dist.all_gather_object(all_aid_lists, error_audio_ids)
            error_audio_ids = [aid for lst in all_aid_lists for aid in (lst or [])]

            all_gt_lists = [None] * dist.get_world_size()
            dist.all_gather_object(all_gt_lists, error_gt_labels)
            error_gt_labels = [g for lst in all_gt_lists for g in (lst or [])]

            all_pred_lists = [None] * dist.get_world_size()
            dist.all_gather_object(all_pred_lists, error_pred_labels)
            error_pred_labels = [p for lst in all_pred_lists for p in (lst or [])]

        if _is_main:
            print(
                f"[SASVDistillation] Dataset forming done: "
                f"total={total_size}, errors={len(error_indices)} "
                f"({100 * len(error_indices) / max(total_size, 1):.1f}%)",
                flush=True,
            )

            if self.save_intermediate_dataset and hasattr(self.logger, "save_intermediate_dataset"):
                path = self.logger.save_intermediate_dataset(
                    distillation_iter,
                    error_indices,
                    lengths_chars=[0] * len(error_indices),
                    lengths_tokens=[0] * len(error_indices),
                    n_raw=total_size,
                    audio_ids=error_audio_ids,
                )
                print(f"[SASVDistillation] Saved error dataset to {path}", flush=True)

            if hasattr(self.logger, "log_dataset_forming_stats"):
                self.logger.log_dataset_forming_stats(
                    distillation_iter,
                    n_raw=total_size,
                    n_final=len(error_indices),
                )

            if hasattr(self.logger, "save_dataset_forming_progress"):
                self.logger.save_dataset_forming_progress(
                    distillation_iter,
                    "done",
                    batch_idx=batch_idx,
                    samples_processed=samples_processed,
                    n_candidates=total_size,
                    n_accumulated=len(error_indices),
                    attempt=0,
                )

        return error_indices

    def _free_batch(self, batch: Dict[str, Any]) -> None:
        heavy_keys = [
            "audio", "input_ids", "attention_mask", "pixel_values", "image",
            "input_features", "feature_attention_mask", "spectrogram", "raw_wav",
        ]
        for k in heavy_keys:
            if k in batch:
                del batch[k]

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch
