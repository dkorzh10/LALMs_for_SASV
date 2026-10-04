from typing import Any, Dict, Optional, List
import os
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from .base import Epoch
from .utils.confidence import extract_confidences, get_tokenizer, get_token_ids
from .utils.text_utils import texts_for_log
from .utils.sasv_metrics import compute_all_sasv_metrics, print_sasv_metrics_summary


class EvalEpoch(Epoch):
    def __init__(self, model, dataloader, logger, device=None, gen_cfg: Optional[Dict[str, Any]] = None, extract_confidence: bool = True):
        super().__init__(model, dataloader, logger, device=device)
        # Default generation config, can be overridden via constructor
        self.gen_cfg = gen_cfg or {"max_new_tokens": 256, "num_beams": 1, "do_sample": False}
        self.extract_confidence = extract_confidence
        
        # Cache token ID lists (support both antispoofing and SASV)
        self._real_token_ids = None
        self._fake_token_ids = None
        self._yes_token_ids = None
        self._no_token_ids = None
        self._gen_token_ids = None

        # ASVspoof5 SASV metrics (same accumulation as SASVEvalEpoch / TestEpoch)
        self._sasv_metric_labels: List[str] = []
        self._sasv_metric_yes_probs: List[float] = []
        self._sasv_metric_gen_probs: List[float] = []
        self._sasv_metric_asv_scores: List[float] = []
        self._sasv_metric_cm_scores: List[float] = []
        _m = getattr(model, "module", model)
        self._has_score_pairs = hasattr(_m, "score_pairs")

    @staticmethod
    def _is_main_process() -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    @staticmethod
    def _gather_lists(local_list: List) -> List:
        if not dist.is_initialized():
            return local_list
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_list)
        merged = []
        for part in gathered:
            merged.extend(part)
        return merged

    def _resolve_plots_dir(self) -> str:
        log_dir = os.path.abspath(getattr(self.logger, "log_dir", "."))
        log_base = os.path.basename(log_dir)
        if log_base.startswith("test_"):
            run_dir = os.path.dirname(os.path.dirname(log_dir))
        else:
            run_dir = os.path.dirname(log_dir)
        return os.path.join(run_dir, "plots")

    def _accumulate_sasv_metrics_batch(
        self, gt_list: List[Any], confidences: List[Dict[str, float]]
    ) -> None:
        for g, conf in zip(gt_list, confidences):
            g_ans = self._extract_answer(g).lower()
            if g_ans in ("yes", "no", "gen"):
                self._sasv_metric_labels.append(g_ans)
                self._sasv_metric_yes_probs.append(conf.get("yes_prob", 0.0))
                self._sasv_metric_gen_probs.append(conf.get("gen_prob", 0.0))

    def _accumulate_subsystem_scores(self, batch, gt_list: List[Any]) -> None:
        if not self._has_score_pairs:
            return
        pair_scores = self.unwrapped_model.score_pairs(batch)
        cos = pair_scores.get("cosine_sim")
        bon = pair_scores.get("bonafide_prob")
        if cos is None or bon is None:
            return
        for g, c_sim, b_prob in zip(
            gt_list, cos.detach().cpu().tolist(), bon.detach().cpu().tolist()
        ):
            g_ans = self._extract_answer(g).lower()
            if g_ans in ("yes", "no", "gen"):
                self._sasv_metric_asv_scores.append(float(c_sim))
                self._sasv_metric_cm_scores.append(float(b_prob))

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from either format (supports both SASV and antispoofing)."""
        from ..analysis.plotter_common import extract_answer
        return extract_answer(text)

    def _extract_confidences(
        self,
        pred_texts: List[str],
        logits: torch.Tensor,
        token_ids: torch.Tensor = None,
        task_type: str = "antispoofing",
    ) -> List[Dict[str, float]]:
        """
        Extract answer class probabilities from logits.
        Supports both antispoofing (Real/Fake) and SASV (yes/no/gen) formats.
        
        Args:
            pred_texts: List of decoded prediction texts
            logits: Logits tensor [batch_size, seq_len, vocab_size]
            token_ids: Optional completion token ids for answer-position lookup
            task_type: "antispoofing" or "sasv"
        
        Returns:
            List of dicts with probabilities and confidence
        """
        if logits is None:
            if task_type == "sasv":
                return [{"yes_prob": 0.0, "no_prob": 0.0, "gen_prob": 0.0, "confidence": 0.33} for _ in pred_texts]
            return [{"real_prob": 0.0, "fake_prob": 0.0, "confidence": 0.5} for _ in pred_texts]

        if task_type == "sasv" and logits.dim() == 2 and logits.shape[-1] == 3:
            return extract_confidences(
                pred_texts=pred_texts,
                logits=logits,
                tokenizer=None,
                token_ids=token_ids,
                extract_answer_fn=self._extract_answer,
                task_type="sasv",
            )

        tokenizer = get_tokenizer(self.unwrapped_model)
        if tokenizer is None:
            if task_type == "sasv":
                return [{"yes_prob": 0.0, "no_prob": 0.0, "gen_prob": 0.0, "confidence": 0.33} for _ in pred_texts]
            return [{"real_prob": 0.0, "fake_prob": 0.0, "confidence": 0.5} for _ in pred_texts]
        
        if task_type == "sasv":
            # Use instance cache for SASV token IDs
            cache = {'_yes_token_ids': self._yes_token_ids, '_no_token_ids': self._no_token_ids, '_gen_token_ids': self._gen_token_ids}
            yes_ids, no_ids, gen_ids = get_token_ids(tokenizer, cache, task_type="sasv")
            self._yes_token_ids = cache['_yes_token_ids']
            self._no_token_ids = cache['_no_token_ids']
            self._gen_token_ids = cache['_gen_token_ids']
            
            return extract_confidences(
                pred_texts=pred_texts,
                logits=logits,
                tokenizer=tokenizer,
                yes_ids=yes_ids,
                no_ids=no_ids,
                gen_ids=gen_ids,
                token_ids=token_ids,
                extract_answer_fn=self._extract_answer,
                task_type="sasv"
            )
        else:
            # Use instance cache for antispoofing token IDs
            cache = {'_real_token_ids': self._real_token_ids, '_fake_token_ids': self._fake_token_ids}
            real_ids, fake_ids = get_token_ids(tokenizer, cache, task_type="antispoofing")
            self._real_token_ids = cache['_real_token_ids']
            self._fake_token_ids = cache['_fake_token_ids']
            
            return extract_confidences(
                pred_texts=pred_texts,
                logits=logits,
                tokenizer=tokenizer,
                real_ids=real_ids,
                fake_ids=fake_ids,
                token_ids=token_ids,
                extract_answer_fn=self._extract_answer,
                task_type="antispoofing"
            )

    def _generate_batch(self, batch):
        if self.extract_confidence:
            try:
                result = self.unwrapped_model.generate(
                    batch, self.gen_cfg, return_outputs=True, return_generation_info=True
                )
                if isinstance(result, tuple) and len(result) == 4:
                    pred_texts, completion_ids, logits, generation_info = result
                    return pred_texts, completion_ids, logits, generation_info
            except TypeError:
                pass
            pred_texts, completion_ids, logits = self.unwrapped_model.generate(
                batch, self.gen_cfg, return_outputs=True
            )
            return pred_texts, completion_ids, logits, None

        try:
            result = self.unwrapped_model.generate(
                batch, self.gen_cfg, return_generation_info=True
            )
            if isinstance(result, tuple) and len(result) == 2:
                pred_texts, generation_info = result
                return pred_texts, None, None, generation_info
        except TypeError:
            pass
        pred_texts = self.unwrapped_model.generate(batch, self.gen_cfg)
        return pred_texts, None, None, None

    def _generation_length_stats(
        self, pred_texts: List[str], generation_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track generation length caps from model metadata, falling back to decoded text."""
        if generation_info:
            generated_lens = [int(x) for x in generation_info.get("generated_lens", [])]
            prefix_lens = [int(x) for x in generation_info.get("prefix_lens", [])]
            total_lens = [int(x) for x in generation_info.get("total_lens", [])]
            max_new_tokens = generation_info.get("max_new_tokens", self.gen_cfg.get("max_new_tokens"))
            context_window = generation_info.get("context_window")
            if not generated_lens:
                return {}

            stats = {
                "generation_max_new_tokens": int(max_new_tokens),
                "generation_max_new_tokens_hits": int(sum(length >= int(max_new_tokens) for length in generated_lens)),
                "generation_output_count": int(len(generated_lens)),
                "generation_output_token_total": int(sum(generated_lens)),
                "generation_output_token_max": int(max(generated_lens)),
                "generation_prefix_token_total": int(sum(prefix_lens)),
                "generation_prefix_token_max": int(max(prefix_lens)) if prefix_lens else 0,
                "generation_total_token_total": int(sum(total_lens)),
                "generation_total_token_max": int(max(total_lens)) if total_lens else 0,
                "generation_padded_prefix_len": int(generation_info.get("padded_prefix_len", 0)),
            }
            if context_window is not None:
                stats["generation_context_window"] = int(context_window)
                stats["generation_context_window_hits"] = int(
                    sum(length >= int(context_window) for length in total_lens)
                )
            return stats

        # Fallback for models that do not expose generation metadata: estimate output
        # length from decoded text. This cannot include prompt/audio prefix lengths.
        max_new_tokens = self.gen_cfg.get("max_new_tokens")
        if not max_new_tokens:
            return {}

        tokenizer = get_tokenizer(self.unwrapped_model)
        if tokenizer is None:
            return {}

        lengths = [
            len(tokenizer(str(text), add_special_tokens=False).input_ids)
            for text in pred_texts
        ]
        hit_count = sum(length >= max_new_tokens for length in lengths)
        total = len(lengths)
        return {
            "generation_max_new_tokens": int(max_new_tokens),
            "generation_max_new_tokens_hits": int(hit_count),
            "generation_output_count": int(total),
            "generation_output_token_total": int(sum(lengths)),
            "generation_output_token_max": int(max(lengths))
        }

    def run(self, **kwargs):
        epoch_num = kwargs.get("epoch_num", 0)
        self.logger.set_epoch(epoch_num, "validation")
        self.model.eval()
        self._sasv_metric_labels = []
        self._sasv_metric_yes_probs = []
        self._sasv_metric_gen_probs = []
        self._sasv_metric_asv_scores = []
        self._sasv_metric_cm_scores = []
        
        # If using distributed sampler, set epoch
        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch_num)

        # Determine autocast dtype - use bfloat16 if supported, else float16
        autocast_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        
        with torch.no_grad():
            # Use same autocast as training with correct dtype
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=self.device.type == "cuda"):
                for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Eval"):
                    batch = self._move_to_device(batch, self.device)
                    
                    # Call forward with verbose=True to get accuracy-related metrics
                    # outputs = self.model.forward(batch, verbose=True)
                    outputs = self.model(batch, verbose=True)
                    loss = outputs["loss"].item()
                    
                    # Detect task type from batch
                    task_type = "antispoofing"
                    if batch.get("task") and isinstance(batch["task"], list) and len(batch["task"]) > 0:
                        if batch["task"][0] == "sasv":
                            task_type = "sasv"
                    elif batch.get("task") == "sasv":
                        task_type = "sasv"
                    
                    # For SASV format, use answer field (just the token) instead of text field
                    if task_type == "sasv" and batch.get("answer"):
                        gt_texts = batch.get("answer", [])
                    else:
                        gt_texts = batch.get("text", [])
                    audio_ids = batch.get("audio_ids", [])

                    pred_texts, completion_ids, logits, generation_info = self._generate_batch(batch)
                    if self.extract_confidence:
                        confidences = self._extract_confidences(
                            pred_texts, logits, token_ids=completion_ids, task_type=task_type
                        )
                    else:
                        confidences = None

                    generation_stats = self._generation_length_stats(pred_texts, generation_info)

                    if self.extract_confidence and confidences and task_type == "sasv":
                        self._accumulate_sasv_metrics_batch(gt_texts, confidences)
                        self._accumulate_subsystem_scores(batch, gt_texts)

                    self.logger.log(
                        loss=loss, 
                        outputs=texts_for_log(self.unwrapped_model, pred_texts), 
                        gt=gt_texts,
                        correct=outputs.get("correct"),
                        total=outputs.get("total"),
                        confidences=confidences,
                        audio_ids=audio_ids,
                        **generation_stats,
                    )

        all_labels = self._gather_lists(self._sasv_metric_labels)
        all_yes_probs = self._gather_lists(self._sasv_metric_yes_probs)
        all_gen_probs = self._gather_lists(self._sasv_metric_gen_probs)
        all_asv = self._gather_lists(self._sasv_metric_asv_scores) if self._sasv_metric_asv_scores else None
        all_cm = self._gather_lists(self._sasv_metric_cm_scores) if self._sasv_metric_cm_scores else None

        if dist.is_initialized():
            dist.barrier()

        if self._is_main_process() and len(all_labels) > 0:
            metric_kwargs = {}
            if all_asv and all_cm and len(all_asv) == len(all_labels):
                metric_kwargs["asv_scores"] = np.array(all_asv)
                metric_kwargs["cm_scores"] = np.array(all_cm)
            sasv_metrics = compute_all_sasv_metrics(
                np.array(all_labels),
                np.array(all_yes_probs),
                np.array(all_gen_probs),
                plot_dir=self._resolve_plots_dir(),
                plot_prefix=f"sasv_eval_epoch_{epoch_num}",
                **metric_kwargs,
            )
            print_sasv_metrics_summary(sasv_metrics)
            self.logger._sasv_metrics = sasv_metrics

        self.logger.log_epoch()

        if dist.is_initialized():
            dist.barrier()

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch



