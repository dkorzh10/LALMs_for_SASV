"""SASV evaluation epoch with ASVspoof5 metrics: min a-DCF, min t-DCF, t-EER×."""

from typing import Any, Dict, List, Optional
import os
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from .base import Epoch
from .utils.confidence import extract_confidences, get_tokenizer, get_token_ids
from .utils.text_utils import texts_for_log
from .utils.sasv_metrics import (
    arcface_pipeline_class,
    compute_all_sasv_metrics,
    compute_cosine_sasv_metrics,
    print_sasv_metrics_summary,
)


class SASVEvalEpoch(Epoch):
    """Evaluation epoch specialized for SASV: generates 1-token answers,
    extracts yes/no/gen probabilities and subsystem scores for ASVspoof5 metrics."""

    def __init__(self, model, dataloader, logger, device=None,
                 gen_cfg: Optional[Dict[str, Any]] = None,
                 extract_confidence: bool = True,
                 decision_backend: str = "llm_only",
                 threshold_mode: str = "fixed",
                 tau_sv: Optional[float] = None,
                 tau_spf: Optional[float] = None,
                 threshold_objective: str = "min_a_dcf"):
        super().__init__(model, dataloader, logger, device=device)
        self.gen_cfg = dict(gen_cfg or {"max_new_tokens": 1, "num_beams": 1, "do_sample": False})
        # Force single-token generation for SASV.
        self.gen_cfg["max_new_tokens"] = 1
        if not self.gen_cfg.get("do_sample", False):
            self.gen_cfg.pop("top_p", None)
            self.gen_cfg.pop("temperature", None)
        self.extract_confidence = extract_confidence
        self.decision_backend = decision_backend
        self.threshold_mode = threshold_mode
        self.tau_sv = tau_sv
        self.tau_spf = tau_spf
        self.threshold_objective = threshold_objective

        # Token ID caches
        self._yes_token_ids = None
        self._no_token_ids = None
        self._gen_token_ids = None

        # Accumulators for ASVspoof5 metrics
        self._labels = []
        self._yes_probs = []
        self._gen_probs = []
        self._asv_scores = []
        self._cm_scores = []
        self._cos_scores = []
        self._spoof_probs = []
        self._arcface_warned = False
        self._threshold_output_warned = False
        _m = getattr(model, "module", model)
        self._has_score_pairs = hasattr(_m, "score_pairs")

    @staticmethod
    def _is_main_process() -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    @staticmethod
    def _gather_lists(local_list: List) -> List:
        """Concatenate per-rank lists for distributed validation."""
        if not dist.is_initialized():
            return local_list
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_list)
        merged = []
        for part in gathered:
            merged.extend(part)
        return merged

    def _extract_answer(self, text: str) -> str:
        from ..analysis.plotter_common import extract_answer
        return extract_answer(text)

    def _extract_confidences(
        self,
        pred_texts: List[str],
        logits: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, float]]:
        if logits is None:
            return [{"yes_prob": 0.0, "no_prob": 0.0, "gen_prob": 0.0, "confidence": 0.33} for _ in pred_texts]

        if logits.dim() == 2 and logits.shape[-1] == 3:
            return extract_confidences(
                pred_texts=pred_texts,
                logits=logits,
                tokenizer=None,
                token_ids=None,
                extract_answer_fn=self._extract_answer,
                task_type="sasv",
            )

        tokenizer = get_tokenizer(self.unwrapped_model)
        if tokenizer is None:
            return [{"yes_prob": 0.0, "no_prob": 0.0, "gen_prob": 0.0, "confidence": 0.33} for _ in pred_texts]

        cache = {
            '_yes_token_ids': self._yes_token_ids,
            '_no_token_ids': self._no_token_ids,
            '_gen_token_ids': self._gen_token_ids,
        }
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
            task_type="sasv",
        )

    def _resolve_plots_dir(self) -> str:
        """Return run-level plots dir: <run_dir>/plots."""
        log_dir = os.path.abspath(getattr(self.logger, "log_dir", "."))
        log_base = os.path.basename(log_dir)
        if log_base.startswith("test_"):
            run_dir = os.path.dirname(os.path.dirname(log_dir))
        else:
            run_dir = os.path.dirname(log_dir)
        return os.path.join(run_dir, "plots")

    @staticmethod
    def _batch_float(values, batch_idx: int) -> Optional[float]:
        if values is None:
            return None
        try:
            if isinstance(values, torch.Tensor):
                if batch_idx >= values.shape[0]:
                    return None
                return float(values[batch_idx].detach().item())
            if batch_idx >= len(values):
                return None
            return float(values[batch_idx])
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _backend_uses_arcface(decision_backend: str) -> bool:
        return decision_backend in {"arcface_plus_llm_spoof", "arcface_only"}

    def _warn_arcface_fallback(self, message: str) -> None:
        if not self._arcface_warned:
            print(message, flush=True)
            self._arcface_warned = True

    def run(self, **kwargs):
        epoch_num = kwargs.get("epoch_num", 0)
        self.logger.set_epoch(epoch_num, "validation")
        self.model.eval()

        # Reset accumulators
        self._labels = []
        self._yes_probs = []
        self._gen_probs = []
        self._asv_scores = []
        self._cm_scores = []
        self._cos_scores = []
        self._spoof_probs = []
        self._arcface_warned = False
        self._threshold_output_warned = False

        if (
            self.decision_backend == "arcface_only"
            and getattr(self.unwrapped_model, "bonafide_spoof_head", None) is None
        ):
            self._warn_arcface_fallback(
                "Warning: arcface_only requested but bonafide_spoof_head is unavailable; "
                "falling back to LLM-only metrics."
            )

        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch_num)

        autocast_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=self.device.type == "cuda"):
                for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="SASV Eval"):
                    batch = self._move_to_device(batch, self.device)

                    # Forward for loss
                    outputs = self.model(batch, verbose=True)
                    loss = outputs["loss"].item()

                    gt_texts = batch.get("answer", batch.get("text", []))
                    audio_ids = batch.get("audio_ids", [])
                    cos_scores_batch = None
                    ce2_spoof_probs_batch = None

                    if self._has_score_pairs:
                        pair_scores = self.unwrapped_model.score_pairs(batch)
                        cos_scores_batch = pair_scores.get("cosine_sim")
                        bonafide_probs_batch = pair_scores.get("bonafide_prob")
                        if bonafide_probs_batch is not None:
                            ce2_spoof_probs_batch = 1.0 - bonafide_probs_batch

                        if cos_scores_batch is not None and bonafide_probs_batch is not None:
                            for g, c_sim, b_prob in zip(
                                gt_texts,
                                cos_scores_batch.detach().cpu().tolist(),
                                bonafide_probs_batch.detach().cpu().tolist(),
                            ):
                                g_ans = self._extract_answer(g).lower()
                                if g_ans in ("yes", "no", "gen"):
                                    self._asv_scores.append(float(c_sim))
                                    self._cm_scores.append(float(b_prob))
                    elif self._backend_uses_arcface(self.decision_backend):
                        if hasattr(self.unwrapped_model, "extract_arcface_embeddings"):
                            emb_out = self.unwrapped_model.extract_arcface_embeddings(batch)
                            if isinstance(emb_out, dict):
                                cos_scores_batch = emb_out.get("cos_scores")
                                ce2_spoof_probs_batch = emb_out.get("query_ce2_spoof_probs")
                        else:
                            self._warn_arcface_fallback(
                                f"Warning: {self.decision_backend} requested but ArcFace scores are unavailable; "
                                "falling back to LLM-only metrics."
                            )

                    # Generate + extract confidences
                    confidences = None
                    if self.extract_confidence:
                        pred_texts, completion_ids, logits = self.unwrapped_model.generate(
                            batch, self.gen_cfg, return_outputs=True
                        )
                        confidences = self._extract_confidences(pred_texts, logits, completion_ids)

                        # Accumulate for SASV metrics
                        for batch_idx, (g, conf) in enumerate(zip(gt_texts, confidences)):
                            g_ans = self._extract_answer(g).lower()
                            if g_ans in ("yes", "no", "gen"):
                                self._labels.append(g_ans)
                                self._yes_probs.append(conf.get("yes_prob", 0.0))
                                self._gen_probs.append(conf.get("gen_prob", 0.0))
                                cos_score = self._batch_float(cos_scores_batch, batch_idx)
                                ce2_spoof_prob = self._batch_float(ce2_spoof_probs_batch, batch_idx)
                                if self._backend_uses_arcface(self.decision_backend) and cos_score is not None:
                                    self._cos_scores.append(cos_score)
                                    if self.decision_backend == "arcface_only":
                                        if ce2_spoof_prob is not None:
                                            self._spoof_probs.append(ce2_spoof_prob)
                                    else:
                                        self._spoof_probs.append(float(conf.get("gen_prob", 0.0)))
                                if confidences is not None:
                                    if cos_score is not None:
                                        conf["cos_score"] = cos_score
                                    if ce2_spoof_prob is not None:
                                        conf["ce2_spoof_prob"] = ce2_spoof_prob
                                    if (
                                        self._backend_uses_arcface(self.decision_backend)
                                        and self.threshold_mode == "fixed"
                                        and self.tau_sv is not None
                                        and self.tau_spf is not None
                                        and cos_score is not None
                                    ):
                                        spoof_prob = ce2_spoof_prob if self.decision_backend == "arcface_only" else float(conf.get("gen_prob", 0.0))
                                        if spoof_prob is not None:
                                            conf["spoof_prob"] = float(spoof_prob)
                                            conf["tau_sv_used"] = float(self.tau_sv)
                                            conf["tau_spf_used"] = float(self.tau_spf)
                                            conf["pred_arcface"] = arcface_pipeline_class(
                                                cos_score, float(spoof_prob), float(self.tau_sv), float(self.tau_spf)
                                            )
                    else:
                        pred_texts = self.unwrapped_model.generate(batch, self.gen_cfg)

                    log_outputs = texts_for_log(self.unwrapped_model, pred_texts)
                    output_source = "llm_generation"
                    if self.decision_backend != "llm_only":
                        thresholded_outputs = [
                            conf.get("pred_arcface") for conf in confidences
                        ] if confidences is not None else []
                        if len(thresholded_outputs) == len(log_outputs) and all(
                            pred is not None for pred in thresholded_outputs
                        ):
                            log_outputs = thresholded_outputs
                            output_source = f"{self.decision_backend}_thresholded"
                        else:
                            output_source = f"{self.decision_backend}_fallback_llm_generation"
                            if not self._threshold_output_warned:
                                print(
                                    f"Warning: {self.decision_backend} requested but thresholded decisions "
                                    "were unavailable for logger outputs; using LLM generation for accuracy.",
                                    flush=True,
                                )
                                self._threshold_output_warned = True

                    self.logger.log(
                        loss=loss,
                        outputs=log_outputs,
                        gt=gt_texts,
                        correct=outputs.get("correct"),
                        total=outputs.get("total"),
                        confidences=confidences,
                        audio_ids=audio_ids,
                        output_source=output_source,
                    )

        # Aggregate trials from all ranks, then compute DCF / EER on rank 0
        all_labels = self._gather_lists(self._labels)
        all_yes_probs = self._gather_lists(self._yes_probs)
        all_gen_probs = self._gather_lists(self._gen_probs)
        all_asv = self._gather_lists(self._asv_scores) if self._asv_scores else None
        all_cm = self._gather_lists(self._cm_scores) if self._cm_scores else None
        all_cos = self._gather_lists(self._cos_scores) if self._cos_scores else None
        all_spoof = self._gather_lists(self._spoof_probs) if self._spoof_probs else None

        if dist.is_initialized():
            dist.barrier()

        if self._is_main_process() and len(all_labels) > 0:
            kwargs = {}
            if all_asv and all_cm and len(all_asv) == len(all_labels):
                kwargs["asv_scores"] = np.array(all_asv)
                kwargs["cm_scores"] = np.array(all_cm)
            sasv_metrics = compute_all_sasv_metrics(
                np.array(all_labels),
                np.array(all_yes_probs),
                np.array(all_gen_probs),
                plot_dir=self._resolve_plots_dir(),
                plot_prefix=f"sasv_eval_epoch_{epoch_num}",
                **kwargs,
            )
            print_sasv_metrics_summary(sasv_metrics)
            if (
                self._backend_uses_arcface(self.decision_backend)
                and all_cos is not None
                and all_spoof is not None
                and len(all_cos) == len(all_labels)
                and len(all_spoof) == len(all_labels)
            ):
                arcface_metrics = compute_cosine_sasv_metrics(
                    np.array(all_labels),
                    np.array(all_cos),
                    np.array(all_spoof),
                    tau_sv=self.tau_sv,
                    tau_spf=self.tau_spf,
                    threshold_mode=self.threshold_mode,
                    threshold_objective=self.threshold_objective,
                )
                sasv_metrics = {
                    **arcface_metrics,
                    **{f"llm_{k}": v for k, v in sasv_metrics.items()},
                }
                tau_sv_sel = float(arcface_metrics["tau_sv_selected"])
                tau_spf_sel = float(arcface_metrics["tau_spf_selected"])
                trials = []
                for idx in range(len(all_labels)):
                    trial = {
                        "gt": all_labels[idx],
                        "cos_score": float(all_cos[idx]),
                        "spoof_prob": float(all_spoof[idx]),
                        "gen_prob": float(all_gen_probs[idx]),
                        "yes_prob": float(all_yes_probs[idx]),
                        "tau_sv_used": tau_sv_sel,
                        "tau_spf_used": tau_spf_sel,
                        "pred_arcface": arcface_pipeline_class(
                            float(all_cos[idx]), float(all_spoof[idx]), tau_sv_sel, tau_spf_sel
                        ),
                    }
                    if self.decision_backend == "arcface_only":
                        trial["ce2_spoof_prob"] = float(all_spoof[idx])
                    trials.append(trial)
                setattr(self.logger, "_arcface_per_trial", trials)
                print_sasv_metrics_summary(arcface_metrics)
            elif self._backend_uses_arcface(self.decision_backend):
                print(
                    f"Warning: {self.decision_backend} requested but cosine/spoof scores were missing or misaligned "
                    f"(n_cos={0 if all_cos is None else len(all_cos)}, "
                    f"n_spoof={0 if all_spoof is None else len(all_spoof)}, n_lab={len(all_labels)}); "
                    "using LLM-only SASV metrics.",
                    flush=True,
                )
            self.logger._sasv_metrics = sasv_metrics

        # if self._is_main_process():
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
