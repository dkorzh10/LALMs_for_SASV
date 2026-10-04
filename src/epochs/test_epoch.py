from typing import Any, Dict, Optional, List
import os
import numpy as np
import torch
from tqdm import tqdm
from .base import Epoch
from ..loggers.base import Logger
from .utils.confidence import extract_confidences, get_tokenizer, get_token_ids
from .utils.text_utils import texts_for_log
from .utils.sasv_metrics import compute_all_sasv_metrics, print_sasv_metrics_summary


class TestEpoch(Epoch):
    def __init__(self, model, dataloader, logger: Logger, device=None, 
                 gen_cfg: Optional[Dict[str, Any]] = None,
                 model_format: str = "hard_label",
                 dataset_format: str = "hard_label",
                 log_freq: int = 10,
                 extract_confidence: bool = True):
        super().__init__(model, dataloader, logger, device=device)
        self.gen_cfg = gen_cfg or {"max_new_tokens": 256, "num_beams": 1, "do_sample": False}
        self.model_format = model_format
        self.dataset_format = dataset_format
        self.log_freq = log_freq
        self.extract_confidence = extract_confidence
        
        # Running stats for intermediate logging
        self.running_correct = 0
        self.running_total = 0
        
        # Cache token ID lists (support both antispoofing and SASV)
        self._real_token_ids = None
        self._fake_token_ids = None
        self._yes_token_ids = None
        self._no_token_ids = None
        self._gen_token_ids = None

        # Accumulators for ASVspoof5 metrics (same logic as SASVEvalEpoch)
        self._sasv_metric_labels: List[str] = []
        self._sasv_metric_yes_probs: List[float] = []
        self._sasv_metric_gen_probs: List[float] = []
        self._sasv_metric_asv_scores: List[float] = []
        self._sasv_metric_cm_scores: List[float] = []
        _m = getattr(model, "module", model)
        self._has_score_pairs = hasattr(_m, "score_pairs")

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from either format (supports both SASV and antispoofing)."""
        from ..analysis.plotter_common import extract_answer
        return extract_answer(text)

    def _extract_answer_from_gt(self, text: str) -> str:
        """Extract ground truth answer (supports both SASV and antispoofing)."""
        from ..analysis.plotter_common import extract_answer_from_gt
        return extract_answer_from_gt(text)

    def _extract_confidences(self, pred_texts: List[str], logits: torch.Tensor, token_ids: torch.Tensor = None, task_type: str = "antispoofing") -> List[Dict[str, float]]:
        """
        Extract answer class probabilities from logits.
        Supports both antispoofing (Real/Fake) and SASV (yes/no/gen) formats.
        
        Args:
            pred_texts: List of decoded prediction texts
            logits: Logits tensor [batch_size, seq_len, vocab_size]
            token_ids: Optional token IDs tensor [batch_size, seq_len] for more reliable position finding
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

    def _accumulate_sasv_metrics_batch(self, gt_list: List[Any], confidences: List[Dict[str, float]]) -> None:
        """Append trials for compute_all_sasv_metrics (requires extract_confidence + SASV GT)."""
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

    def _resolve_plots_dir(self) -> str:
        """Return run-level plots dir: <run_dir>/plots."""
        log_dir = os.path.abspath(getattr(self.logger, "log_dir", "."))
        log_base = os.path.basename(log_dir)
        if log_base.startswith("test_"):
            run_dir = os.path.dirname(os.path.dirname(log_dir))
        else:
            run_dir = os.path.dirname(log_dir)
        return os.path.join(run_dir, "plots")

    def run(self, **kwargs):
        epoch_num = kwargs.get("epoch_num", 0)
        self.logger.set_epoch(epoch_num, "test")
        self.model.eval()
        self._sasv_metric_labels = []
        self._sasv_metric_yes_probs = []
        self._sasv_metric_gen_probs = []
        self._sasv_metric_asv_scores = []
        self._sasv_metric_cm_scores = []
        
        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(kwargs.get("epoch_num", 0))

        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=self.device.type == "cuda"):
                for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Test"):
                    batch = self._move_to_device(batch, self.device)
                    
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
                    
                    # Generate with logits for confidence extraction
                    if self.extract_confidence:
                        pred_texts, completion_ids, logits = self.unwrapped_model.generate(
                            batch, self.gen_cfg, return_outputs=True
                        )
                        confidences = self._extract_confidences(pred_texts, logits, completion_ids, task_type=task_type)
                    else:
                        pred_texts = self.unwrapped_model.generate(batch, self.gen_cfg)
                        confidences = None
                    
                    # For SASV format, use answer field (just the token) instead of text field
                    if task_type == "sasv" and batch.get("answer"):
                        gt_texts = batch.get("answer", [])
                    else:
                        gt_texts = batch.get("text", [])
                    audio_ids = batch.get("audio_ids", [])
                    
                    # Handle format conversion for reasoning model on hard_label dataset
                    if self.model_format == "reasoning" and self.dataset_format == "hard_label":
                        pred_answers = [self._extract_answer(p) for p in pred_texts]
                        gt_answers = [self._extract_answer_from_gt(g) for g in gt_texts]
                        if (
                            self.extract_confidence
                            and confidences
                            and task_type == "sasv"
                        ):
                            self._accumulate_sasv_metrics_batch(gt_answers, confidences)
                            self._accumulate_subsystem_scores(batch, gt_answers)
                        self.logger.log(
                            loss=loss,
                            outputs=pred_answers,
                            gt=gt_answers,
                            correct=outputs.get("correct"),
                            total=outputs.get("total"),
                            confidences=confidences,
                            audio_ids=audio_ids
                        )
                        # Update running stats
                        for p, g in zip(pred_answers, gt_answers):
                            if p.lower() == g.lower():
                                self.running_correct += 1
                            self.running_total += 1
                    else:
                        if (
                            self.extract_confidence
                            and confidences
                            and task_type == "sasv"
                        ):
                            self._accumulate_sasv_metrics_batch(gt_texts, confidences)
                            self._accumulate_subsystem_scores(batch, gt_texts)
                        self.logger.log(
                            loss=loss,
                            outputs=texts_for_log(self.unwrapped_model, pred_texts),
                            gt=gt_texts,
                            correct=outputs.get("correct"),
                            total=outputs.get("total"),
                            confidences=confidences,
                            audio_ids=audio_ids
                        )
                        # Update running stats (extract answers for comparison)
                        for p, g in zip(pred_texts, gt_texts):
                            p_ans = self._extract_answer(p).lower()
                            g_ans = self._extract_answer_from_gt(g).lower()
                            if p_ans == g_ans:
                                self.running_correct += 1
                            self.running_total += 1
                    
                    # Print intermediate results
                    if (i + 1) % self.log_freq == 0:
                        running_acc = self.running_correct / self.running_total if self.running_total > 0 else 0
                        print(f"  [{i+1}/{len(self.dataloader)}] Running Accuracy: {running_acc:.4f} ({self.running_correct}/{self.running_total})", flush=True)

        if len(self._sasv_metric_labels) > 0:
            kwargs = {}
            n = len(self._sasv_metric_labels)
            if len(self._sasv_metric_asv_scores) == n and len(self._sasv_metric_cm_scores) == n:
                kwargs["asv_scores"] = np.array(self._sasv_metric_asv_scores)
                kwargs["cm_scores"] = np.array(self._sasv_metric_cm_scores)
            sasv_metrics = compute_all_sasv_metrics(
                np.array(self._sasv_metric_labels),
                np.array(self._sasv_metric_yes_probs),
                np.array(self._sasv_metric_gen_probs),
                plot_dir=self._resolve_plots_dir(),
                plot_prefix=f"sasv_test_epoch_{epoch_num}",
                **kwargs,
            )
            print_sasv_metrics_summary(sasv_metrics)
            self.logger._sasv_metrics = sasv_metrics

        self.logger.log_epoch()

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch
