from typing import Any, Optional
import os
import json
import numpy as np
import torch.distributed as dist
from .base import Logger

class HardLabelLogger(Logger):
    def __init__(self, log_dir: str, log_freq: int, save_all_predictions: bool = False):
        super().__init__(log_dir, log_freq)
        self.save_all_predictions = save_all_predictions
        self._predictions_file = None
        self.audio_ids = []
        self._train_ce1_losses: list = []
        self._train_ce2_losses: list = []
        self._train_arcface_losses: list = []
        self._train_asv_pair_losses: list = []
        self._reset_confidence_stats()

    def reset_epoch(self):
        super().reset_epoch()
        self.audio_ids = []
        self.output_sources = []
        self.ce1_losses = []
        self.ce2_losses = []
        self.arcface_losses = []
        self.arcface_layer_cosine_mean = []
        self.arcface_layer_cosine_min = []
        self.arcface_layer_cosine_max = []
        self.arcface_layer_cosine_enroll_mean = []
        self.arcface_layer_cosine_enroll_min = []
        self.arcface_layer_cosine_enroll_max = []
        self.arcface_layer_cosine_query_mean = []
        self.arcface_layer_cosine_query_min = []
        self.arcface_layer_cosine_query_max = []
        self.arcface_layer_weights_grad_norm = []
        self.arcface_layer_weights_update_norm = []
        self.arcface_layer_weights_lr = []
        self._train_ce1_losses = []
        self._train_ce2_losses = []
        self._train_arcface_losses = []
        self._train_asv_pair_losses = []
        self._reset_confidence_stats()

    def _reset_confidence_stats(self):
        """Reset confidence tracking for new epoch."""
        self.all_confidences = []
        self.correct_confidences = []
        self.incorrect_confidences = []
        self.real_confidences = []
        self.fake_confidences = []
        
        # SASV format: three classes
        self.yes_confidences = []
        self.no_confidences = []
        self.gen_confidences = []
        
        # TP/FP/TN/FN confidence tracking (antispoofing format)
        self.tp_confidences = []  # True Positive: predicted Fake, actually Fake
        self.fp_confidences = []  # False Positive: predicted Fake, actually Real
        self.tn_confidences = []  # True Negative: predicted Real, actually Real
        self.fn_confidences = []  # False Negative: predicted Real, actually Fake
        
        # Confidence by predicted class and correctness (for plotting)
        # Antispoofing: predicted Fake (TP=correct, FP=incorrect), predicted Real (TN=correct, FN=incorrect)
        self.pred_fake_correct_confidences = []  # TP: predicted Fake, correct
        self.pred_fake_incorrect_confidences = []  # FP: predicted Fake, incorrect
        self.pred_real_correct_confidences = []  # TN: predicted Real, correct
        self.pred_real_incorrect_confidences = []  # FN: predicted Real, incorrect
        
        # SASV: predicted Yes/No/Gen, split by correct/incorrect
        self.pred_yes_correct_confidences = []
        self.pred_yes_incorrect_confidences = []
        self.pred_no_correct_confidences = []
        self.pred_no_incorrect_confidences = []
        self.pred_gen_correct_confidences = []
        self.pred_gen_incorrect_confidences = []
    
        self.eer_labels = []      # ground truth class per sample (str)
        self.eer_yes_probs = []   # yes_prob per sample
        self.eer_gen_probs = []   # gen_prob per sample
        self.eer_real_probs = []  # real_prob per sample

    def set_epoch(self, epoch_num: int, epoch_type: str):
        super().set_epoch(epoch_num, epoch_type)
        self._reset_confidence_stats()
        
        # Open predictions file for streaming writes
        if self.save_all_predictions and epoch_type in ["validation", "test"]:
            os.makedirs(self.log_dir, exist_ok=True)
            # Add rank suffix for distributed training
            rank_suffix = ""
            if dist.is_initialized():
                rank = dist.get_rank()
                rank_suffix = f"_rank{rank}"
            pred_path = os.path.join(self.log_dir, f"predictions_{epoch_type}_epoch_{epoch_num}{rank_suffix}.jsonl")
            self._predictions_file = open(pred_path, "w")
        else:
            self._predictions_file = None

    def log(self, loss: Optional[float] = None, lr: Optional[float] = None, 
            outputs: Optional[Any] = None, gt: Optional[Any] = None, 
            correct: Optional[float] = None, total: Optional[float] = None,
            confidences: Optional[list] = None, audio_ids: Optional[list] = None, **kwargs):
        
        if loss is not None:
            self.losses.append(loss)
        if lr is not None:
            self.lrs.append(lr)
        if outputs is not None:
            self.outputs.extend(outputs if isinstance(outputs, list) else [outputs])
        if gt is not None:
            self.gts.extend(gt if isinstance(gt, list) else [gt])
        if audio_ids is not None:
            self.audio_ids.extend(audio_ids if isinstance(audio_ids, list) else [audio_ids])
        output_source = kwargs.get("output_source")
        if output_source is not None:
            self.output_sources.append(str(output_source))

        ce1_loss = kwargs.get("ce1_loss")
        if ce1_loss is None:
            ce1_loss = kwargs.get("ce_loss")
        if ce1_loss is not None:
            self.ce1_losses.append(float(ce1_loss))

        ce2_loss = kwargs.get("ce2_loss")
        if ce2_loss is not None:
            self.ce2_losses.append(float(ce2_loss))

        arcface_loss = kwargs.get("arcface_loss")
        if arcface_loss is not None:
            self.arcface_losses.append(float(arcface_loss))

        if kwargs.get("arcface_layer_cosine_mean") is not None:
            self.arcface_layer_cosine_mean.append(float(kwargs["arcface_layer_cosine_mean"]))
        if kwargs.get("arcface_layer_cosine_min") is not None:
            self.arcface_layer_cosine_min.append(float(kwargs["arcface_layer_cosine_min"]))
        if kwargs.get("arcface_layer_cosine_max") is not None:
            self.arcface_layer_cosine_max.append(float(kwargs["arcface_layer_cosine_max"]))
        if kwargs.get("arcface_layer_cosine_enroll_mean") is not None:
            self.arcface_layer_cosine_enroll_mean.append(float(kwargs["arcface_layer_cosine_enroll_mean"]))
        if kwargs.get("arcface_layer_cosine_enroll_min") is not None:
            self.arcface_layer_cosine_enroll_min.append(float(kwargs["arcface_layer_cosine_enroll_min"]))
        if kwargs.get("arcface_layer_cosine_enroll_max") is not None:
            self.arcface_layer_cosine_enroll_max.append(float(kwargs["arcface_layer_cosine_enroll_max"]))
        if kwargs.get("arcface_layer_cosine_query_mean") is not None:
            self.arcface_layer_cosine_query_mean.append(float(kwargs["arcface_layer_cosine_query_mean"]))
        if kwargs.get("arcface_layer_cosine_query_min") is not None:
            self.arcface_layer_cosine_query_min.append(float(kwargs["arcface_layer_cosine_query_min"]))
        if kwargs.get("arcface_layer_cosine_query_max") is not None:
            self.arcface_layer_cosine_query_max.append(float(kwargs["arcface_layer_cosine_query_max"]))
        if kwargs.get("arcface_layer_weights_grad_norm") is not None:
            self.arcface_layer_weights_grad_norm.append(float(kwargs["arcface_layer_weights_grad_norm"]))
        if kwargs.get("arcface_layer_weights_update_norm") is not None:
            self.arcface_layer_weights_update_norm.append(float(kwargs["arcface_layer_weights_update_norm"]))
        if kwargs.get("arcface_layer_weights_lr") is not None:
            self.arcface_layer_weights_lr.append(float(kwargs["arcface_layer_weights_lr"]))
        
        # Track confidences
        if confidences and outputs and gt and self.epoch_type in ["validation", "test"]:
            outputs_list = outputs if isinstance(outputs, list) else [outputs]
            gt_list = gt if isinstance(gt, list) else [gt]
            
            # Detect task type from first confidence dict
            is_sasv = "yes_prob" in confidences[0] if confidences else False
            
            for o, g, conf in zip(outputs_list, gt_list, confidences):
                self.all_confidences.append(conf["confidence"])
                # Extract answers for comparison
                o_ans = self._extract_answer(o).lower()
                g_ans = self._extract_answer(g).lower()
                is_correct = (o_ans == g_ans)
                
                if is_correct:
                    self.correct_confidences.append(conf["confidence"])
                else:
                    self.incorrect_confidences.append(conf["confidence"])
                
                if is_sasv:
                    # SASV format: three classes
                    if "yes_prob" in conf:
                        if g_ans == "yes":
                            self.yes_confidences.append(conf["yes_prob"])
                        if g_ans == "no":
                            self.no_confidences.append(conf["no_prob"])
                        if g_ans == "gen":
                            self.gen_confidences.append(conf["gen_prob"])

                        self.eer_labels.append(g_ans)
                        self.eer_yes_probs.append(conf["yes_prob"])
                        self.eer_gen_probs.append(conf["gen_prob"])

                    
                    # Track by predicted class and correctness
                    if o_ans == "yes":
                        if is_correct:
                            self.pred_yes_correct_confidences.append(conf["confidence"])
                        else:
                            self.pred_yes_incorrect_confidences.append(conf["confidence"])
                    elif o_ans == "no":
                        if is_correct:
                            self.pred_no_correct_confidences.append(conf["confidence"])
                        else:
                            self.pred_no_incorrect_confidences.append(conf["confidence"])
                    elif o_ans == "gen":
                        if is_correct:
                            self.pred_gen_correct_confidences.append(conf["confidence"])
                        else:
                            self.pred_gen_incorrect_confidences.append(conf["confidence"])
                else:
                    # Antispoofing format: two classes
                    # Per-class confidence
                    if g_ans == "real":
                        self.real_confidences.append(conf["confidence"])
                    elif g_ans == "fake":
                        self.fake_confidences.append(conf["confidence"])
                    
                    if "real_prob" in conf and g_ans in ("real", "fake"):
                        self.eer_labels.append(g_ans)
                        self.eer_real_probs.append(conf["real_prob"])
                    
                    # TP/FP/TN/FN tracking
                    # "Fake" is the positive class for CM metrics
                    if o_ans == "fake" and g_ans == "fake":
                        self.tp_confidences.append(conf["confidence"])  # True Positive
                        self.pred_fake_correct_confidences.append(conf["confidence"])
                    elif o_ans == "fake" and g_ans == "real":
                        self.fp_confidences.append(conf["confidence"])  # False Positive
                        self.pred_fake_incorrect_confidences.append(conf["confidence"])
                    elif o_ans == "real" and g_ans == "real":
                        self.tn_confidences.append(conf["confidence"])  # True Negative
                        self.pred_real_correct_confidences.append(conf["confidence"])
                    elif o_ans == "real" and g_ans == "fake":
                        self.fn_confidences.append(conf["confidence"])  # False Negative
                        self.pred_real_incorrect_confidences.append(conf["confidence"])
        
        # Stream predictions to file
        if self._predictions_file and outputs is not None and gt is not None:
            outputs_list = outputs if isinstance(outputs, list) else [outputs]
            gt_list = gt if isinstance(gt, list) else [gt]
            conf_list = confidences if confidences else [None] * len(outputs_list)
            audio_ids_list = (audio_ids if isinstance(audio_ids, list) else [audio_ids] if audio_ids is not None else [])
            audio_ids_list = (audio_ids_list + [None] * len(outputs_list))[:len(outputs_list)]
            for o, g, conf, aid in zip(outputs_list, gt_list, conf_list, audio_ids_list):
                record = {"gt": g, "output": o}
                if aid is not None:
                    record["audio_id"] = aid
                if conf:
                    record.update(conf)
                self._predictions_file.write(json.dumps(record) + "\n")
            
        if correct is not None and total is not None:
            if not hasattr(self, 'token_correct'): self.token_correct = 0
            if not hasattr(self, 'token_total'): self.token_total = 0
            self.token_correct += correct
            self.token_total += total

        if self.epoch_type == "train":
            if kwargs.get("ce1_loss") is not None:
                self._train_ce1_losses.append(float(kwargs["ce1_loss"]))
            if kwargs.get("ce2_loss") is not None:
                self._train_ce2_losses.append(float(kwargs["ce2_loss"]))
            if kwargs.get("arcface_loss") is not None:
                self._train_arcface_losses.append(float(kwargs["arcface_loss"]))
            if kwargs.get("asv_pair_loss") is not None:
                self._train_asv_pair_losses.append(float(kwargs["asv_pair_loss"]))
            
        self.iteration_num += 1
        
        if self.epoch_type == "train" and self.iteration_num % self.log_freq == 0:
            avg_loss = float(np.mean(self.losses[-self.log_freq:])) if self.losses else 0.0
            parts = [f"Loss={avg_loss:.4f}", f"LR={lr}"]
            if self._train_ce1_losses:
                parts.append(f"CE1={float(np.mean(self._train_ce1_losses[-self.log_freq:])):.4f}")
            if self._train_ce2_losses:
                parts.append(f"CE2={float(np.mean(self._train_ce2_losses[-self.log_freq:])):.4f}")
            if self._train_arcface_losses:
                parts.append(f"ArcFace={float(np.mean(self._train_arcface_losses[-self.log_freq:])):.4f}")
            if self._train_asv_pair_losses:
                parts.append(f"ASVPair={float(np.mean(self._train_asv_pair_losses[-self.log_freq:])):.4f}")
            print(
                f"Epoch {self.epoch_num} [{self.iteration_num}] ({self.epoch_type}): "
                + " ".join(parts),
                flush=True,
            )
            
            # Also save batch metrics to file immediately (rank 0 only)
            if self.log_freq < 1e8 and (not dist.is_initialized() or dist.get_rank() == 0):
                batch_data = {
                    "epoch": self.epoch_num,
                    "iteration": self.iteration_num,
                    "type": "train_batch",
                    "loss": avg_loss,
                    "lr": lr
                }
                if self._train_ce1_losses:
                    batch_data["loss_ce1"] = float(np.mean(self._train_ce1_losses[-self.log_freq:]))
                if self._train_ce2_losses:
                    batch_data["loss_ce2"] = float(np.mean(self._train_ce2_losses[-self.log_freq:]))
                if self._train_arcface_losses:
                    batch_data["loss_arcface"] = float(np.mean(self._train_arcface_losses[-self.log_freq:]))
                if self._train_asv_pair_losses:
                    batch_data["loss_asv_pair"] = float(np.mean(self._train_asv_pair_losses[-self.log_freq:]))
                self._save_json(batch_data, "metrics.jsonl")

    def _extract_answer(self, text: str) -> str:
        """Extract answer from hard label format (supports both SASV and antispoofing)."""
        from ..analysis.plotter_common import extract_answer
        return extract_answer(text)

    @staticmethod
    def _compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple:
        """Compute Equal Error Rate from binary labels and continuous scores.

        Args:
            labels: binary array (1 = positive class, 0 = negative class)
            scores: continuous scores (higher → more likely positive)

        Returns:
            (eer, threshold) or (None, None) if computation fails.
        """
        if len(labels) < 2 or len(np.unique(labels)) < 2:
            return None, None
        labels = np.asarray(labels, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        if pos_scores.size == 0 or neg_scores.size == 0:
            return None, None
        from ..epochs.utils.sasv_metrics import compute_det_curve

        frr, far, thresholds = compute_det_curve(pos_scores, neg_scores)
        idx = int(np.nanargmin(np.abs(frr - far)))
        eer = float((frr[idx] + far[idx]) / 2)
        threshold = float(thresholds[idx])
        return eer, threshold

    def _gather_distributed_epoch_state(self) -> bool:
        """Gather per-rank epoch lists onto rank 0 for validation/test metrics."""
        if not dist.is_initialized():
            return True

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        list_attrs = [
            "losses",
            "lrs",
            "outputs",
            "gts",
            "audio_ids",
            "output_sources",
            "all_confidences",
            "correct_confidences",
            "incorrect_confidences",
            "real_confidences",
            "fake_confidences",
            "yes_confidences",
            "no_confidences",
            "gen_confidences",
            "tp_confidences",
            "fp_confidences",
            "tn_confidences",
            "fn_confidences",
            "pred_fake_correct_confidences",
            "pred_fake_incorrect_confidences",
            "pred_real_correct_confidences",
            "pred_real_incorrect_confidences",
            "pred_yes_correct_confidences",
            "pred_yes_incorrect_confidences",
            "pred_no_correct_confidences",
            "pred_no_incorrect_confidences",
            "pred_gen_correct_confidences",
            "pred_gen_incorrect_confidences",
            "eer_labels",
            "eer_yes_probs",
            "eer_gen_probs",
            "eer_real_probs",
            "_train_ce1_losses",
            "_train_ce2_losses",
            "_train_arcface_losses",
            "_train_asv_pair_losses",
        ]
        local_state = {name: getattr(self, name, []) for name in list_attrs}
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_state)

        if rank != 0:
            return False

        for name in list_attrs:
            merged = []
            for state in gathered:
                if state:
                    merged.extend(state.get(name, []))
            setattr(self, name, merged)
        return True


    def log_epoch(self):
        should_write_metrics = self._gather_distributed_epoch_state()

        # Close predictions file if open. Keep per-rank prediction files, but only rank 0 writes metrics.
        if self._predictions_file:
            self._predictions_file.close()
            self._predictions_file = None
            rank_suffix = f"_rank{dist.get_rank()}" if dist.is_initialized() else ""
            print(
                f"All predictions saved to: {self.log_dir}/"
                f"predictions_{self.epoch_type}_epoch_{self.epoch_num}{rank_suffix}.jsonl",
                flush=True,
            )

        if not should_write_metrics:
            return

        # Convert to float and handle NaNs
        clean_losses = [l for l in self.losses if not np.isnan(l)]
        avg_loss = float(np.mean(clean_losses)) if clean_losses else 0.0
        
        metrics = {
            "epoch": self.epoch_num,
            "type": self.epoch_type,
            "loss": avg_loss
        }

        if self.epoch_type == "train":
            if self._train_ce1_losses:
                metrics["loss_ce1"] = float(np.mean(self._train_ce1_losses))
            if self._train_ce2_losses:
                metrics["loss_ce2"] = float(np.mean(self._train_ce2_losses))
            if self._train_arcface_losses:
                metrics["loss_arcface"] = float(np.mean(self._train_arcface_losses))
            if self._train_asv_pair_losses:
                metrics["loss_asv_pair"] = float(np.mean(self._train_asv_pair_losses))
        
        if self.epoch_type in ["validation", "test"]:
            if self.output_sources:
                unique_sources = sorted(set(self.output_sources))
                if len(unique_sources) == 1:
                    metrics["output_source"] = unique_sources[0]
                else:
                    metrics["output_source"] = "mixed"
                    metrics["output_source_counts"] = {
                        source: self.output_sources.count(source) for source in unique_sources
                    }

            if self.outputs and self.gts:
                # Extract answers before comparison
                extracted_outputs = [self._extract_answer(o) for o in self.outputs]
                extracted_gts = [self._extract_answer(g) for g in self.gts]
                
                # Text-based accuracy (from generate or extracted tags)
                correct_list = [o.lower() == g.lower() for o, g in zip(extracted_outputs, extracted_gts)]
                acc = sum(correct_list) / len(correct_list)
                metrics["accuracy"] = acc
                self.last_accuracy = acc
                
                # Per-class accuracy
                # Detect task type from extracted answers
                all_gts_lower = [g.lower() for g in extracted_gts[:len(correct_list)]]
                is_sasv = any(g in ["yes", "no", "gen"] for g in all_gts_lower)
                
                if is_sasv:
                    # SASV format: three classes
                    yes_indices = [i for i, g in enumerate(all_gts_lower) if g == "yes"]
                    no_indices = [i for i, g in enumerate(all_gts_lower) if g == "no"]
                    gen_indices = [i for i, g in enumerate(all_gts_lower) if g == "gen"]
                    
                    if yes_indices:
                        yes_acc = sum([correct_list[i] for i in yes_indices]) / len(yes_indices)
                        metrics["accuracy_yes"] = yes_acc
                    if no_indices:
                        no_acc = sum([correct_list[i] for i in no_indices]) / len(no_indices)
                        metrics["accuracy_no"] = no_acc
                    if gen_indices:
                        gen_acc = sum([correct_list[i] for i in gen_indices]) / len(gen_indices)
                        metrics["accuracy_gen"] = gen_acc
                    
                    # Balanced accuracy (average of all three classes)
                    accs = []
                    if yes_indices:
                        accs.append(yes_acc)
                    if no_indices:
                        accs.append(no_acc)
                    if gen_indices:
                        accs.append(gen_acc)
                    if accs:
                        metrics["accuracy_balanced"] = sum(accs) / len(accs)
                else:
                    # Antispoofing format: two classes
                    real_indices = [i for i, g in enumerate(all_gts_lower) if g == "real"]
                    fake_indices = [i for i, g in enumerate(all_gts_lower) if g == "fake"]
                    
                    if real_indices:
                        real_acc = sum([correct_list[i] for i in real_indices]) / len(real_indices)
                        metrics["accuracy_real"] = real_acc
                    
                    if fake_indices:
                        fake_acc = sum([correct_list[i] for i in fake_indices]) / len(fake_indices)
                        metrics["accuracy_fake"] = fake_acc
                        
                    if real_indices and fake_indices:
                        metrics["accuracy_balanced"] = (real_acc + fake_acc) / 2
            
            # Token-level accuracy from forward (if text-based not possible for all)
            if hasattr(self, 'token_correct') and self.token_total > 0:
                metrics["token_accuracy"] = float(self.token_correct / self.token_total)
                # Reset for next epoch
                self.token_correct = 0
                self.token_total = 0
            
            # Confidence metrics
            if self.all_confidences:
                metrics["confidence_mean"] = float(np.mean(self.all_confidences))
                metrics["confidence_std"] = float(np.std(self.all_confidences))
                if self.correct_confidences:
                    metrics["confidence_correct_mean"] = float(np.mean(self.correct_confidences))
                if self.incorrect_confidences:
                    metrics["confidence_incorrect_mean"] = float(np.mean(self.incorrect_confidences))
                
                # Detect task type from confidence stats
                is_sasv = len(self.yes_confidences) > 0 or len(self.no_confidences) > 0 or len(self.gen_confidences) > 0
                
                if is_sasv:
                    # SASV format: three classes
                    if self.yes_confidences:
                        metrics["confidence_yes_mean"] = float(np.mean(self.yes_confidences))
                    if self.no_confidences:
                        metrics["confidence_no_mean"] = float(np.mean(self.no_confidences))
                    if self.gen_confidences:
                        metrics["confidence_gen_mean"] = float(np.mean(self.gen_confidences))
                    
                    # Confidence by predicted class and correctness
                    if self.pred_yes_correct_confidences:
                        metrics["confidence_pred_yes_correct_mean"] = float(np.mean(self.pred_yes_correct_confidences))
                    if self.pred_yes_incorrect_confidences:
                        metrics["confidence_pred_yes_incorrect_mean"] = float(np.mean(self.pred_yes_incorrect_confidences))
                    if self.pred_no_correct_confidences:
                        metrics["confidence_pred_no_correct_mean"] = float(np.mean(self.pred_no_correct_confidences))
                    if self.pred_no_incorrect_confidences:
                        metrics["confidence_pred_no_incorrect_mean"] = float(np.mean(self.pred_no_incorrect_confidences))
                    if self.pred_gen_correct_confidences:
                        metrics["confidence_pred_gen_correct_mean"] = float(np.mean(self.pred_gen_correct_confidences))
                    if self.pred_gen_incorrect_confidences:
                        metrics["confidence_pred_gen_incorrect_mean"] = float(np.mean(self.pred_gen_incorrect_confidences))
                else:
                    # Antispoofing format: two classes
                    if self.real_confidences:
                        metrics["confidence_real_mean"] = float(np.mean(self.real_confidences))
                    if self.fake_confidences:
                        metrics["confidence_fake_mean"] = float(np.mean(self.fake_confidences))
                    
                    # TP/FP/TN/FN confidence metrics
                    if self.tp_confidences:
                        metrics["confidence_tp_mean"] = float(np.mean(self.tp_confidences))
                        metrics["confidence_tp_count"] = len(self.tp_confidences)
                    if self.fp_confidences:
                        metrics["confidence_fp_mean"] = float(np.mean(self.fp_confidences))
                        metrics["confidence_fp_count"] = len(self.fp_confidences)
                    if self.tn_confidences:
                        metrics["confidence_tn_mean"] = float(np.mean(self.tn_confidences))
                        metrics["confidence_tn_count"] = len(self.tn_confidences)
                    if self.fn_confidences:
                        metrics["confidence_fn_mean"] = float(np.mean(self.fn_confidences))
                        metrics["confidence_fn_count"] = len(self.fn_confidences)
                    
                    # Confidence by predicted class and correctness
                    if self.pred_fake_correct_confidences:
                        metrics["confidence_pred_fake_correct_mean"] = float(np.mean(self.pred_fake_correct_confidences))
                    if self.pred_fake_incorrect_confidences:
                        metrics["confidence_pred_fake_incorrect_mean"] = float(np.mean(self.pred_fake_incorrect_confidences))
                    if self.pred_real_correct_confidences:
                        metrics["confidence_pred_real_correct_mean"] = float(np.mean(self.pred_real_correct_confidences))
                    if self.pred_real_incorrect_confidences:
                        metrics["confidence_pred_real_incorrect_mean"] = float(np.mean(self.pred_real_incorrect_confidences))

            if self.eer_labels:
                labels_arr = np.array(self.eer_labels)
                is_sasv_eer = any(l in ("yes", "no", "gen") for l in self.eer_labels)

                if is_sasv_eer and self.eer_yes_probs:
                    yes_probs = np.array(self.eer_yes_probs)
                    gen_probs = np.array(self.eer_gen_probs)

                    # SASV-EER: "yes" (target bonafide) vs "no"+"gen"
                    sasv_binary = (labels_arr == "yes").astype(np.int32)
                    eer_sasv, thr_sasv = self._compute_eer(sasv_binary, yes_probs)
                    if eer_sasv is not None:
                        metrics["eer_sasv"] = eer_sasv
                        metrics["eer_sasv_threshold"] = thr_sasv

                    # SV-EER: "yes" vs "no" (speaker verification only)
                    sv_mask = np.isin(labels_arr, ["yes", "no"])
                    if sv_mask.sum() >= 2 and len(np.unique(labels_arr[sv_mask])) == 2:
                        sv_binary = (labels_arr[sv_mask] == "yes").astype(np.int32)
                        eer_sv, thr_sv = self._compute_eer(sv_binary, yes_probs[sv_mask])
                        if eer_sv is not None:
                            metrics["eer_sv"] = eer_sv
                            metrics["eer_sv_threshold"] = thr_sv

                    # SPF-EER: bonafide ("yes"+"no") vs spoof ("gen")
                    spf_binary = (labels_arr != "gen").astype(np.int32)
                    spf_scores = 1.0 - gen_probs
                    eer_spf, thr_spf = self._compute_eer(spf_binary, spf_scores)
                    if eer_spf is not None:
                        metrics["eer_spf"] = eer_spf
                        metrics["eer_spf_threshold"] = thr_spf

                elif self.eer_real_probs:
                    # Antispoofing EER: "real" vs "fake"
                    real_probs = np.array(self.eer_real_probs)
                    asv_binary = (labels_arr == "real").astype(np.int32)
                    eer_asv, thr_asv = self._compute_eer(asv_binary, real_probs)
                    if eer_asv is not None:
                        metrics["eer"] = eer_asv
                        metrics["eer_threshold"] = thr_asv

            # min a-DCF / min t-DCF / t-EER× from SASVEvalEpoch or TestEpoch
            extra_sasv = getattr(self, "_sasv_metrics", None)
            if isinstance(extra_sasv, dict) and extra_sasv:
                metrics.update(extra_sasv)
                self._sasv_metrics = None

            # Save some samples
            if self.outputs and self.gts:
                os.makedirs(self.log_dir, exist_ok=True)
                samples_path = os.path.join(self.log_dir, f"samples_{self.epoch_type}_epoch_{self.epoch_num}.jsonl")
                with open(samples_path, "w") as f:
                    for i in range(min(10, len(self.outputs))):
                        sample = {"gt": self.gts[i], "output": self.outputs[i]}
                        if i < len(self.audio_ids) and self.audio_ids[i] is not None:
                            sample["audio_id"] = self.audio_ids[i]
                        f.write(json.dumps(sample) + "\n")

            sasv_detail = getattr(self, "_sasv_metrics", None)
            if isinstance(sasv_detail, dict) and sasv_detail:
                for k, v in sasv_detail.items():
                    if isinstance(v, (np.floating, np.integer)):
                        metrics[f"sasv_{k}"] = float(v)
                    elif isinstance(v, (int, float)):
                        metrics[f"sasv_{k}"] = float(v)
                self._sasv_metrics = None

            arc_trials = getattr(self, "_arcface_per_trial", None)
            if isinstance(arc_trials, list) and arc_trials:
                os.makedirs(self.log_dir, exist_ok=True)
                arc_path = os.path.join(
                    self.log_dir, f"arcface_trials_{self.epoch_type}_epoch_{self.epoch_num}.jsonl"
                )
                with open(arc_path, "w") as f:
                    for row in arc_trials:
                        f.write(json.dumps(row) + "\n")
                metrics["arcface_trials_path"] = arc_path
                metrics["arcface_trials_count"] = len(arc_trials)
                self._arcface_per_trial = None
        
        print(f"End of Epoch {self.epoch_num} ({self.epoch_type}): {metrics}", flush=True)
        self.last_epoch_metrics = metrics
        self.last_epoch_meta_path = self.resolve_epoch_meta_path()
        if self.log_freq < 1e8 and (not dist.is_initialized() or dist.get_rank() == 0):
            self._save_json(metrics, "metrics.jsonl")
