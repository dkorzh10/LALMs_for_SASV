"""SASV Trainer: SFT training with ArcFace loss and SASV-specific evaluation."""

from .base import Trainer
from ..epochs.sasv_sft_epoch import SASVSFTTrainEpoch
from ..epochs.sasv_eval_epoch import SASVEvalEpoch


class SASVTrainer(Trainer):
    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Starting SASV Epoch {epoch}", flush=True)
            train_epoch = SASVSFTTrainEpoch(
                self.model, self.train_loader, self.logger,
                self.optimizer, self.scheduler, self.scaler,
                device=self.device,
                amp=self.amp,
                iters_per_epoch=self.config.get("iters_per_epoch"),
                accum_grad_iters=self.config.get("accum_grad_iters", 1),
                max_grad_norm=self.max_grad_norm,
            )
            train_epoch.run(epoch_num=epoch)

            val_accuracy = 0.0
            if self.val_loader:
                val_accuracy = self.validate(epoch)

            self.save_checkpoint(epoch, val_accuracy, name="sasv")

    def validate(self, epoch_num: int) -> float:
        """Override to use SASVEvalEpoch with t-EER/min a-DCF."""
        gen_cfg = self.config.get("generation", {})
        if not gen_cfg:
            gen_cfg = {"max_new_tokens": 1, "num_beams": 1, "do_sample": False}
        decision_backend = self.config.get("decision_backend", "llm_only")
        threshold_mode = self.config.get("threshold_mode", "fixed")
        tau_sv = self.config.get("tau_sv")
        tau_spf = self.config.get("tau_spf")
        threshold_objective = self.config.get("threshold_objective", "min_a_dcf")
        extract_confidence = self.config.get("extract_confidence", True)

        eval_epoch = SASVEvalEpoch(
            self.model, self.val_loader, self.logger,
            device=self.device, gen_cfg=gen_cfg,
            decision_backend=decision_backend,
            threshold_mode=threshold_mode,
            tau_sv=tau_sv,
            tau_spf=tau_spf,
            threshold_objective=threshold_objective,
            extract_confidence=extract_confidence,
        )
        eval_epoch.run(epoch_num=epoch_num)
        accuracy = getattr(self.logger, 'last_accuracy', 0.0)
        return accuracy
