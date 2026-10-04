from .base import Trainer
from ..epochs.sft_epoch import SFTTrainEpoch


class SFTTrainer(Trainer):
    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Starting Epoch {epoch}", flush=True)
            train_epoch = SFTTrainEpoch(
                self.model, self.train_loader, self.logger,
                self.optimizer, self.scheduler, self.scaler,
                device=self.device,
                amp=self.amp,
                iters_per_epoch=self.config.get("iters_per_epoch"),
                accum_grad_iters=self.config.get("accum_grad_iters", 1),
                max_grad_norm=self.max_grad_norm
            )
            train_epoch.run(epoch_num=epoch)

            val_accuracy = 0.0
            if self.val_loader:
                val_accuracy = self.validate(epoch)

            self.save_checkpoint(epoch, val_accuracy, name="sft")
