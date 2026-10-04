from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import csv
import json
import os
import torch
import math
from ..loggers.base import Logger
from ..epochs.eval_epoch import EvalEpoch

class Trainer(ABC):
    def __init__(self, config: Dict[str, Any], model: torch.nn.Module, 
                 train_loader: Any, val_loader: Any, logger: Logger,
                 device: Optional[torch.device] = None, output_dir: Optional[str] = None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.output_dir = output_dir
        
        # If device not provided, infer from model parameters
        if device is None:
            if hasattr(model, "parameters") and list(model.parameters()):
                self.device = list(model.parameters())[0].device
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.num_epochs = config.get("num_epochs", 1)
        self.start_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_epoch = None

        # Init optimizer/scheduler
        opt_cfg = config.get("optimizator", {})
        head_opt_cfg = config.get("head_optimizator")

        from ..utils.optimizer_param_groups import build_adamw_param_groups, split_trainable_params

        optim_params = build_adamw_param_groups(self.model, opt_cfg, head_opt_cfg)
        base_lr = float(opt_cfg.get("init_lr", config.get("lr", 1e-4)))
        for g in optim_params:
            g["lr"] = base_lr * float(g.get("lr_ratio_to_base", 1.0))

        if head_opt_cfg:
            _, _, hw, hnw = split_trainable_params(self.model)
            n_head = sum(p.numel() for p in hw) + sum(p.numel() for p in hnw)
            if n_head > 0:
                h_lr = float(head_opt_cfg.get("init_lr", base_lr))
                print(
                    f"[Optimizer] head_optimizator: {n_head} params, init_lr={h_lr} "
                    f"(base init_lr={base_lr}, ratio={h_lr / base_lr:.3f})",
                    flush=True,
                )

        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=base_lr,
            betas=(0.9, opt_cfg.get("beta2", 0.999)),
        )
        
        from ..utils.optims import LinearWarmupCosineLRScheduler
        train_iters_per_epoch = config.get("iters_per_epoch") or len(train_loader)
        accum_grad_iters = max(1, int(config.get("accum_grad_iters", 1)))
        optimizer_steps_per_epoch = max(1, math.ceil(train_iters_per_epoch / accum_grad_iters))

        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.num_epochs,
            iters_per_epoch=optimizer_steps_per_epoch,
            min_lr=opt_cfg.get("min_lr", 1e-6),
            init_lr=opt_cfg.get("init_lr", config.get("lr", 1e-4)),
            warmup_steps=opt_cfg.get("warmup_steps", 0),
            warmup_start_lr=opt_cfg.get("warmup_start_lr", -1)
        )
        # Start at warmup LR immediately, before the first optimizer step/log.
        self.scheduler.step(0, 0)
        
        # AMP settings
        self.amp = config.get("amp", True)
        
        # Best checkpoint tracking (by accuracy, higher is better)
        self.best_val_accuracy = 0.0
        self.best_epoch = None
        
        # Grad Scaler for AMP
        self.scaler = None
        self.max_grad_norm = opt_cfg.get("max_grad_norm", 1.0)
        if config.get("amp", True) and self.device.type == "cuda" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                print("Using bfloat16 (no scaler needed)", flush=True)
                self.scaler = None
            else:
                print("Using float16 with GradScaler", flush=True)
                # If model is already in float16, this might fail with "Attempting to unscale FP16 gradients"
                self.scaler = torch.cuda.amp.GradScaler()

    @abstractmethod
    def train(self):
        pass

    def validate(self, epoch_num: int) -> float:
        gen_cfg = self.config.get("generation", {})
        eval_epoch = EvalEpoch(self.model, self.val_loader, self.logger, device=self.device, gen_cfg=gen_cfg if gen_cfg else None)
        eval_epoch.run(epoch_num=epoch_num)
        accuracy = getattr(self.logger, 'last_accuracy', 0.0)
        return accuracy

    def save_checkpoint(self, epoch_num: int, val_accuracy: float, name: str = "checkpoint"):
        if self.output_dir is None:
            return
            
        import os
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        
        # if torch.distributed.is_initialized():
        #     if torch.distributed.get_rank() != 0:
        #         return

        is_dist = torch.distributed.is_initialized()
        is_main = (not is_dist) or (torch.distributed.get_rank() == 0)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        is_best = val_accuracy > self.best_val_accuracy

        if is_main:
            os.makedirs(ckpt_dir, exist_ok=True)
            if is_best:
                if self.best_epoch is not None and self.best_epoch != epoch_num: # Only delete if it's a different epoch
                    old_best = os.path.join(ckpt_dir, f"{name}_epoch_{self.best_epoch}_best.pt")
                    if os.path.exists(old_best):
                        os.remove(old_best)
                        print(f"Deleted old best checkpoint: {old_best}", flush=True)

                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch_num

                save_path = os.path.join(ckpt_dir, f"{name}_epoch_{epoch_num}_best.pt")
                torch.save({
                    'epoch': epoch_num,
                    'model': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'config': self.config,
                }, save_path)
                print(f"Best checkpoint saved to {save_path} (accuracy={val_accuracy:.4f})", flush=True)
                self._write_dev_summary_csv(save_path, val_accuracy)

            # Always save latest (overwrites each epoch) on main rank 
            latest_path = os.path.join(ckpt_dir, f"{name}_latest.pt")
            torch.save({
                'epoch': epoch_num,
                'model': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'config': self.config,
            }, latest_path)
            print(f"Latest checkpoint saved to {latest_path}", flush=True)
        
        if is_dist:
            torch.distributed.barrier()

    def _write_dev_summary_csv(self, best_ckpt_path: str, val_accuracy: float) -> None:
        """Upsert train/dev best-checkpoint summary into outputs/checkpoint_dev_summary.csv."""
        if self.output_dir is None:
            return

        outputs_root = self._outputs_root_dir()
        os.makedirs(outputs_root, exist_ok=True)
        csv_path = os.path.join(outputs_root, "checkpoint_dev_summary.csv")

        row = self._dev_summary_identity(best_ckpt_path)
        metrics = getattr(self.logger, "last_epoch_metrics", None)
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                row[key] = self._csv_metric_value(value)
        row.setdefault("accuracy", float(val_accuracy))

        id_fields = ["experiment", "run", "best_ckpt_path", "meta_path"]
        metric_fields = [
            "accuracy",
            "accuracy_balanced",
            "accuracy_yes",
            "accuracy_no",
            "accuracy_gen",
            "t_eer",
            "min_a_dcf",
            "min_t_dcf",
        ]
        fieldnames = id_fields + metric_fields

        rows = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))

        # Keep one row per run, replacing previous best of the same run.
        key = (row["experiment"], row["run"])
        updated = False
        for i, existing in enumerate(rows):
            if (existing.get("experiment", ""), existing.get("run", "")) == key:
                rows[i] = {**existing, **row}
                updated = True
                break
        if not updated:
            rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"Checkpoint dev summary saved to {csv_path}", flush=True)

    def _outputs_root_dir(self) -> str:
        """Resolve shared outputs root from run output dir."""
        run_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        parts = os.path.normpath(run_dir).split(os.sep)
        for i, part in enumerate(parts):
            if part == "outputs":
                return os.sep.join(parts[: i + 1]) or os.sep
        return os.path.dirname(run_dir)

    def _dev_summary_identity(self, best_ckpt_path: str) -> Dict[str, str]:
        """Infer experiment/run identifiers from run output dir."""
        run_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        run_name = os.path.basename(run_dir)
        exp_name = os.path.basename(os.path.dirname(run_dir))
        if exp_name == "outputs":
            exp_name = self.config.get("model_name", "default")
        return {
            "experiment": exp_name,
            "run": run_name,
            "best_ckpt_path": os.path.abspath(os.path.expanduser(best_ckpt_path)),
            "meta_path": self._resolve_meta_path(),
        }

    def _resolve_meta_path(self) -> str:
        """Resolve dataset path used for validation metrics."""
        meta_path = self.config.get("meta_path", "")
        if isinstance(meta_path, str) and meta_path:
            return os.path.abspath(os.path.expanduser(meta_path))
        for cfg_key in ("dataset_val_path", "dataset_dev_path"):
            value = self.config.get(cfg_key, "")
            if isinstance(value, str) and value:
                return os.path.abspath(os.path.expanduser(value))

        dataset = getattr(self.val_loader, "dataset", None)
        for attr_name in ("meta_path", "dataset_path", "path", "file_path", "json_path"):
            value = getattr(dataset, attr_name, "")
            if isinstance(value, str) and value:
                return os.path.abspath(os.path.expanduser(value))
        return ""

    @staticmethod
    def _csv_metric_value(value: Any) -> Any:
        if isinstance(value, (str, int, float)) or value is None:
            return value
        return json.dumps(value, sort_keys=True)

    def load_checkpoint(self, resume_path: str):
        import os
        import torch
        if not os.path.exists(resume_path):
            print(f"Warning: Resume path {resume_path} does not exist. Starting training from scratch.")
            return

        print(f"Loading checkpoint from {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location="cpu")

        # Load model state dict
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(checkpoint['model'], strict=False)
        print("Model state dict loaded.", flush=True)

        # Load optimizer state dict
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state dict loaded.", flush=True)
        else:
            print("Warning: Optimizer state dict not found in checkpoint or optimizer not initialized. Optimizer will restart from scratch.", flush=True)

        # Load scaler state dict if AMP is used
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("AMP GradScaler state dict loaded.", flush=True)
        else:
            print("Warning: AMP GradScaler state dict not found in checkpoint or scaler not initialized. Scaler will restart from scratch.", flush=True)

        # Resume epoch, best accuracy, and best epoch index
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_accuracy = checkpoint.get('val_accuracy', 0.0)
        self.best_epoch = checkpoint.get('best_epoch')
        print(f"Resuming from epoch {self.start_epoch} with best validation accuracy {self.best_val_accuracy:.4f}", flush=True)

        # Update scheduler's last_epoch to ensure correct LR scheduling
        if hasattr(self.scheduler, 'last_epoch'):
            self.scheduler.last_epoch = self.start_epoch - 1
            print(f"Scheduler last_epoch set to {self.scheduler.last_epoch}", flush=True)





