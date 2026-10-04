import math
import torch


def _warmup_lr_scalar(step: int, max_step: int, start_lr: float, end_lr: float) -> float:
    return min(end_lr, start_lr + (end_lr - start_lr) * step / max(max_step, 1))


def _cosine_lr_scalar(step: int, max_step: int, init_lr: float, min_lr: float) -> float:
    if max_step <= 0:
        return init_lr
    return (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * step / max_step)) + min_lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    lr = _warmup_lr_scalar(step, max_step, init_lr, max_lr)
    for param_group in optimizer.param_groups:
        mult = param_group.get("lr_ratio_to_base", 1.0)
        param_group["lr"] = lr * mult


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    lr = _cosine_lr_scalar(epoch, max_epoch, init_lr, min_lr)
    for param_group in optimizer.param_groups:
        mult = param_group.get("lr_ratio_to_base", 1.0)
        param_group["lr"] = lr * mult


class LinearWarmupCosineLRScheduler:
    def __init__(self, optimizer, max_epoch, iters_per_epoch, min_lr, init_lr, warmup_steps=0, warmup_start_lr=-1, **kwargs):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            initial_lr = param_group.setdefault("initial_lr", param_group["lr"])
            self.base_lrs.append(initial_lr)

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        max_total = max(self.max_epoch * self.iters_per_epoch, 1)
        if total_cur_step < self.warmup_steps:
            lr_base = _warmup_lr_scalar(
                total_cur_step, self.warmup_steps, self.warmup_start_lr, self.init_lr
            )
        else:
            lr_base = _cosine_lr_scalar(
                total_cur_step, max_total, self.init_lr, self.min_lr
            )
        for param_group in self.optimizer.param_groups:
            mult = param_group.get("lr_ratio_to_base", 1.0)
            param_group["lr"] = lr_base * mult
