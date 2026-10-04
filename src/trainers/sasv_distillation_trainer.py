"""
SASV Distillation trainer: find model errors → SFT on error subset.
No reasoning, no GRPO, no judge.
"""
from typing import Any, List, Optional

from torch.utils.data import DataLoader, Subset

from .sft_trainer import SFTTrainer


def _get_base_dataset(dataloader: DataLoader):
    ds = dataloader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def _create_subset_loader(
    base_loader: DataLoader,
    indices: List[int],
) -> DataLoader:
    base_ds = _get_base_dataset(base_loader)
    subset = Subset(base_ds, indices)
    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=True,
        collate_fn=base_loader.collate_fn,
        num_workers=base_loader.num_workers,
        pin_memory=getattr(base_loader, "pin_memory", False),
    )


def create_forming_loader(base_loader: DataLoader, batch_size: int) -> DataLoader:
    """Create a DataLoader for dataset forming with a custom batch_size."""
    return DataLoader(
        base_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=base_loader.collate_fn,
        num_workers=base_loader.num_workers,
        pin_memory=getattr(base_loader, "pin_memory", False),
    )


class SASVDistillationTrainer:
    """
    Orchestrates SASV distillation cycles:
      1. Run inference, collect error indices (dataset forming)
      2. SFT on error subset
    Repeats for num_distillation_iters.
    """

    def __init__(
        self,
        config: dict,
        sft_trainer: SFTTrainer,
        dataset_forming_epoch: Any,
        initial_train_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.sft_trainer = sft_trainer
        self.dataset_forming_epoch = dataset_forming_epoch
        self.initial_train_loader = initial_train_loader or sft_trainer.train_loader
        self.num_iters = config.get("num_distillation_iters", 1)

    def train(self):
        for i in range(self.num_iters):
            print(f"\n{'='*60}", flush=True)
            print(f"SASV Distillation Iteration {i}", flush=True)
            print(f"{'='*60}", flush=True)

            # 1. Form error dataset
            error_indices = self.dataset_forming_epoch.run(distillation_iter=i)

            if not error_indices:
                print("  No errors found; model is correct on all samples. Stopping.", flush=True)
                break

            print(f"  Forming SFT loader from {len(error_indices)} error samples...", flush=True)
            sft_loader = _create_subset_loader(self.initial_train_loader, error_indices)
            self.sft_trainer.train_loader = sft_loader

            # 2. SFT on error subset
            logger = getattr(self.sft_trainer, "logger", None)
            if logger and hasattr(logger, "set_distillation_iter"):
                logger.set_distillation_iter(i, "sft")
            self.sft_trainer.train()

            print(f"  SASV Distillation iteration {i} complete.", flush=True)
