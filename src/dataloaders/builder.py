import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional, Any, List
from functools import partial

from .dataset import AudioDataset, collate_fn
from .samplers import StatefulSampler, LengthGroupedDistributedSampler


def get_dataloader(
    dataset_path: str,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    task_type: str = "hard_label",
    whisper_path: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    distributed: bool = False,
    iters_per_epoch: Optional[int] = None,
    stateful: bool = True,
    prompt_templates: Optional[List[str]] = None,
    samples_offset: int = 0,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    silence_delay_seconds: float = 0.5,
    audio_cfg: Optional[dict] = None,
    is_train: bool = True,
    split: Optional[str] = None,
    reasoning_version: str = "long",
) -> DataLoader:
    dataset = AudioDataset(
        dataset_path,
        max_samples=max_samples,
        task_type=task_type,
        samples_offset=samples_offset,
        audio_cfg=audio_cfg,
        is_train=is_train,
        split=split,
        reasoning_version=reasoning_version,
    )

    processor = None
    if model_name == "qwen_audio" and model_path:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    elif whisper_path:
        try:
            from transformers import WhisperFeatureExtractor
            processor = WhisperFeatureExtractor.from_pretrained(
                whisper_path,
                local_files_only=True
            )
        except Exception as e:
            print(f"Warning: Could not load WhisperFeatureExtractor from {whisper_path}: {e}", flush=True)
            processor = None

    custom_collate = partial(
        collate_fn,
        processor=processor,
        model_name=model_name,
        prompt_templates=prompt_templates,
        silence_delay_seconds=silence_delay_seconds,
        deterministic_prompts=not dataset.is_train,
    )

    if distributed:
        sampler = LengthGroupedDistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            iters_per_epoch=iters_per_epoch,
            batch_size=batch_size,
            stateful=stateful
        )
        shuffle = False
    else:
        sampler = StatefulSampler(
            dataset,
            shuffle=shuffle,
            iters_per_epoch=iters_per_epoch,
            batch_size=batch_size,
            stateful=stateful
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate,
        sampler=sampler
    )


def get_subset_dataloader(
    source_loader: DataLoader,
    indices: List[int],
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    distributed: bool = False,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader over a subset of the dataset backing *source_loader*.

    Re-uses the same collate_fn so batch format is identical.
    """
    base_dataset = source_loader.dataset
    subset = Subset(base_dataset, indices)

    bs = batch_size or source_loader.batch_size
    nw = num_workers if num_workers is not None else source_loader.num_workers

    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            subset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        loader_shuffle = False
    else:
        sampler = None
        loader_shuffle = shuffle

    return DataLoader(
        subset,
        batch_size=bs,
        shuffle=loader_shuffle,
        num_workers=nw,
        collate_fn=source_loader.collate_fn,
        sampler=sampler,
    )


def get_scoring_dataloader(
    source_loader: DataLoader,
    batch_size: int = 8,
    num_workers: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> DataLoader:
    """Build a non-shuffled DataLoader for offline scoring (hard mining).

    Uses the same dataset and collate_fn as *source_loader*, but with a
    (potentially larger) batch size and no sampler shuffling.
    """
    nw = num_workers if num_workers is not None else source_loader.num_workers
    dataset = source_loader.dataset if indices is None else Subset(source_loader.dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=source_loader.collate_fn,
    )
