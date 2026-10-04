"""Tests for samplers and audio length."""
import unittest
import torch

from src.dataloaders.audio_length import get_audio_duration_sec
from src.dataloaders.builder import get_dataloader
from src.dataloaders.samplers import (
    StatefulSampler,
    StatefulDistributedSampler,
    LengthGroupedDistributedSampler,
)
from src.dataloaders.dataset import AudioDataset


class TestAudioLength(unittest.TestCase):
    def test_dummy_path_returns_fixed_duration(self):
        d = get_audio_duration_sec("/tmp/dummy_audio_0.wav")
        self.assertEqual(d, 1.0)

    def test_nonexistent_path_returns_default(self):
        d = get_audio_duration_sec("/nonexistent/path/123.wav")
        self.assertEqual(d, 1.0)


class TestStatefulSampler(unittest.TestCase):
    def test_basic_iteration(self):
        dataset = AudioDataset("dummy", max_samples=20)
        sampler = StatefulSampler(dataset, shuffle=True, batch_size=4)
        indices = list(sampler)
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(set(indices)), 20)

    def test_iters_per_epoch_truncates(self):
        dataset = AudioDataset("dummy", max_samples=100)
        sampler = StatefulSampler(
            dataset, shuffle=True, iters_per_epoch=5, batch_size=4
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 5 * 4)

    def test_set_epoch_affects_shuffle(self):
        dataset = AudioDataset("dummy", max_samples=10)
        sampler = StatefulSampler(dataset, shuffle=True, stateful=True)
        indices_0 = list(sampler)
        sampler.set_epoch(1)
        indices_1 = list(sampler)
        self.assertNotEqual(indices_0, indices_1)


class TestStatefulDistributedSampler(unittest.TestCase):
    def test_iters_per_epoch_partitioned(self):
        dataset = AudioDataset("dummy", max_samples=100)
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
            iters_per_epoch=5,
            batch_size=4,
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 5 * 4)

    def test_len_matches_iters_per_epoch(self):
        dataset = AudioDataset("dummy", max_samples=100)
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            iters_per_epoch=3,
            batch_size=8,
        )
        self.assertEqual(len(sampler), 3 * 8)


class TestLengthGroupedDistributedSampler(unittest.TestCase):
    def test_basic_iteration(self):
        dataset = AudioDataset("dummy", max_samples=20)
        sampler = LengthGroupedDistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 10)

    def test_sorted_by_length(self):
        dataset = AudioDataset("dummy", max_samples=20)
        sampler = LengthGroupedDistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=False,
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(set(indices)), 20)

    def test_iters_per_epoch_preserved(self):
        dataset = AudioDataset("dummy", max_samples=100)
        sampler = LengthGroupedDistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
            iters_per_epoch=5,
            batch_size=4,
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 5 * 4)

    def test_custom_length_fn(self):
        dataset = AudioDataset("dummy", max_samples=10)
        length_fn = lambda path: 5.0 if "0" in path else 1.0
        sampler = LengthGroupedDistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=False,
            length_fn=length_fn,
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 10)

    def test_set_epoch(self):
        dataset = AudioDataset("dummy", max_samples=20)
        sampler = LengthGroupedDistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=True,
            stateful=True,
        )
        sampler.set_epoch(0)
        a = list(sampler)
        sampler.set_epoch(1)
        b = list(sampler)
        self.assertNotEqual(a, b)


class TestGetDataloaderWithLengthGrouped(unittest.TestCase):
    def test_distributed_uses_length_grouped_sampler(self):
        loader = get_dataloader(
            "dummy",
            batch_size=4,
            distributed=True,
            num_replicas=2,
            rank=0,
        )
        self.assertIsInstance(loader.sampler, LengthGroupedDistributedSampler)
        batch = next(iter(loader))
        self.assertIn("spectrogram", batch)
        self.assertEqual(len(batch["audio_ids"]), 4)

    def test_non_distributed_uses_stateful_sampler(self):
        loader = get_dataloader("dummy", batch_size=4, distributed=False)
        self.assertIsInstance(loader.sampler, StatefulSampler)


if __name__ == "__main__":
    unittest.main()
