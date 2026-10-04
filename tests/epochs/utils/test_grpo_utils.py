"""Tests for grpo_utils: work item serialization, load balancer, and rollout split."""
import unittest
from unittest.mock import MagicMock

import torch

from src.epochs.utils.grpo_utils import (
    WorkItem,
    serialize_work_item,
    deserialize_work_item,
    SkepticLoadBalancer,
    chunk_ranges_for_rollouts,
    chunk_ranges_by_rollout_count,
    compute_grpo_loss_for_chunk,
)


class TestWorkItemSerialization(unittest.TestCase):
    def test_serialize_deserialize_roundtrip(self):
        work_item = {
            "batch": {
                "audio_ids": ["test_1"],
                "text": ["Hello"],
                "prompts": torch.randn(1, 10),  # dummy tensor
            },
            "reused_rollouts": {
                "texts": ["a", "b", "c"],
                "judge_meta": [{"is_correct": True}, {"is_correct": False}, {"is_correct": True}],
                "completion_ids": [
                    torch.tensor([[1, 2]], dtype=torch.long),
                    torch.tensor([[3, 4]], dtype=torch.long),
                    torch.tensor([[5, 6]], dtype=torch.long),
                ],
            },
        }
        data = serialize_work_item(work_item)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

        device = torch.device("cpu")
        restored = deserialize_work_item(data, device)
        self.assertIn("batch", restored)
        self.assertIn("reused_rollouts", restored)
        self.assertEqual(restored["batch"]["audio_ids"], work_item["batch"]["audio_ids"])
        self.assertEqual(restored["reused_rollouts"]["texts"], work_item["reused_rollouts"]["texts"])

    def test_work_item_create(self):
        batch = {"batch": "x"}
        reused = {"reused": "y"}
        item = WorkItem.create(batch, reused)
        self.assertEqual(item["batch"], batch)
        self.assertEqual(item["reused_rollouts"], reused)

    def test_work_item_is_valid(self):
        self.assertTrue(WorkItem.is_valid({"batch": 1, "reused_rollouts": 2}))
        self.assertFalse(WorkItem.is_valid({}))
        self.assertFalse(WorkItem.is_valid({"batch": 1}))
        self.assertFalse(WorkItem.is_valid({"reused_rollouts": 2}))


class TestSkepticLoadBalancer(unittest.TestCase):
    """Test load balancer logic without distributed (no dist.init)."""

    def setUp(self):
        self.device = torch.device("cpu")

    def test_leader_put_work_adds_to_pending(self):
        balancer = SkepticLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_leader()
        work = {"batch": {"x": 1}, "reused_rollouts": [1, 2]}
        self.assertTrue(balancer.leader_put_work(work))
        self.assertEqual(balancer.leader_pending_count(), 1)
        balancer.leader_put_work(work)
        self.assertEqual(balancer.leader_pending_count(), 2)

    def test_leader_put_work_ignored_when_loser(self):
        balancer = SkepticLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_loser()
        work = {"batch": {"x": 1}, "reused_rollouts": []}
        self.assertFalse(balancer.leader_put_work(work))

    def test_leader_put_work_ignored_when_all_done(self):
        balancer = SkepticLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_leader()
        balancer.set_all_done()
        work = {"batch": {"x": 1}, "reused_rollouts": []}
        self.assertFalse(balancer.leader_put_work(work))

    def test_exchange_round_no_dist_returns_empty(self):
        balancer = SkepticLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_loser()
        received, all_done = balancer.exchange_round()
        self.assertEqual(received, [])
        self.assertTrue(all_done)


class TestRolloutSplit(unittest.TestCase):
    def test_chunk_ranges_for_rollouts(self):
        self.assertEqual(
            chunk_ranges_for_rollouts(2, 4, 1),
            [(0, 2), (2, 4), (4, 6), (6, 8)],
        )
        self.assertEqual(
            chunk_ranges_for_rollouts(2, 4, 2),
            [(0, 4), (4, 8)],
        )
        self.assertEqual(
            chunk_ranges_for_rollouts(3, 6, 2),
            [(0, 6), (6, 12), (12, 18)],
        )
        self.assertEqual(
            chunk_ranges_for_rollouts(2, 4, 4),
            [(0, 8)],
        )
        self.assertEqual(
            chunk_ranges_for_rollouts(2, 4, 10),
            [(0, 8)],
        )

    def test_chunk_ranges_by_rollout_count(self):
        """True 1-rollout-at-a-time chunking."""
        self.assertEqual(
            chunk_ranges_by_rollout_count(8, 1),
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
        )
        self.assertEqual(
            chunk_ranges_by_rollout_count(8, 2),
            [(0, 2), (2, 4), (4, 6), (6, 8)],
        )
        self.assertEqual(
            chunk_ranges_by_rollout_count(16, 1),
            [(i, i + 1) for i in range(16)],
        )
        self.assertEqual(
            chunk_ranges_by_rollout_count(8, 10),
            [(0, 8)],
        )

    def test_compute_grpo_loss_for_chunk(self):
        B, T, V = 4, 8, 100
        chunk_completion_ids = torch.randint(0, V, (B, T))
        chunk_advantages = torch.randn(B)
        ref_logits = torch.randn(B, T, V)
        current_logits = torch.randn(B, T, V)
        pad_id = 0
        pg_loss, kl_div = compute_grpo_loss_for_chunk(
            chunk_completion_ids, chunk_advantages, ref_logits, current_logits, pad_id
        )
        self.assertIsInstance(pg_loss.item(), float)
        self.assertIsInstance(kl_div.item(), float)


if __name__ == "__main__":
    unittest.main()
