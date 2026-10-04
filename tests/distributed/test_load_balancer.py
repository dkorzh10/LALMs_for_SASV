"""Tests for distributed load balancer and work items."""
import unittest
import torch

from src.distributed import (
    WorkItem,
    serialize_work_item,
    deserialize_work_item,
    GenericWorkLoadBalancer,
    SkepticLoadBalancer,
)


class TestWorkItemSerialization(unittest.TestCase):
    def test_serialize_deserialize_roundtrip(self):
        work_item = {
            "batch": {
                "audio_ids": ["test_1"],
                "text": ["Hello"],
                "prompts": torch.randn(1, 10),
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


class TestGenericWorkLoadBalancer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_leader_put_work_adds_to_pending(self):
        balancer = GenericWorkLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_leader()
        work = {"batch": {"x": 1}, "reused_rollouts": [1, 2]}
        self.assertTrue(balancer.leader_put_work(work))
        self.assertEqual(balancer.leader_pending_count(), 1)
        balancer.leader_put_work(work)
        self.assertEqual(balancer.leader_pending_count(), 2)

    def test_leader_put_work_ignored_when_loser(self):
        balancer = GenericWorkLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_loser()
        work = {"batch": {"x": 1}, "reused_rollouts": []}
        self.assertFalse(balancer.leader_put_work(work))

    def test_leader_put_work_ignored_when_all_done(self):
        balancer = GenericWorkLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_leader()
        balancer.set_all_done()
        work = {"batch": {"x": 1}, "reused_rollouts": []}
        self.assertFalse(balancer.leader_put_work(work))

    def test_exchange_round_no_dist_returns_empty(self):
        balancer = GenericWorkLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_loser()
        received, all_done = balancer.exchange_round()
        self.assertEqual(received, [])
        self.assertTrue(all_done)


class TestSkepticLoadBalancer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_leader_put_work_adds_to_pending(self):
        balancer = SkepticLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_leader()
        work = {"batch": {"x": 1}, "reused_rollouts": [1, 2]}
        self.assertTrue(balancer.leader_put_work(work))
        self.assertEqual(balancer.leader_pending_count(), 1)

    def test_exchange_round_no_dist_returns_empty(self):
        balancer = SkepticLoadBalancer(rank=0, world_size=2, device=self.device)
        balancer.set_loser()
        received, all_done = balancer.exchange_round()
        self.assertEqual(received, [])
        self.assertTrue(all_done)


if __name__ == "__main__":
    unittest.main()
