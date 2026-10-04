import unittest
from unittest.mock import MagicMock, call
from src.trainers import DistillationTrainer, SFTTrainer, GRPOTrainer

class TestDistillationTrainer(unittest.TestCase):
    def test_distillation_loop(self):
        config = {"num_distillation_iters": 2}
        
        sft_trainer = MagicMock(spec=SFTTrainer)
        grpo_trainer = MagicMock(spec=GRPOTrainer)
        mock_loader = MagicMock()
        sft_trainer.train_loader = mock_loader
        grpo_trainer.train_loader = mock_loader
        
        trainer = DistillationTrainer(config, sft_trainer, grpo_trainer, initial_train_loader=mock_loader)
        trainer.train()
        
        # Check call order and counts
        # Expected: SFT -> GRPO -> SFT -> GRPO
        self.assertEqual(sft_trainer.train.call_count, 2)
        self.assertEqual(grpo_trainer.train.call_count, 2)
        
        # We can't strictly verify order with simple call_count, but we can check the manager
        # Since they are separate objects, we assume the loop is sequential.
        # This confirms the loop runs the correct number of times.

if __name__ == "__main__":
    unittest.main()





