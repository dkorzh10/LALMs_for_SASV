import unittest
from unittest.mock import MagicMock
from src.trainers import GRPOTrainer
from src.models.dummy.model import DummyModel
from src.loggers.hard_label_logger import HardLabelLogger
from src.judges.format_judge import FormatJudge

class TestGRPOTrainer(unittest.TestCase):
    def test_train_loop(self):
        config = {"num_epochs": 1, "lr": 0.001}
        model = DummyModel({})
        logger = MagicMock(spec=HardLabelLogger)
        judge = FormatJudge()
        train_loader = [{"audio_id": "1", "raw_wav": "dummy", "text": "in", "reasoning": "gt"}]
        val_loader = []
        
        trainer = GRPOTrainer(config, model, train_loader, val_loader, logger, judge)
        trainer.train()
        
        self.assertTrue(logger.log_epoch.called)

if __name__ == "__main__":
    unittest.main()

