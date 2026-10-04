import unittest
from unittest.mock import MagicMock
from src.trainers import SFTTrainer
from src.models.dummy.model import DummyModel
from src.loggers.hard_label_logger import HardLabelLogger

class TestSFTTrainer(unittest.TestCase):
    def test_train_loop(self):
        config = {"num_epochs": 1, "lr": 0.001}
        model = DummyModel({})
        logger = MagicMock(spec=HardLabelLogger)
        train_loader = [{"audio_id": "1", "raw_wav": "dummy"}]
        val_loader = []
        
        trainer = SFTTrainer(config, model, train_loader, val_loader, logger)
        trainer.train()
        
        # Verify epoch creation and run (implicit via logger calls)
        self.assertTrue(logger.log_epoch.called)

if __name__ == "__main__":
    unittest.main()

