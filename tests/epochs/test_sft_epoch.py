import unittest
from unittest.mock import MagicMock
import torch
from src.epochs.sft_epoch import SFTTrainEpoch
from src.models.dummy.model import DummyModel
from src.loggers.hard_label_logger import HardLabelLogger

class TestSFTTrainEpoch(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel({})
        self.logger = MagicMock(spec=HardLabelLogger)
        self.optimizer = MagicMock(spec=torch.optim.Optimizer)
        self.optimizer.param_groups = [{'lr': 0.001}]
        # Mock dataloader
        self.dataloader = [
            {"audio_id": "1", "raw_wav": torch.randn(1, 16000), "text": "target"}
        ]
        
    def test_run(self):
        epoch = SFTTrainEpoch(self.model, self.dataloader, self.logger, self.optimizer)
        epoch.run(0)
        
        self.logger.set_epoch.assert_called_with(0, "train")
        self.assertTrue(self.logger.log.called)
        self.assertTrue(self.logger.log_epoch.called)

if __name__ == "__main__":
    unittest.main()

