import unittest
import os
import shutil
from src.loggers.hard_label_logger import HardLabelLogger

class TestHardLabelLogger(unittest.TestCase):
    def setUp(self):
        self.log_dir = "/tmp/test_logger"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
            
    def test_logging(self):
        logger = HardLabelLogger(self.log_dir, log_freq=1)
        logger.set_epoch(0, "train")
        
        logger.log(loss=0.5, lr=0.001)
        self.assertEqual(len(logger.losses), 1)
        
        logger.set_epoch(0, "validation")
        logger.log(outputs=[0, 1], gt=[0, 1])
        logger.log_epoch()
        
        self.assertTrue(os.path.exists(os.path.join(self.log_dir, "metrics.jsonl")))

if __name__ == "__main__":
    unittest.main()





