import unittest
import os
import shutil
import json
from src.loggers.hard_label_logger import HardLabelLogger
from src.loggers.reasoning_logger import ReasoningLogger


class TestHardLabelLoggerConfidence(unittest.TestCase):
    def setUp(self):
        self.log_dir = "/tmp/test_logger_confidence"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
    
    def tearDown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
            
    def test_confidence_tracking(self):
        """Test that confidences are tracked correctly."""
        logger = HardLabelLogger(self.log_dir, log_freq=10)
        logger.set_epoch(0, "test")
        
        # Log with confidences
        confidences = [
            {"real_prob": 0.2, "fake_prob": 0.8, "confidence": 0.8},
            {"real_prob": 0.9, "fake_prob": 0.1, "confidence": 0.9}
        ]
        logger.log(
            loss=0.5,
            outputs=["Fake", "Real"],
            gt=["Fake", "Real"],  # Both correct
            confidences=confidences
        )
        
        self.assertEqual(len(logger.all_confidences), 2)
        self.assertEqual(len(logger.correct_confidences), 2)  # Both correct
        self.assertEqual(len(logger.incorrect_confidences), 0)
        
    def test_confidence_metrics_in_log_epoch(self):
        """Test that confidence metrics are computed in log_epoch."""
        logger = HardLabelLogger(self.log_dir, log_freq=10)
        logger.set_epoch(0, "test")
        
        # Log correct and incorrect predictions
        logger.log(
            outputs=["Fake", "Real"],
            gt=["Fake", "Fake"],  # Second is incorrect
            confidences=[
                {"real_prob": 0.1, "fake_prob": 0.9, "confidence": 0.9},
                {"real_prob": 0.6, "fake_prob": 0.4, "confidence": 0.6}
            ]
        )
        
        logger.log_epoch()
        
        # Check metrics file
        with open(os.path.join(self.log_dir, "metrics.jsonl")) as f:
            metrics = json.loads(f.readline())
        
        self.assertIn("confidence_mean", metrics)
        self.assertIn("confidence_correct_mean", metrics)
        self.assertIn("confidence_incorrect_mean", metrics)
        self.assertAlmostEqual(metrics["confidence_mean"], 0.75, places=2)
        
    def test_save_all_predictions_with_confidence(self):
        """Test that predictions file includes confidence."""
        logger = HardLabelLogger(self.log_dir, log_freq=10, save_all_predictions=True)
        logger.set_epoch(0, "test")
        
        logger.log(
            outputs=["Fake"],
            gt=["Fake"],
            confidences=[{"real_prob": 0.2, "fake_prob": 0.8, "confidence": 0.8}]
        )
        logger.log_epoch()
        
        pred_file = os.path.join(self.log_dir, "predictions_test_epoch_0.jsonl")
        self.assertTrue(os.path.exists(pred_file))
        
        with open(pred_file) as f:
            record = json.loads(f.readline())
        
        self.assertIn("real_prob", record)
        self.assertIn("fake_prob", record)
        self.assertIn("confidence", record)


class TestReasoningLoggerConfidence(unittest.TestCase):
    def setUp(self):
        self.log_dir = "/tmp/test_reasoning_logger_confidence"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
    
    def tearDown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
            
    def test_confidence_with_reasoning_format(self):
        """Test confidence tracking with reasoning format outputs."""
        logger = ReasoningLogger(self.log_dir, log_freq=10)
        logger.set_epoch(0, "test")
        
        # Log reasoning format outputs
        logger.log(
            outputs=["<think>blah</think><reasons>[]</reasons><answer>Fake</answer>"],
            gt=["<think>analysis</think><reasons>[]</reasons><answer>Fake</answer>"],
            confidences=[{"real_prob": 0.1, "fake_prob": 0.9, "confidence": 0.9}]
        )
        
        self.assertEqual(len(logger.all_confidences), 1)
        self.assertEqual(logger.all_confidences[0], 0.9)
        
    def test_per_class_confidence(self):
        """Test that per-class confidences are tracked."""
        logger = ReasoningLogger(self.log_dir, log_freq=10)
        logger.set_epoch(0, "test")
        
        logger.log(
            outputs=["<answer>Real</answer>", "<answer>Fake</answer>"],
            gt=["<answer>Real</answer>", "<answer>Fake</answer>"],
            confidences=[
                {"real_prob": 0.8, "fake_prob": 0.2, "confidence": 0.8},
                {"real_prob": 0.3, "fake_prob": 0.7, "confidence": 0.7}
            ]
        )
        
        self.assertEqual(len(logger.real_confidences), 1)
        self.assertEqual(len(logger.fake_confidences), 1)
        self.assertEqual(logger.real_confidences[0], 0.8)
        self.assertEqual(logger.fake_confidences[0], 0.7)


class TestLoggerWithoutConfidence(unittest.TestCase):
    """Test that loggers work fine when confidences are not provided."""
    
    def setUp(self):
        self.log_dir = "/tmp/test_logger_no_confidence"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
    
    def tearDown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
    
    def test_hard_label_without_confidence(self):
        """Test HardLabelLogger works without confidence."""
        logger = HardLabelLogger(self.log_dir, log_freq=10)
        logger.set_epoch(0, "test")
        
        logger.log(outputs=["Fake"], gt=["Fake"])
        logger.log_epoch()
        
        with open(os.path.join(self.log_dir, "metrics.jsonl")) as f:
            metrics = json.loads(f.readline())
        
        self.assertIn("accuracy", metrics)
        self.assertNotIn("confidence_mean", metrics)  # No confidence logged


if __name__ == "__main__":
    unittest.main()
