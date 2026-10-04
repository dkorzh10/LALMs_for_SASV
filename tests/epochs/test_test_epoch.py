import unittest
from unittest.mock import MagicMock, patch
import torch
from src.epochs.test_epoch import TestEpoch
from src.loggers.hard_label_logger import HardLabelLogger


class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Simple mock: each word is one token
        return list(range(len(text.split())))


class MockModel:
    def __init__(self):
        self.processor = MagicMock()
        self.processor.tokenizer = MockTokenizer()
    
    def eval(self):
        pass
    
    def forward(self, batch, verbose=False):
        return {"loss": torch.tensor(0.5), "correct": 1, "total": 1}
    
    def generate(self, batch, gen_cfg, return_outputs=False):
        batch_size = len(batch.get("audio_ids", ["1"]))
        texts = ["<think>analysis</think><reasons>[]</reasons><answer>Fake</answer>"] * batch_size
        
        if return_outputs:
            completion_ids = torch.randint(0, 100, (batch_size, 10))
            # Logits: [batch, seq_len, vocab_size]
            # Make "Fake" token (id=1) have higher probability
            logits = torch.randn(batch_size, 10, 1000)
            logits[:, :, 0] = -5  # Real token
            logits[:, :, 1] = 5   # Fake token
            return texts, completion_ids, logits
        return texts


class TestTestEpoch(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.logger = MagicMock(spec=HardLabelLogger)
        self.dataloader = [
            {
                "audio_ids": ["1", "2"],
                "text": ["<answer>Fake</answer>", "<answer>Real</answer>"],
                "raw_wav": [torch.randn(16000), torch.randn(16000)]
            }
        ]
    
    def test_extract_answer_reasoning_format(self):
        """Test answer extraction from reasoning format."""
        epoch = TestEpoch(self.model, self.dataloader, self.logger, 
                         model_format="reasoning", dataset_format="reasoning")
        
        text = "<think>blah</think><reasons>['STRANGE_VOICE']</reasons><answer>Fake</answer>"
        answer = epoch._extract_answer(text)
        self.assertEqual(answer, "Fake")
        
    def test_extract_answer_hard_label_format(self):
        """Test answer extraction from hard-label format."""
        epoch = TestEpoch(self.model, self.dataloader, self.logger)
        
        text = "Final Answer: Real"
        answer = epoch._extract_answer(text)
        self.assertEqual(answer, "Real")
        
    def test_extract_answer_fallback(self):
        """Test fallback answer extraction."""
        epoch = TestEpoch(self.model, self.dataloader, self.logger)
        
        text = "I think this is fake audio"
        answer = epoch._extract_answer(text)
        self.assertEqual(answer, "Fake")
        
    def test_find_answer_position_reasoning(self):
        """Test finding answer position in reasoning format."""
        epoch = TestEpoch(self.model, self.dataloader, self.logger)
        tokenizer = MockTokenizer()
        
        text = "<think>analysis</think><reasons>[]</reasons><answer>Fake</answer>"
        pos = epoch._find_answer_position(text, tokenizer)
        # Position should be after <answer>
        self.assertGreater(pos, 0)
        
    def test_extract_confidences(self):
        """Test confidence extraction from logits (single-token Real/Fake)."""
        epoch = TestEpoch(self.model, self.dataloader, self.logger)
        
        # Create logits where Fake has higher probability at position 0
        logits = torch.zeros(2, 5, 1000)
        logits[:, 0, 0] = -2  # Real token id=0 = low prob
        logits[:, 0, 1] = 2   # Fake token id=1 = high prob
        
        texts = ["<answer>Fake</answer>", "<answer>Real</answer>"]
        
        # Mock tokenizer and token IDs: Real -> [0], Fake -> [1] (single-token each)
        with patch.object(epoch, '_get_tokenizer', return_value=MockTokenizer()):
            with patch.object(epoch, '_find_answer_position', return_value=0):
                with patch.object(epoch, '_get_token_ids', return_value=([0], [1])):
                    confidences = epoch._extract_confidences(texts, logits)
        
        self.assertEqual(len(confidences), 2)
        # First sample predicted Fake with high confidence
        self.assertGreater(confidences[0]["fake_prob"], confidences[0]["real_prob"])
        self.assertIn("confidence", confidences[0])
        
    def test_run_calls_logger_with_confidences(self):
        """Test that run() passes confidences to logger."""
        epoch = TestEpoch(self.model, self.dataloader, self.logger,
                         device=torch.device("cpu"),
                         extract_confidence=True)
        
        # Mock the confidence extraction
        with patch.object(epoch, '_extract_confidences', return_value=[
            {"real_prob": 0.2, "fake_prob": 0.8, "confidence": 0.8},
            {"real_prob": 0.7, "fake_prob": 0.3, "confidence": 0.7}
        ]):
            epoch.run(epoch_num=0)
        
        # Check logger was called with confidences
        self.assertTrue(self.logger.log.called)
        call_kwargs = self.logger.log.call_args[1]
        self.assertIn("confidences", call_kwargs)


class TestTestEpochFormatConversion(unittest.TestCase):
    """Test format conversion when reasoning model tested on hard_label dataset."""
    
    def setUp(self):
        self.model = MockModel()
        self.logger = MagicMock(spec=HardLabelLogger)
        self.dataloader = [
            {
                "audio_ids": ["1"],
                "text": ["Fake"],  # Hard-label format GT
                "raw_wav": [torch.randn(16000)]
            }
        ]
    
    def test_reasoning_model_on_hard_label_dataset(self):
        """Test that reasoning model output is converted for hard_label dataset."""
        epoch = TestEpoch(self.model, self.dataloader, self.logger,
                         device=torch.device("cpu"),
                         model_format="reasoning",
                         dataset_format="hard_label",
                         extract_confidence=False)
        
        epoch.run(epoch_num=0)
        
        # Logger should receive extracted answers, not full reasoning
        call_kwargs = self.logger.log.call_args[1]
        outputs = call_kwargs.get("outputs")
        # Should be ["Fake"] extracted from "<answer>Fake</answer>"
        self.assertEqual(outputs[0], "Fake")


if __name__ == "__main__":
    unittest.main()
