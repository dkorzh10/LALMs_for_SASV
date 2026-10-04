import unittest
import os
import torch
import numpy as np
from src.dataloaders.dataset import AudioDataset, collate_fn

class TestDataset(unittest.TestCase):
    def test_dummy_dataset(self):
        dataset = AudioDataset("dummy", max_samples=5)
        self.assertEqual(len(dataset), 5)
        sample = dataset[0]
        self.assertIn("audio_id", sample)
        self.assertIn("raw_wav", sample)
        self.assertEqual(sample["raw_wav"].shape[1], 16000)

    def test_collate_fn(self):
        dataset = AudioDataset("dummy", max_samples=4)
        batch = [dataset[i] for i in range(4)]
        collated = collate_fn(batch)
        self.assertIn("spectrogram", collated)
        self.assertIn("text", collated)
        self.assertEqual(len(collated["audio_ids"]), 4)


    def test_eval_splits_disable_augmentations(self):
        cfg = {"aug": {"enabled": True, "p_noise": 1.0, "p_rawboost": 1.0}}
        train_ds = AudioDataset("dummy", max_samples=1, audio_cfg=cfg, split="train")
        val_ds = AudioDataset("dummy", max_samples=1, audio_cfg=cfg, split="val")
        test_ds = AudioDataset("dummy", max_samples=1, audio_cfg=cfg, split="test")
        dev_ds = AudioDataset("dummy", max_samples=1, audio_cfg=cfg, split="dev")
        self.assertTrue(train_ds.augmenter.aug_cfg.enabled)
        self.assertEqual(train_ds.augmenter.crop_mode, "random")
        for ds in (val_ds, test_ds, dev_ds):
            self.assertFalse(ds.augmenter.aug_cfg.enabled)
            self.assertEqual(ds.augmenter.crop_mode, "center")

    def test_collate_fn_sasv_format(self):
        """Test collate_fn with SASV format (reference_audios and query_audios)"""
        import numpy as np
        
        # Create mock SASV batch
        batch = [
            {
                "audio_id": "test_1",
                "reference_audios": [torch.randn(1, 16000), torch.randn(1, 16000)],
                "query_audios": [torch.randn(1, 16000)],
                "gt": "verified",
                "answer": "yes",
                "task": "sasv",
                "text": "Final Answer: yes",
                "reasoning": ""
            },
            {
                "audio_id": "test_2",
                "reference_audios": [torch.randn(1, 16000)],
                "query_audios": [torch.randn(1, 16000), torch.randn(1, 16000)],
                "gt": "rejected",
                "answer": "no",
                "task": "sasv",
                "text": "Final Answer: no",
                "reasoning": ""
            }
        ]
        
        # Test with SALMON model (no processor, should use else branch)
        collated = collate_fn(batch, processor=None, model_name="salmon", prompt_templates=None)
        
        # Check that it handles numpy arrays correctly (no ValueError)
        self.assertIn("spectrogram", collated)
        self.assertIn("audio_ids", collated)
        self.assertEqual(len(collated["audio_ids"]), 2)
        self.assertIn("gt", collated)
        self.assertIn("answer", collated)
        
        # SALMON gets padded raw audio plus a BEATs/WavLM padding mask.
        self.assertIsInstance(collated["raw_wav"], torch.Tensor)
        self.assertIsInstance(collated["padding_mask"], torch.Tensor)
        self.assertEqual(collated["raw_wav"].shape[0], 2)
        self.assertEqual(collated["padding_mask"].shape, collated["raw_wav"].shape)

    def test_collate_fn_sasv_format_empty_audio(self):
        """Test collate_fn with SASV format when audio is empty/None"""
        import numpy as np
        
        # Create mock SASV batch with empty reference audios
        batch = [
            {
                "audio_id": "test_empty",
                "reference_audios": [],
                "query_audios": [],
                "gt": "verified",
                "answer": "yes",
                "task": "sasv",
                "text": "Final Answer: yes",
                "reasoning": ""
            }
        ]
        
        # Test with SALMON model (no processor, should use else branch)
        collated = collate_fn(batch, processor=None, model_name="salmon", prompt_templates=None)
        
        # Should handle empty audio gracefully (create dummy audio)
        self.assertIn("raw_wav", collated)
        self.assertIn("padding_mask", collated)
        self.assertEqual(collated["raw_wav"].shape[0], 1)
        # Should have created a dummy audio array
        self.assertIsInstance(collated["raw_wav"], torch.Tensor)
        self.assertEqual(collated["raw_wav"].shape[1], 16000)  # Dummy audio length

if __name__ == "__main__":
    unittest.main()





