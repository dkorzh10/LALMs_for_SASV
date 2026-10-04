import unittest
import os
import json
import tempfile
import torch
from src.dataloaders.dataset import AudioDataset

class TestDatasetReal(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "dataset.json")
        
        # Create a dummy audio file
        self.audio_path = os.path.join(self.test_dir, "test.wav")
        with open(self.audio_path, "w") as f:
            f.write("dummy")

        self.data = [
            {"audio_id": "1", "original_path": self.audio_path, "is_bonafide": True, "reasoning": "Real"},
            {"audio_id": "2", "original_path": self.audio_path, "is_bonafide": False, "reasoning": "Fake"}
        ]
        
        with open(self.data_path, "w") as f:
            json.dump(self.data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_json_loading(self):
        # Mock torchaudio to avoid errors reading the dummy file
        with unittest.mock.patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.randn(1, 16000), 16000)
            
            dataset = AudioDataset(self.data_path)
            
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["audio_id"], "1")
            self.assertEqual(dataset[1]["is_bonafide"], False)
            self.assertEqual(dataset[0]["text"], "Real")

if __name__ == "__main__":
    import unittest.mock
    unittest.main()





