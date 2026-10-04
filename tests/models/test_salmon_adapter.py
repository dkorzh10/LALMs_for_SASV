import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock heavy dependencies BEFORE importing the model
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["src.models.SALMON.salmonn"] = MagicMock()

# Now we can safely import
from src.models.salmon import SalmonModel

class TestSalmonAdapter(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model_name": "salmon",
            "additional_kwargs": {"salmon": {}},
            "lora": {"enabled": False}
        }
        
        # Patch the SALMONN class inside the module where it is used
        with patch("src.models.salmon.SALMONN") as MockSalmonn:
            self.mock_inner_model = MockSalmonn.from_config.return_value
            self.model = SalmonModel(self.config)

    def test_forward_adapter(self):
        samples = {
            "text": ["Describe <Audio>", "No tag here"],
            "prompts": ["<Audio> is the input"]
        }
        
        self.model.forward(samples)
        
        # Verify the inner model received converted tags
        call_args = self.model.model.forward.call_args[0][0]
        self.assertEqual(call_args["text"][0], "Describe <SpeechHere>")
        self.assertEqual(call_args["text"][1], "No tag here")
        self.assertEqual(call_args["prompts"][0], "<SpeechHere> is the input")

    def test_generate_adapter(self):
        prompts = ["Analyze <Audio>"]
        samples = {}
        generate_cfg = {}
        
        self.model.generate(samples, generate_cfg, prompts=prompts)
        
        # Verify prompts were converted
        call_kwargs = self.model.model.generate.call_args[1]
        self.assertEqual(call_kwargs["prompts"][0], "Analyze <SpeechHere>")

if __name__ == "__main__":
    unittest.main()





