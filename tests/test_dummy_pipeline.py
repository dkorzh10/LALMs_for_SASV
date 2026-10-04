import unittest
import sys
import os
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.runner import Runner

class TestDummyPipeline(unittest.TestCase):
    def setUp(self):
        self.base_config_dir = os.path.join(os.path.dirname(__file__), "../experiment_configs/tests")
        self.output_dir = "./outputs"
        
        # Cleanup potentially existing outputs
        for subdir in ["test_run", "test_run_grpo", "test_run_distill"]:
            path = os.path.join(self.output_dir, subdir)
            if os.path.exists(path):
                shutil.rmtree(path)

    def test_run_sft_dummy(self):
        config_path = os.path.join(self.base_config_dir, "config.yaml") # Original SFT config
        
        runner = Runner(config_path)
        runner.run()
        
        output_path = runner.output_dir
        self.assertTrue(os.path.exists(os.path.join(output_path, "logs", "metrics.jsonl")))

    def test_run_grpo_dummy(self):
        config_path = os.path.join(self.base_config_dir, "config_grpo.yaml")
        
        runner = Runner(config_path)
        runner.run()
        
        output_path = runner.output_dir
        self.assertTrue(os.path.exists(os.path.join(output_path, "logs", "metrics.jsonl")))

    def test_run_distill_dummy(self):
        config_path = os.path.join(self.base_config_dir, "config_distill.yaml")
        
        runner = Runner(config_path)
        runner.run()
        
        output_path = runner.output_dir
        self.assertTrue(os.path.exists(os.path.join(output_path, "logs", "metrics.jsonl")))

if __name__ == "__main__":
    unittest.main()
