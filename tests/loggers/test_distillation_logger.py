"""Tests for DistillationLogger."""
import unittest
import tempfile
import os
import json
from src.loggers.distillation_logger import DistillationLogger


class TestDistillationLogger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_set_distillation_iter(self):
        logger = DistillationLogger(self.tmpdir, 10)
        logger.set_distillation_iter(2, "dataset_forming")
        self.assertEqual(logger.distillation_iter, 2)
        self.assertEqual(logger.distillation_phase, "dataset_forming")

    def test_log_dataset_forming_stats(self):
        logger = DistillationLogger(self.tmpdir, 10)
        logger.log_dataset_forming_stats(
            distillation_iter=0,
            n_raw=100,
            n_final=50,
            lengths_chars=[10, 20, 30],
            lengths_tokens=[2, 4, 6],
        )
        stats_path = os.path.join(self.tmpdir, "iteration_0", "dataset_forming", "form_stats_iter_0.json")
        self.assertTrue(os.path.exists(stats_path))
        with open(stats_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["n_raw"], 100)
        self.assertEqual(data["n_final"], 50)
        self.assertEqual(data["lengths_chars"], [10, 20, 30])
        self.assertEqual(data["lengths_tokens"], [2, 4, 6])
