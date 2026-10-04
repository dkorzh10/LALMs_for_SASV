"""Tests for DistillationPlotter."""
import unittest
import tempfile
import os
import json
from src.analysis.plotter_distillation import DistillationPlotter
from src.analysis.plotter_distillation_common import load_distillation_metrics, load_distillation_form_stats, get_distillation_iterations


class TestDistillationPlotter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.outdir = os.path.join(self.tmpdir, "plots")
        self.distill_dir = os.path.join(self.tmpdir, "distillation")
        os.makedirs(self.distill_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_load_distillation_form_stats(self):
        stats_dir = self.distill_dir
        with open(os.path.join(stats_dir, "form_stats_iter_0.json"), "w") as f:
            json.dump({"n_raw": 100, "n_final": 50, "lengths_chars": [10, 20], "lengths_tokens": [1, 2]}, f)
        loaded = load_distillation_form_stats(self.tmpdir)
        self.assertIn(0, loaded)
        self.assertEqual(loaded[0]["n_raw"], 100)

    def test_get_distillation_iterations(self):
        stats_dir = self.distill_dir
        with open(os.path.join(stats_dir, "form_stats_iter_1.json"), "w") as f:
            json.dump({"n_raw": 50}, f)
        iters = get_distillation_iterations(self.tmpdir)
        self.assertEqual(iters, [1])

    def test_distillation_plotter_generate(self):
        stats_dir = self.distill_dir
        with open(os.path.join(stats_dir, "form_stats_iter_0.json"), "w") as f:
            json.dump({
                "n_raw": 100, "n_final": 50,
                "lengths_chars": [10, 20, 30, 40, 50],
                "lengths_tokens": [2, 4, 6, 8, 10],
            }, f)
        plotter = DistillationPlotter(self.tmpdir, self.outdir)
        plotter.generate_plots()
        self.assertTrue(os.path.exists(os.path.join(self.outdir, "text_len_per_iteration.png")))
        self.assertTrue(os.path.exists(os.path.join(self.outdir, "iteration_0", "dataset_forming", "text_len_chars_distribution.png")))
