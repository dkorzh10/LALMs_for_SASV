import json
import os
import shutil
import unittest

import numpy as np

from src.analysis.plotter_test import load_test_data
from src.epochs.utils.sasv_metrics import (
    arrays_from_predictions,
    recompute_sasv_metrics_from_predictions,
)


class TestRecomputeSasvMetrics(unittest.TestCase):
    def setUp(self):
        self.test_dir = "/tmp/test_recompute_sasv_metrics"
        self.log_dir = os.path.join(self.test_dir, "logs", "test_sasv_ds")
        os.makedirs(self.log_dir, exist_ok=True)

        rows = [
            {
                "gt": "yes",
                "output": "yes",
                "audio_id": "a1",
                "yes_prob": 0.8,
                "no_prob": 0.15,
                "gen_prob": 0.05,
            },
            {
                "gt": "no",
                "output": "no",
                "audio_id": "a2",
                "yes_prob": 0.1,
                "no_prob": 0.85,
                "gen_prob": 0.05,
            },
            {
                "gt": "gen",
                "output": "gen",
                "audio_id": "a3",
                "yes_prob": 0.05,
                "no_prob": 0.05,
                "gen_prob": 0.9,
            },
        ]
        with open(os.path.join(self.log_dir, "predictions_test_epoch_0_rank0.jsonl"), "w") as f:
            for row in rows[:2]:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(self.log_dir, "predictions_test_epoch_0_rank1.jsonl"), "w") as f:
            f.write(json.dumps(rows[2]) + "\n")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_and_merge_rank_files(self):
        predictions, metrics_jsonl = load_test_data(self.log_dir)
        self.assertIsNone(metrics_jsonl)
        self.assertEqual(len(predictions), 3)

    def test_arrays_from_predictions_proxy_scores(self):
        predictions, _ = load_test_data(self.log_dir)
        labels, yes_probs, gen_probs, asv_scores, cm_scores = arrays_from_predictions(predictions)

        self.assertEqual(labels.tolist(), ["yes", "no", "gen"])
        np.testing.assert_allclose(yes_probs, [0.8, 0.1, 0.05])
        np.testing.assert_allclose(gen_probs, [0.05, 0.05, 0.9])
        np.testing.assert_allclose(asv_scores, [0.8 / 0.95, 0.1 / 0.95, 0.5], rtol=1e-6)
        np.testing.assert_allclose(cm_scores, [0.05, 0.05, 0.9])

    def test_recompute_metrics_keys(self):
        predictions, _ = load_test_data(self.log_dir)
        metrics = recompute_sasv_metrics_from_predictions(predictions, plot_dir=None)

        for key in ("min_a_dcf", "min_t_dcf", "t_eer", "sv_eer", "spf_eer"):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
