import unittest
import os
import shutil
import json
from src.analysis.plotter import (
    Plotter, detect_run_type, load_test_data, compute_test_metrics,
    plot_test_run, extract_answer, extract_answer_from_gt,
)
from src.analysis.plotter_common import extract_answer_from_tag, compute_tag_parse_metrics, extract_pred_answer


class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.test_dir = "/tmp/test_plotter"
        self.run_dir = os.path.join(self.test_dir, "logs")
        self.output_dir = os.path.join(self.test_dir, "plots")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create dummy metrics (format matches hard_label_logger/reasoning_logger)
        self.metrics = [
            {"epoch": 0, "iteration": 1, "type": "train_batch", "loss": 1.0, "lr": 0.001},
            {"epoch": 0, "type": "validation", "loss": 0.9, "accuracy": 0.5},
            {"epoch": 1, "iteration": 2, "type": "train_batch", "loss": 0.8, "lr": 0.0009},
            {"epoch": 1, "type": "validation", "loss": 0.7, "accuracy": 0.6},
        ]
        with open(os.path.join(self.run_dir, "metrics.jsonl"), "w") as f:
            for m in self.metrics:
                f.write(json.dumps(m) + "\n")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_generation(self):
        plotter = Plotter(self.run_dir, self.output_dir)
        plotter.generate_plots()
        
        # Check if files exist
        expected_files = ["learning_rate.png", "training_overview.png", "validation_accuracy.png"]
        for f in expected_files:
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, f)))


class TestRunTypeDetection(unittest.TestCase):
    def setUp(self):
        self.test_dir = "/tmp/test_run_type_detection"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_detect_train(self):
        os.makedirs(os.path.join(self.test_dir, "logs"), exist_ok=True)
        with open(os.path.join(self.test_dir, "logs", "metrics.jsonl"), "w") as f:
            f.write(json.dumps({"type": "train_batch"}) + "\n")
        run_type, test_dirs = detect_run_type(self.test_dir)
        self.assertEqual(run_type, "train")
        self.assertEqual(test_dirs, [])

    def test_detect_test(self):
        test_log_dir = os.path.join(self.test_dir, "logs", "test_asvspoof")
        os.makedirs(test_log_dir, exist_ok=True)
        with open(os.path.join(test_log_dir, "predictions_test_epoch_0.jsonl"), "w") as f:
            f.write(json.dumps({"gt": "Fake", "output": "Fake"}) + "\n")
        run_type, test_dirs = detect_run_type(self.test_dir)
        self.assertEqual(run_type, "test")
        self.assertEqual(len(test_dirs), 1)
        self.assertIn("test_asvspoof", test_dirs[0])


class TestAnswerExtraction(unittest.TestCase):
    def test_extract_reasoning_format(self):
        self.assertEqual(extract_answer("<answer>Real</answer>"), "Real")
        self.assertEqual(extract_answer("<answer>Fake</answer>"), "Fake")

    def test_extract_hard_label_format(self):
        self.assertEqual(extract_answer("Final Answer: Real"), "Real")
        self.assertEqual(extract_answer("Final Answer: Fake"), "Fake")

    def test_extract_answer_from_gt(self):
        # Simple cases
        self.assertEqual(extract_answer_from_gt("real"), "Real")
        self.assertEqual(extract_answer_from_gt("fake"), "Fake")
        self.assertEqual(extract_answer_from_gt("Real"), "Real")
        self.assertEqual(extract_answer_from_gt("Fake"), "Fake")
        
        # Hard-label format
        self.assertEqual(extract_answer_from_gt("Final Answer: Real"), "Real")
        self.assertEqual(extract_answer_from_gt("Final Answer: Fake"), "Fake")
        
        # Reasoning format
        self.assertEqual(extract_answer_from_gt("<answer>Real</answer>"), "Real")
        self.assertEqual(extract_answer_from_gt("<answer>Fake</answer>"), "Fake")

    def test_extract_answer_from_tag_strict(self):
        self.assertEqual(extract_answer_from_tag("<answer>yes</answer>"), "yes")
        self.assertEqual(extract_answer_from_tag("<answer>Real</answer>"), "Real")
        self.assertEqual(extract_answer_from_tag("reasoning says no <answer>gen</answer>"), "gen")
        self.assertEqual(extract_answer_from_tag("reasoning says no without tag"), "")
        self.assertEqual(extract_answer_from_tag("<answer>yes</"), "")
        self.assertEqual(extract_answer_from_tag("Final Answer: yes"), "")

    def test_compute_tag_parse_metrics(self):
        outputs = [
            "<answer>yes</answer>",
            "thinking mentions no but no tag",
            "<answer>no</answer>",
        ]
        gts = ["yes", "yes", "yes"]
        m = compute_tag_parse_metrics(outputs, gts)
        self.assertAlmostEqual(m["ans_parsed"], 2 / 3)
        self.assertAlmostEqual(m["acc2parse"], 0.5)

    def test_extract_pred_answer_reasoning_requires_tag(self):
        self.assertEqual(
            extract_pred_answer(
                "<think>mentions no</think><answer>yes</answer>"
            ),
            "yes",
        )
        self.assertEqual(
            extract_pred_answer("<think>mentions no</think>"),
            "",
        )
        self.assertEqual(extract_pred_answer("Fake"), "Fake")


class TestTestRunPlotting(unittest.TestCase):
    def setUp(self):
        self.test_dir = "/tmp/test_test_run_plotting"
        self.test_log_dir = os.path.join(self.test_dir, "logs", "test_ds")
        os.makedirs(self.test_log_dir, exist_ok=True)
        preds = [
            {"gt": "Fake", "output": "Fake", "confidence": 0.9, "fake_prob": 0.9},
            {"gt": "Real", "output": "Real", "confidence": 0.8, "fake_prob": 0.2},
        ]
        with open(os.path.join(self.test_log_dir, "predictions_test_epoch_0.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_and_compute_metrics(self):
        preds, metrics_jsonl = load_test_data(self.test_log_dir)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 2)
        self.assertIsNone(metrics_jsonl)
        metrics = compute_test_metrics(preds)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_prediction_distribution_with_unknown(self):
        """Test prediction distribution includes unknown count."""
        preds = [
            {"gt": "Fake", "output": "Fake", "confidence": 0.9},
            {"gt": "Real", "output": "Real", "confidence": 0.8},
            {"gt": "Fake", "output": "gibberish", "confidence": 0.5},
        ]
        with open(os.path.join(self.test_log_dir, "predictions_test_epoch_1.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        preds_loaded, _ = load_test_data(self.test_log_dir)
        self.assertEqual(len(preds_loaded), 3)
        # compute_test_metrics skips unknown - should get 2 valid
        metrics = compute_test_metrics(preds_loaded)
        self.assertEqual(metrics["total"], 2)

    def test_plot_test_run(self):
        output_dir = os.path.join(self.test_dir, "plots")
        metrics = plot_test_run(self.test_log_dir, output_dir, "ds")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertTrue(os.path.exists(os.path.join(output_dir, "confidence_histogram.png")))

    def test_metrics_jsonl_fallback(self):
        """When no predictions file, load from metrics.jsonl."""
        metrics_only_dir = os.path.join(self.test_dir, "metrics_only")
        os.makedirs(os.path.join(metrics_only_dir, "logs", "test_ds"), exist_ok=True)
        with open(os.path.join(metrics_only_dir, "logs", "test_ds", "metrics.jsonl"), "w") as f:
            f.write(json.dumps({"type": "test", "epoch": 0, "accuracy": 0.9, "accuracy_balanced": 0.88}) + "\n")
        preds, metrics_jsonl = load_test_data(os.path.join(metrics_only_dir, "logs", "test_ds"))
        self.assertIsNone(preds)
        self.assertIsNotNone(metrics_jsonl)
        self.assertEqual(metrics_jsonl["accuracy"], 0.9)
        result = plot_test_run(os.path.join(metrics_only_dir, "logs", "test_ds"), os.path.join(metrics_only_dir, "plots"), "ds")
        self.assertIsNotNone(result)
        self.assertEqual(result["accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()





