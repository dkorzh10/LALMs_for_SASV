import os
import json
import unittest
from src.judges.format_judge import (
    FormatJudge,
    _compute_correct_features_overlap,
    _compute_correct_reasons_overlap,
    _flatten_acoustic_features,
    _parse_features,
    _parse_reasons,
)


class TestFormatJudge(unittest.TestCase):
    def test_scoring(self):
        judge = FormatJudge()
        
        inputs = ["audio1", "audio2", "audio3"]
        outputs = [
            "<think>analysis</think><reasons>[]</reasons><answer>real</answer>",   # Correct format & answer (gt bonafide -> real)
            "<think>analysis</think><reasons>[]</reasons><answer>spoof</answer>",   # Wrong format (spoof not accepted); wrong answer
            "Malformed output"                                                # Wrong format
        ]
        gt = ["bonafide", "bonafide", "bonafide"]
        
        score = judge.score(inputs, outputs, gt)
        per_sample = score["meta"]["per_sample"]
        self.assertEqual(per_sample[0]["score"], 1.0)
        self.assertEqual(per_sample[1]["score"], 0.5)  # format ok, wrong answer (spoof not real/fake)
        self.assertEqual(per_sample[2]["score"], 0.0)
        self.assertEqual(score["score"], 0.5)  # (1.0 + 0.5 + 0.0) / 3 = 0.5
        self.assertEqual(per_sample[0]["answer"], "real")
        self.assertIsNone(per_sample[1]["answer"])  # spoof not accepted, model should use "fake"
        self.assertIsNone(per_sample[2]["answer"])

    def test_correct_reasons_overlap(self):
        judge = FormatJudge({"weights": {"reasons_correctness": 0.5, "format": 0.5, "correctness": 0.5}})
        inputs = ["a1", "a2"]
        # gt with reasons: STRANGE_VOICE, UNNATURAL_PAUSES (2 items)
        # out1: same 2 -> overlap 1.0 * 1.0 = 1.0
        # out2: format wrong -> overlap 0
        outputs = [
            "<think>x</think><reasons>['STRANGE_VOICE', 'UNNATURAL_PAUSES']</reasons><answer>Fake</answer>",
            "Malformed",
        ]
        gt = [
            "<think>x</think><reasons>['STRANGE_VOICE', 'UNNATURAL_PAUSES']</reasons><answer>Fake</answer>",
            "<think>x</think><reasons>['STRANGE_VOICE']</reasons><answer>Fake</answer>",
        ]
        score = judge.score(inputs, outputs, gt)
        per_sample = score["meta"]["per_sample"]
        # Sample 0: format_ok=1, is_correct=1, overlap=1.0 -> 0.5*1+0.5*1+0.5*1.0=1.5
        self.assertEqual(per_sample[0]["correct_reasons_overlap"], 1.0)
        self.assertEqual(per_sample[0]["score"], 1.5)
        # Sample 1: format wrong -> overlap=0
        self.assertEqual(per_sample[1]["correct_reasons_overlap"], 0.0)
        self.assertEqual(per_sample[1]["score"], 0.0)

    def test_correct_reasons_overlap_hard_label_gt(self):
        judge = FormatJudge({"weights": {"reasons_correctness": 0.5}})
        # gt has no <reasons> (hard-label Fake) -> overlap stays 0 (Fake samples need reasons to compare)
        outputs = ["<think>x</think><reasons>['STRANGE_VOICE']</reasons><answer>Fake</answer>"]
        gt = ["Final Answer: Fake"]  # no reasons
        score = judge.score(["a1"], outputs, gt)
        self.assertEqual(score["meta"]["per_sample"][0]["correct_reasons_overlap"], 0.0)

    def test_correct_reasons_overlap_real_gt(self):
        judge = FormatJudge({"weights": {"reasons_correctness": 0.5, "format": 0.5, "correctness": 0.5}})
        # gt is Real (no reasons) -> correct Real prediction should get overlap=1.0, not 0
        outputs = ["<think>x</think><reasons>[]</reasons><answer>Real</answer>"]
        gt = ["<think>x</think><reasons>[]</reasons><answer>Real</answer>"]
        score = judge.score(["a1"], outputs, gt)
        self.assertEqual(score["meta"]["per_sample"][0]["correct_reasons_overlap"], 1.0)
        # gt as plain "Real" (from grpo_logger when is_bonafide=True)
        score2 = judge.score(["a2"], outputs, ["Real"])
        self.assertEqual(score2["meta"]["per_sample"][0]["correct_reasons_overlap"], 1.0)

    def test_sasv_hard_label_gt(self):
        judge = FormatJudge({"answer_labels": "sasv", "require_reasons": False})
        outputs = [
            "<think>same speaker</think><answer>yes</answer>",
            "<think>spoof cues</think><answer>gen</answer>",
            "Malformed output",
        ]
        gt = ["yes", "no", "gen"]
        score = judge.score(["a1", "a2", "a3"], outputs, gt)
        per_sample = score["meta"]["per_sample"]
        self.assertEqual(per_sample[0]["score"], 1.0)
        self.assertEqual(per_sample[0]["answer"], "yes")
        self.assertEqual(per_sample[1]["score"], 0.5)  # format ok, wrong label
        self.assertEqual(per_sample[2]["score"], 0.0)
        self.assertAlmostEqual(score["score"], (1.0 + 0.5 + 0.0) / 3)


class TestAcousticFeaturesJudge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("ENRICHED_TRACES_JSON", "")
        if not path or not os.path.exists(path):
            raise unittest.SkipTest("ENRICHED_TRACES_JSON not set")
        with open(path) as f:
            cls.sample = json.load(f)[0]

    def _build_gt(self, item):
        answer = item.get("gt_label", item.get("gt", "gen"))
        return (
            f"<features>{item['Acoustic features']}</features>"
            f"<think>{item['reasoning']}</think>"
            f"<reasons>{item['reasons']}</reasons>"
            f"<answer>{answer}</answer>"
        )

    def test_parse_features_from_gt(self):
        gt = self._build_gt(self.sample)
        features = _parse_features(gt)
        self.assertIn("age", features)
        self.assertIn("reference", features["age"])

    def test_features_reasons_reward(self):
        gt = self._build_gt(self.sample)
        judge = FormatJudge({
            "answer_labels": "sasv",
            "require_reasons": True,
            "require_features": True,
            "weights": {
                "format": 0.4,
                "correctness": 0.4,
                "reasons_correctness": 0.1,
                "features_correctness": 0.1,
            },
        })
        perfect = (
            f"<features>{self.sample['Acoustic features']}</features>"
            f"<think>analysis</think>"
            f"<reasons>{self.sample['reasons']}</reasons>"
            f"<answer>{self.sample['gt_label']}</answer>"
        )
        score = judge.score(["a1"], [perfect], [gt])
        per = score["meta"]["per_sample"][0]
        self.assertTrue(per["format_ok"])
        self.assertTrue(per["is_correct"])
        self.assertEqual(per["correct_reasons_overlap"], 1.0)
        self.assertEqual(per["correct_features_overlap"], 1.0)
        self.assertAlmostEqual(per["score"], 1.0)

    def test_partial_features_overlap(self):
        gt_features = {"age": {"reference": "26-35", "query": "18-25"}}
        out_features = {"age": {"reference": "26-35", "query": "wrong"}}
        overlap = _compute_correct_features_overlap(gt_features, out_features)
        self.assertAlmostEqual(overlap, 0.25)

    def test_flatten_acoustic_features(self):
        flat = _flatten_acoustic_features({
            "gender": {"reference": "Male", "query": "male"},
        })
        self.assertIn(("gender", "reference", "male"), flat)
        self.assertIn(("gender", "query", "male"), flat)


class TestParseReasons(unittest.TestCase):
    def test_parse_reasons(self):
        self.assertEqual(_parse_reasons("<reasons>['A','B']</reasons>"), {"A", "B"})
        self.assertEqual(_parse_reasons("<reasons>[]</reasons>"), set())
        self.assertEqual(_parse_reasons("no tags"), set())


class TestComputeOverlap(unittest.TestCase):
    def test_overlap_metric(self):
        # 2 gt, 2 out, 1 correct -> (1/2)*(1/2)=0.25
        self.assertAlmostEqual(
            _compute_correct_reasons_overlap({"A", "B"}, {"A", "C"}), 0.25
        )
        # 2 gt, 2 out, 2 correct -> 1.0
        self.assertEqual(_compute_correct_reasons_overlap({"A", "B"}, {"A", "B"}), 1.0)
        # gt empty -> 0
        self.assertEqual(_compute_correct_reasons_overlap(set(), {"A"}), 0.0)
        # out empty -> 0
        self.assertEqual(_compute_correct_reasons_overlap({"A"}, set()), 0.0)


if __name__ == "__main__":
    unittest.main()


