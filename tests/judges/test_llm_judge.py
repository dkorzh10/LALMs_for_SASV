import unittest
from unittest.mock import MagicMock
from src.judges.llm_as_a_judge import LLMAsAJudge

class MockLLMJudge(LLMAsAJudge):
    def _generate(self, prompts):
        # Mock responses for 4 aspects
        return ["Detail: 8\nRelevance: 7\nLogic: 9\nHelpfulness: 10"] * len(prompts)

class TestLLMAsAJudge(unittest.TestCase):
    def test_scoring_logic(self):
        config = {
            "weights": {"correctness": 1.0, "format": 0.1, "judge": 1.0},
            "aspect_weights": {"detail": 0.25, "relevance": 0.25, "logic": 0.25, "helpfulness": 0.25}
        }
        judge = MockLLMJudge(config)
        
        inputs = ["audio1"]
        outputs = ["<think>...</think><reasons>[]</reasons><answer>bonafide</answer>"]
        gt = ["bonafide"]
        
        result = judge.score(inputs, outputs, gt)
        
        # correctness: 1.0 (matches)
        # format: 1.0 (all tags)
        # llm_score: (8+7+9+10)/40 = 34/40 = 0.85
        # total = 1.0*1.0 + 1.0*0.1 + 0.85*1.0 = 1.95
        
        self.assertAlmostEqual(result["meta"]["raw_scores"][0], 1.95)
        self.assertEqual(result["meta"]["correctness"][0], 1.0)
        self.assertEqual(result["meta"]["format"][0], 1.0)
        self.assertAlmostEqual(result["meta"]["llm_scores"][0], 0.85)

if __name__ == "__main__":
    unittest.main()


