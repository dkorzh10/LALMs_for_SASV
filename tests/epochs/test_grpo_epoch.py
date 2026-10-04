import unittest
from unittest.mock import MagicMock, patch, create_autospec
import torch
from src.epochs.grpo_epoch import GRPOTrainEpoch, _parse_answer_from_text
from src.models.dummy.model import DummyModel
from src.loggers.hard_label_logger import HardLabelLogger
from src.judges.format_judge import FormatJudge


def _fake_process_batch(batch, backward):
    """Lightweight mock return for _process_batch to avoid slow model ops."""
    b = len(batch.get("audio_ids", batch.get("audio_paths", [1])))
    if isinstance(b, list):
        b = len(b)
    return (
        torch.tensor(0.5),
        torch.tensor(0.01),
        torch.ones(b, 4) * 0.5,
        [{"generations": [{"reward": 0.5, "correct_reasons_overlap": 0.0} for _ in range(4)]} for _ in range(b)],
    )


class TestParseAnswer(unittest.TestCase):
    def test_parse_real(self):
        self.assertEqual(_parse_answer_from_text("x<answer>Real</answer>y"), "real")
        self.assertEqual(_parse_answer_from_text("<answer>real</answer>"), "real")

    def test_parse_fake(self):
        self.assertEqual(_parse_answer_from_text("<answer>Fake</answer>"), "fake")
        self.assertEqual(_parse_answer_from_text("<answer>fake</answer>"), "fake")

    def test_parse_none(self):
        self.assertIsNone(_parse_answer_from_text("no answer here"))
        self.assertIsNone(_parse_answer_from_text("<answer>unknown</answer>"))
        self.assertIsNone(_parse_answer_from_text("<answer>bonafide</answer>"))
        self.assertIsNone(_parse_answer_from_text("<answer>spoof</answer>"))


class TestGRPOTrainEpoch(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel({})
        from src.loggers.grpo_logger import GRPOLogger
        self.logger = create_autospec(GRPOLogger, instance=True)
        self.logger.iteration_num = 0
        self.logger.epoch_num = 0
        self.logger.epoch_type = "train"
        self.logger.log_freq = 1
        self.logger.losses = []
        self.optimizer = MagicMock(spec=torch.optim.Optimizer)
        self.optimizer.param_groups = [{"lr": 0.001}]
        self.judge = FormatJudge()
        self.dataloader = [
            {"audio_ids": ["1"], "audio_paths": ["p1"], "prompts": ["p1"], "raw_wav": torch.randn(1, 16000), "text": ["input"], "reasoning": ["gt"]}
        ]

    def test_run(self):
        epoch = GRPOTrainEpoch(self.model, self.dataloader, self.logger, self.optimizer, self.judge)
        with patch.object(epoch, "_process_batch", side_effect=_fake_process_batch):
            epoch.run(0)

        self.logger.set_epoch.assert_called_with(0, "train")
        self.assertTrue(self.logger.log.called)
        self.assertTrue(self.logger.log_epoch.called)

    def test_skeptic_filter_controversial(self):
        """With mock judge returning mix of correct/incorrect, samples are accepted as controversial."""
        def mock_score(inputs, outputs, gt):
            n = len(gt)
            # Check phase: 3 gens * batch_size (e.g. 6). Need mix of correct/incorrect for controversial.
            # GRPO phase: 5 gens * batch_size (e.g. 10). Any scores ok.
            num_gens = 3 if n % 3 == 0 and n <= 12 else 5
            batch_size = n // num_gens
            per_sample = []
            for g in range(num_gens):
                for b in range(batch_size):
                    is_correct = (num_gens == 5) or (g != 1)  # check: gen 0,2 correct; gen 1 incorrect
                    per_sample.append({
                        "score": 0.5, "format_ok": True, "is_correct": is_correct,
                        "correct_reasons_overlap": 0.0, "answer": "real",
                    })
            return {"score": 0.5, "meta": {"per_sample": per_sample}}

        mock_judge = MagicMock()
        mock_judge.score.side_effect = mock_score

        # Single-sample batches so we get 2 controversial total (one per batch)
        dataloader = [
            {"audio_ids": ["1"], "audio_paths": ["p1"], "prompts": ["p1"], "raw_wav": torch.randn(1, 16000), "text": ["input"], "reasoning": ["gt"]},
            {"audio_ids": ["2"], "audio_paths": ["p2"], "prompts": ["p2"], "raw_wav": torch.randn(1, 16000), "text": ["input"], "reasoning": ["gt"]},
        ]
        from src.loggers.grpo_logger import GRPOLogger
        logger = create_autospec(GRPOLogger, instance=True)
        logger.iteration_num = 0
        logger.epoch_num = 0
        logger.epoch_type = "train"
        logger.log_freq = 1
        logger.losses = []

        epoch = GRPOTrainEpoch(
            self.model, dataloader, logger, self.optimizer, mock_judge,
            num_generations=4, filter_controversial=True, skeptic_buffer_size=2,
        )
        process_mock = MagicMock(side_effect=lambda b, r, backward: _fake_process_batch(b, backward))
        with patch.object(epoch, "_process_batch_with_reused_rollouts", process_mock):
            epoch.run(0)

        logger.set_epoch.assert_called_with(0, "train")
        self.assertEqual(process_mock.call_count, 1)  # 1 GRPO batch (2 controversial samples)
        logger.set_skeptic_epoch_stats.assert_called_once()
        rejected, accepted = logger.set_skeptic_epoch_stats.call_args[0]
        self.assertEqual(accepted, 2)  # 2 batches, 1 sample each, all controversial
        self.assertEqual(rejected, 0)

    def test_skeptic_filter_all_same_rejected(self):
        """With mock judge returning all correct, samples are rejected."""
        def mock_score_all_correct(inputs, outputs, gt):
            num_gens = 3
            batch_size = len(gt) // num_gens
            per_sample = [{"score": 0.5, "format_ok": True, "is_correct": True, "correct_reasons_overlap": 0.0, "answer": "real"}
                         for _ in range(num_gens * batch_size)]
            return {"score": 0.5, "meta": {"per_sample": per_sample}}

        mock_judge = MagicMock()
        mock_judge.score.side_effect = mock_score_all_correct

        dataloader = [{"audio_ids": ["1"], "audio_paths": ["p1"], "prompts": ["p1"], "raw_wav": torch.randn(1, 16000), "text": ["input"], "reasoning": ["gt"]}]
        from src.loggers.grpo_logger import GRPOLogger
        logger = create_autospec(GRPOLogger, instance=True)
        logger.iteration_num = 0
        logger.epoch_num = 0
        logger.epoch_type = "train"
        logger.log_freq = 1
        logger.losses = []

        epoch = GRPOTrainEpoch(
            self.model, dataloader, logger, self.optimizer, mock_judge,
            num_generations=4, filter_controversial=True, skeptic_buffer_size=1,
        )
        with patch.object(epoch, "_process_batch", side_effect=_fake_process_batch):
            epoch.run(0)

        logger.set_skeptic_epoch_stats.assert_called_once()
        rejected, accepted = logger.set_skeptic_epoch_stats.call_args[0]
        self.assertEqual(rejected, 1)
        self.assertEqual(accepted, 0)

    def test_skeptic_generate_and_score_controversial(self):
        """Unit test: _skeptic_generate_and_score returns controversial when mix of correct/incorrect."""
        mock_model = MagicMock()

        def fake_generate(batch, gen_cfg, prompts=None, return_outputs=False, return_logits=True, **kwargs):
            n = int(gen_cfg.get("num_return_sequences", 1))
            bs = len(batch.get("prompts", [1]))
            total = bs * n
            return ["x"] * total, torch.zeros(total, 5), None

        mock_model.generate.side_effect = fake_generate
        mock_judge = MagicMock()
        mock_judge.score.return_value = {
            "meta": {
                "per_sample": [
                    {"is_correct": True},
                    {"is_correct": False},
                    {"is_correct": True},
                ]
            }
        }
        batch = {"prompts": ["p1"], "text": ["in"], "reasoning": ["gt"]}
        epoch = GRPOTrainEpoch(MagicMock(), [], MagicMock(), MagicMock(), mock_judge)
        epoch.unwrapped_model = mock_model
        epoch.device = torch.device("cpu")
        epoch.amp = False

        controversial, rejected, _, check_logs, reused_rollouts = epoch._skeptic_generate_and_score(batch)
        self.assertEqual(controversial, [0])
        self.assertEqual(rejected, [])
        self.assertEqual(len(check_logs), 1)
        self.assertIn("has_passed", check_logs[0])
        self.assertIn("rollouts", check_logs[0])
        self.assertEqual(len(reused_rollouts), 1)
        self.assertIn("texts", reused_rollouts[0])
        self.assertIn("judge_meta", reused_rollouts[0])
        self.assertIn("completion_ids", reused_rollouts[0])

    def test_skeptic_generate_and_score_all_same_rejected(self):
        """Unit test: _skeptic_generate_and_score returns rejected when all correct or all incorrect."""
        mock_model = MagicMock()

        def fake_generate(batch, gen_cfg, prompts=None, return_outputs=False, return_logits=True, **kwargs):
            n = int(gen_cfg.get("num_return_sequences", 1))
            bs = len(batch.get("prompts", [1]))
            total = bs * n
            return ["x"] * total, torch.zeros(total, 5), None

        mock_model.generate.side_effect = fake_generate
        mock_judge = MagicMock()
        mock_judge.score.return_value = {
            "meta": {"per_sample": [{"is_correct": True}, {"is_correct": True}, {"is_correct": True}]}
        }
        batch = {"prompts": ["p1"], "text": ["in"], "reasoning": ["gt"]}
        epoch = GRPOTrainEpoch(MagicMock(), [], MagicMock(), MagicMock(), mock_judge)
        epoch.unwrapped_model = mock_model
        epoch.device = torch.device("cpu")
        epoch.amp = False

        controversial, rejected, _, check_logs, reused_rollouts = epoch._skeptic_generate_and_score(batch)
        self.assertEqual(controversial, [])
        self.assertEqual(rejected, [0])
        self.assertEqual(check_logs[0]["has_passed"], False)
        self.assertEqual(len(reused_rollouts), 0)


if __name__ == "__main__":
    unittest.main()

