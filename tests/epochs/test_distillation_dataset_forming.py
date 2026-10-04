"""Tests for DistillationDatasetFormingEpoch."""
import unittest
import os
import random
from unittest.mock import MagicMock, patch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


class TestDistillationDatasetFormingEpoch(unittest.TestCase):
    def test_filter_chunk_correctness_margins(self):
        """Test that filtering respects min/max correct margins."""
        from src.epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch

        model = MagicMock()
        model.generate = MagicMock(return_value=["<answer>Real</answer>"] * 6)
        model.count_tokens = MagicMock(return_value=50)
        if hasattr(model, "module"):
            del model.module
        dataloader = MagicMock()
        dataloader.dataset = MagicMock()
        dataloader.dataset.dataset = None
        dataloader.__iter__ = lambda s: iter([])

        logger = MagicMock()
        judge = MagicMock()
        judge.score = MagicMock(return_value={
            "meta": {
                "per_sample": [
                    {"is_correct": i % 2 == 0, "format_ok": True}
                    for i in range(6)
                ]
            }
        })

        config = {
            "num_generations_per_sample": 2,
            "filtering_chunk_size": 10,
            "max_dataset_forming_attempts": 1,
            "min_intermediate_dataset_size": 1,
            "Filtering": {
                "intermediate_dataset_forming": {
                    "min_correct": 1,
                    "max_correct": 1,
                    "min_text_length": 10,
                    "max_text_length": 1000,
                    "min_tokens_len": 5,
                    "max_tokens_len": 500,
                    "skeptical_presampling": {"enable": False},
                }
            },
            "GRPO": {"generation": {"max_new_tokens": 100}},
        }

        epoch = DistillationDatasetFormingEpoch(
            model, dataloader, logger, judge, config,
            device=torch.device("cpu"), amp=False
        )
        epoch.unwrapped_model = model

        batch = {
            "prompts": ["p1", "p2", "p3"],
            "text": ["t1", "t2", "t3"],
            "audio_ids": ["a1", "a2", "a3"],
        }
        accepted, lchars, ltokens, audio_ids, reasoning_texts, verdicts = epoch._filter_main_only(batch, 0)
        self.assertIsInstance(accepted, list)
        self.assertIsInstance(lchars, list)
        self.assertIsInstance(ltokens, list)
        self.assertIsInstance(audio_ids, list)
        self.assertIsInstance(reasoning_texts, list)
        self.assertIsNone(verdicts)  # save_presampling=False by default

    def test_skeptic_length_at_least_one_passes(self):
        """Skeptic length filter: at least ONE rollout must pass. Sample with 1 short + 1 long rollout should pass."""
        from src.epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch

        model = MagicMock()
        # Batch of 1 sample: skeptic phase 2 rollouts, main phase 2 rollouts
        model.generate = MagicMock(side_effect=[
            ["x" * 5],   # skeptic gen 0
            ["x" * 60],  # skeptic gen 1
            ["x" * 5],   # main gen 0
            ["x" * 60],  # main gen 1
        ])
        model.count_tokens = lambda t: len(t) // 4  # 5->1, 60->15 tokens

        dataloader = MagicMock()
        dataloader.dataset = MagicMock()
        dataloader.dataset.dataset = None
        dataloader.__iter__ = lambda s: iter([])

        logger = MagicMock()
        judge = MagicMock()
        # Skeptic: 2 correct. Main: 2 correct. Both correct so valid_texts has both.
        judge.score = MagicMock(return_value={
            "meta": {
                "per_sample": [
                    {"is_correct": True, "format_ok": True},   # rollout 0 (short)
                    {"is_correct": True, "format_ok": True},   # rollout 1 (long)
                ]
            }
        })

        config = {
            "num_generations_per_sample": 2,
            "filtering_chunk_size": 10,
            "max_dataset_forming_attempts": 1,
            "min_intermediate_dataset_size": 1,
            "Filtering": {
                "intermediate_dataset_forming": {
                    "min_correct": 1,
                    "max_correct": 2,
                    "min_text_length": 10,
                    "max_text_length": 1000,
                    "min_tokens_len": 5,
                    "max_tokens_len": 500,
                    "skeptical_presampling": {
                        "enable": True,
                        "num_generations": 2,
                        "min_correct": 1,
                        "max_correct": 2,
                        "min_text_length": 30,   # rollout 0 (5 chars) fails, rollout 1 (60) passes
                        "max_text_length": 100,
                        "min_tokens_len": 10,   # rollout 0 (1 tok) fails, rollout 1 (15) passes
                        "max_tokens_len": 100,
                    },
                }
            },
            "GRPO": {"generation": {"max_new_tokens": 100}},
        }

        epoch = DistillationDatasetFormingEpoch(
            model, dataloader, logger, judge, config,
            device=torch.device("cpu"), amp=False
        )
        epoch.unwrapped_model = model

        batch = {
            "prompts": ["p1"],
            "text": ["t1"],
            "audio_ids": ["a1"],
        }
        # Phase 1: skeptic filter
        accepted_skeptic = epoch._filter_skeptic_only(batch)
        self.assertEqual(len(accepted_skeptic), 1, "Skeptic: at least one rollout passes length")
        # Phase 2: main filter (candidate is the one that passed skeptic)
        accepted, lchars, ltokens, audio_ids, reasoning_texts, verdicts = epoch._filter_main_only(batch, 0)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(lchars[0], 60)
        self.assertEqual(ltokens[0], 15)
        self.assertEqual(audio_ids[0], "a1")
        self.assertIsNone(verdicts)  # save_presampling=False by default

    def test_main_filter_includes_all_good_rollouts_per_audio(self):
        """Main filter: when min_correct rollouts pass, ALL passing rollouts are included (not just one)."""
        from src.epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch

        model = MagicMock()
        # texts_for_log returns the raw generated text; we use different lengths so both pass length filter
        model.generate = MagicMock(side_effect=[
            ["x" * 50],   # rollout 0 - passes length
            ["y" * 60],   # rollout 1 - passes length
        ])
        model.count_tokens = lambda t: len(t) // 4  # 50->12, 60->15 tokens

        dataloader = MagicMock()
        dataloader.dataset = MagicMock()
        dataloader.dataset.dataset = None
        dataloader.__iter__ = lambda s: iter([])

        logger = MagicMock()
        judge = MagicMock()
        judge.score = MagicMock(return_value={
            "meta": {
                "per_sample": [
                    {"is_correct": True, "format_ok": True},
                    {"is_correct": True, "format_ok": True},
                ]
            }
        })

        config = {
            "num_generations_per_sample": 2,
            "Filtering": {
                "intermediate_dataset_forming": {
                    "min_correct": 2,
                    "max_correct": 2,
                    "min_text_length": 10,
                    "max_text_length": 1000,
                    "min_tokens_len": 5,
                    "max_tokens_len": 500,
                    "skeptical_presampling": {"enable": False},
                }
            },
            "GRPO": {"generation": {"max_new_tokens": 100}},
        }

        epoch = DistillationDatasetFormingEpoch(
            model, dataloader, logger, judge, config,
            device=torch.device("cpu"), amp=False
        )
        epoch.unwrapped_model = model

        batch = {
            "prompts": ["p1"],
            "text": ["t1"],
            "audio_ids": ["a1"],
        }
        accepted, lchars, ltokens, audio_ids, reasoning_texts, _ = epoch._filter_main_only(batch, 0)
        # Both rollouts pass: we should get 2 entries for the same audio
        self.assertEqual(len(accepted), 2, "Both good rollouts should be included")
        self.assertEqual(accepted, [0, 0])
        self.assertEqual(lchars, [50, 60])
        self.assertEqual(ltokens, [12, 15])
        self.assertEqual(audio_ids, ["a1", "a1"])
        self.assertEqual(len(reasoning_texts), 2)
        self.assertEqual(reasoning_texts[0], "x" * 50)
        self.assertEqual(reasoning_texts[1], "y" * 60)

    def test_save_presampling_verdicts_structure(self):
        """When save_presampling=True, verdicts contain audio_id, char_len, token_len, is_correct, verdict, reason, ratios."""
        from src.epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch

        model = MagicMock()
        model.generate = MagicMock(return_value=["<answer>Real</answer>"] * 4)  # 2 samples * 2 gens
        model.count_tokens = lambda t: len(t) // 4
        dataloader = MagicMock()
        dataloader.dataset = MagicMock()
        dataloader.dataset.dataset = None
        dataloader.__iter__ = lambda s: iter([])

        logger = MagicMock()
        judge = MagicMock()
        judge.score = MagicMock(return_value={
            "meta": {
                "per_sample": [
                    {"is_correct": True, "format_ok": True},
                    {"is_correct": True, "format_ok": True},
                    {"is_correct": True, "format_ok": True},
                    {"is_correct": True, "format_ok": True},
                ]
            }
        })

        config = {
            "num_generations_per_sample": 2,
            "Filtering": {
                "intermediate_dataset_forming": {
                    "min_correct": 1,
                    "max_correct": 2,
                    "min_text_length": 10,
                    "max_text_length": 1000,
                    "min_tokens_len": 5,
                    "max_tokens_len": 500,
                    "save_presampling": True,
                    "skeptical_presampling": {"enable": False},
                }
            },
            "GRPO": {"generation": {"max_new_tokens": 100}},
        }

        epoch = DistillationDatasetFormingEpoch(
            model, dataloader, logger, judge, config,
            device=torch.device("cpu"), amp=False
        )
        epoch.unwrapped_model = model

        batch = {
            "prompts": ["p1", "p2"],
            "text": ["t1", "t2"],
            "audio_ids": ["a1", "a2"],
        }
        accepted, lchars, ltokens, audio_ids, reasoning_texts, verdicts = epoch._filter_main_only(batch, 0)
        self.assertIsNotNone(verdicts)
        self.assertEqual(len(verdicts), 2)
        for v in verdicts:
            self.assertIn("audio_id", v)
            self.assertIn("char_len", v)
            self.assertIn("token_len", v)
            self.assertIn("n_correct", v)
            self.assertIn("is_correct", v)
            self.assertIn("verdict", v)
            self.assertIn("reason", v)
            self.assertIn("relative_text_length_ratio", v)
            self.assertIn("relative_token_length_ratio", v)
        self.assertEqual([v["verdict"] for v in verdicts], ["accepted", "accepted"])

    def test_parse_reasoning_output(self):
        """Parse <think>, <reasons>, <answer> from generated text for dataloader."""
        from src.epochs.distillation_dataset_forming_epoch import _parse_reasoning_output

        text = "<think>The audio sounds natural.</think><reasons>[\"STRANGE_VOICE\"]</reasons><answer>Fake</answer>"
        think, reasons, is_bonafide = _parse_reasoning_output(text)
        self.assertEqual(think, "The audio sounds natural.")
        self.assertEqual(reasons, ["STRANGE_VOICE"])
        self.assertFalse(is_bonafide)

        text2 = "<think></think><reasons>[]</reasons><answer>Real</answer>"
        think2, reasons2, is_bonafide2 = _parse_reasoning_output(text2)
        self.assertEqual(think2, "")
        self.assertEqual(reasons2, [])
        self.assertTrue(is_bonafide2)

        # Invented reasons are filtered out
        text3 = "<think>X</think><reasons>[\"STRANGE_VOICE\", \"INVENTED_REASON\", \"OTHER\"]</reasons><answer>Fake</answer>"
        think3, reasons3, _ = _parse_reasoning_output(text3)
        self.assertIn("STRANGE_VOICE", reasons3)
        self.assertIn("OTHER", reasons3)
        self.assertNotIn("INVENTED_REASON", reasons3)

    def test_build_reasoning_samples_parses_tags(self):
        """_build_reasoning_samples parses generated text into reasoning, reasons, is_bonafide."""
        from src.epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch, _parse_reasoning_output

        base_ds = MagicMock()
        base_ds.samples = [
            {"audio_id": "a1", "original_path": "/p1.wav", "is_bonafide": True, "reasons": None},
        ]
        config = {"Filtering": {"intermediate_dataset_forming": {}}, "GRPO": {"generation": {}}}
        epoch = DistillationDatasetFormingEpoch(MagicMock(), MagicMock(), MagicMock(), MagicMock(), config, device=torch.device("cpu"), amp=False)
        gen_text = "<think>Natural speech.</think><reasons>[]</reasons><answer>Real</answer>"
        samples = epoch._build_reasoning_samples(base_ds, [0], [gen_text])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["reasoning"], "Natural speech.")
        self.assertEqual(samples[0]["reasons"], None)
        self.assertTrue(samples[0]["is_bonafide"])

    def test_oom_handling_skeptic_splits_and_retries(self):
        """OOM in skeptic phase: split batch, retry sub-batches, combine accepted indices."""
        from src.epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch

        model = MagicMock()
        model.generate = MagicMock(return_value=["x" * 60] * 6)
        model.count_tokens = lambda t: len(t) // 4
        dataloader = MagicMock()
        dataloader.dataset = MagicMock()
        dataloader.dataset.dataset = None
        logger = MagicMock()
        judge = MagicMock()
        judge.score = MagicMock(return_value={
            "meta": {
                "per_sample": [
                    {"is_correct": True, "format_ok": True},
                    {"is_correct": True, "format_ok": True},
                    {"is_correct": True, "format_ok": True},
                ]
            }
        })
        config = {
            "num_generations_per_sample": 2,
            "Filtering": {"intermediate_dataset_forming": {"skeptical_presampling": {"enable": False}}},
            "GRPO": {"generation": {}},
        }
        epoch = DistillationDatasetFormingEpoch(
            model, dataloader, logger, judge, config,
            device=torch.device("cpu"), amp=False
        )
        epoch.unwrapped_model = model

        batch = {"prompts": ["p1", "p2", "p3"], "text": ["t1", "t2", "t3"], "audio_ids": ["a1", "a2", "a3"]}
        with patch.object(epoch, "_filter_skeptic_only") as mock_filter:
            def side_effect(b):
                sz = len(b.get("prompts", b.get("audio_ids", [1])))
                if sz >= 3:
                    raise torch.cuda.OutOfMemoryError("test OOM")
                return [0] if sz >= 1 else []

            mock_filter.side_effect = side_effect
            accepted = epoch._handle_out_of_memory_skeptic(batch, 3, 0, "skeptic")
            # Sub-batches [0:2] and [2:3]. First returns [0] -> mapped to [0], second returns [0] -> mapped to [2]
            self.assertEqual(accepted, [0, 2])

    def test_oom_handling_main_splits_and_retries(self):
        """OOM in main phase: split batch, retry sub-batches, combine results."""
        from src.epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch

        model = MagicMock()
        model.generate = MagicMock(return_value=["x" * 60] * 6)
        model.count_tokens = lambda t: len(t) // 4
        dataloader = MagicMock()
        dataloader.dataset = MagicMock()
        dataloader.dataset.dataset = None
        logger = MagicMock()
        judge = MagicMock()
        judge.score = MagicMock(return_value={
            "meta": {
                "per_sample": [
                    {"is_correct": True, "format_ok": True},
                    {"is_correct": True, "format_ok": True},
                    {"is_correct": True, "format_ok": True},
                ]
            }
        })
        config = {
            "num_generations_per_sample": 2,
            "Filtering": {"intermediate_dataset_forming": {}},
            "GRPO": {"generation": {}},
        }
        epoch = DistillationDatasetFormingEpoch(
            model, dataloader, logger, judge, config,
            device=torch.device("cpu"), amp=False
        )
        epoch.unwrapped_model = model

        batch = {"prompts": ["p1", "p2", "p3"], "text": ["t1", "t2", "t3"], "audio_ids": ["a1", "a2", "a3"]}
        with patch.object(epoch, "_filter_main_only") as mock_filter:
            def side_effect(b, dist_iter):
                sz = len(b.get("prompts", b.get("audio_ids", [1])))
                if sz >= 3:
                    raise torch.cuda.OutOfMemoryError("test OOM")
                return ([0], [60], [15], ["a1"], ["x" * 60], None) if sz >= 1 else ([], [], [], [], [], None)

            mock_filter.side_effect = side_effect
            accepted, lchars, ltokens, aids, reasoning, verdicts = epoch._handle_out_of_memory_main(
                batch, 3, 0, 0, "main"
            )
            self.assertIsInstance(accepted, list)
            self.assertIsInstance(lchars, list)
            self.assertIsInstance(ltokens, list)


def _run_skeptic_distributed_worker(rank: int, world_size: int, port: int = 29500) -> list:
    """Worker for distributed skeptic test. Returns candidate_indices."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from src.epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch

        class DummyDS(Dataset):
            def __len__(self):
                return 32

            def __getitem__(self, i):
                return {"prompts": [f"p{i}"], "text": [f"t{i}"], "audio_ids": [f"a{i}"]}

        model = MagicMock()
        model.generate = MagicMock(return_value=["<think>x</think><reasons>[]</reasons><answer>Real</answer>"] * 3)
        model.count_tokens = lambda t: 50
        if hasattr(model, "module"):
            del model.module

        loader = DataLoader(DummyDS(), batch_size=4, shuffle=False)
        logger = MagicMock()
        judge = MagicMock()
        judge.score = MagicMock(return_value={
            "meta": {"per_sample": [{"is_correct": True, "format_ok": True}] * 3},
        })

        config = {
            "num_generations_per_sample": 2,
            "filtering_chunk_size": 100,
            "max_dataset_forming_attempts": 1,
            "min_intermediate_dataset_size": 1,
            "Filtering": {"intermediate_dataset_forming": {
                "min_correct": 1, "max_correct": 2,
                "min_text_length": 5, "max_text_length": 1000,
                "min_tokens_len": 2, "max_tokens_len": 500,
                "skeptical_presampling": {
                    "enable": True,
                    "num_generations": 3,
                    "skeptic_chunk_size": 32,
                    "min_skeptic_dataset_size": 2,
                    "max_skeptic_attempts": 1,
                    "min_correct": 1, "max_correct": 2,
                    "min_text_length": 5, "max_text_length": 1000,
                    "min_tokens_len": 2, "max_tokens_len": 500,
                },
            }},
            "GRPO": {"generation": {"max_new_tokens": 10}},
        }

        epoch = DistillationDatasetFormingEpoch(
            model, loader, logger, judge, config,
            device=torch.device("cpu"), amp=False
        )
        epoch.unwrapped_model = model

        base_ds = loader.dataset
        total_size = len(base_ds)
        _is_main = dist.get_rank() == 0
        candidates, _, _ = epoch._run_skeptic_phase(base_ds, total_size, 0, _is_main)
        return candidates
    finally:
        dist.destroy_process_group()


def _run_skeptic_distributed_wrapper(rank: int, world_size: int, port: int, results_list):
    """Wrapper that appends result to shared list."""
    out = _run_skeptic_distributed_worker(rank, world_size, port)
    results_list.append((rank, out))


class TestDistillationDatasetFormingDistributed(unittest.TestCase):
    """Distributed tests for skeptic phase. Uses gloo backend (no GPUs required)."""

    def test_skeptic_phase_distributed_no_deadlock(self):
        """Run skeptic phase with 2 ranks; verify no deadlock and same result on both."""
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        world_size = 2
        port = 29500 + random.randint(0, 999)
        ctx = mp.get_context("spawn")
        results = ctx.Manager().list()

        procs = [
            ctx.Process(target=_run_skeptic_distributed_wrapper, args=(r, world_size, port, results))
            for r in range(world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=5)
            self.assertFalse(p.is_alive(), f"Process {p.pid} did not finish in 5s (deadlock?)")
            self.assertEqual(p.exitcode, 0, f"Process {p.pid} exited with code {p.exitcode}")

        self.assertEqual(len(results), 2)
        _, out0 = results[0]
        _, out1 = results[1]
        self.assertEqual(out0, out1, "Both ranks must return same candidate_indices")
        self.assertIsInstance(out0, list)
