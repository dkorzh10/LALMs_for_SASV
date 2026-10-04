import argparse
import csv
import os
import json
import re
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, List, Optional, Union

from .utils.config import load_config
from .models.dummy.model import DummyModel

from .dataloaders.builder import get_dataloader

# Judges
from .judges.format_judge import FormatJudge
from .judges.openrouter_judge import OpenRouterJudge

from .trainers import SFTTrainer, GRPOTrainer, DistillationTrainer, SASVDistillationTrainer, SASVTrainer, SASVHardMiningTrainer

class Runner:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        # Setup Distributed Training
        self.num_gpus = self.config.get("Runner", {}).get("num_gpus", 1)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        if self.num_gpus > 1:
            if "RANK" in os.environ:
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
                torch.cuda.set_device(self.device)
            else:
                print(f"Warning: num_gpus={self.num_gpus} but distributed environment variables (RANK) not set. "
                      f"Falling back to single GPU training.", flush=True)
                self.num_gpus = 1

        self._preprocess_sasv_speaker_num_classes()
        self._preprocess_sasv_fusion_checkpoint()
        self._setup_output_dir()

        run_type = self.config.get("Runner", {}).get("type", "train")
        if self._is_fusion_gridsearch_run(run_type):
            self._fusion_gridsearch_mode = True
            self._score_calibration_mode = False
            self.model = None
            self.logger = None
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            if self.local_rank == 0:
                print(
                    f"[runner] Runner.type={run_type} — fusion gridsearch "
                    "(AASIST CM + ECAPA ASV cascade thresholds).",
                    flush=True,
                )
            return

        if self._is_score_calibration_run(run_type):
            self._fusion_gridsearch_mode = False
            self._score_calibration_mode = True
            self.model = None
            self.logger = None
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            if self.local_rank == 0:
                print(
                    f"[runner] Runner.type={run_type} — score calibration "
                    "(ECAPA + AASIST via CalibratedClassifierCV).",
                    flush=True,
                )
            return

        self._fusion_gridsearch_mode = False
        self._score_calibration_mode = False
        self.model = self._build_model()
        self._load_sasv_wrapper_checkpoint(self.model)
        self.model.to(self.device)
        
        # Set SASV class weights (yes/no higher, gen lower) if configured
        if hasattr(self.model, "model") and hasattr(self.model.model, "set_sasv_class_weights"):
            m = self.model.model
            if m.sasv_class_weights is not None and m.use_class_weights:
                applied = m.set_sasv_class_weights()
                if self.local_rank == 0:
                    print(f"Applied SASV class weights: {applied}", flush=True)

        if hasattr(self.model, "set_speaker_mapping") and getattr(self.model, "arcface_head", None) is not None:
            spk_ids = getattr(self, "_sasv_sorted_speaker_ids", [])
            if spk_ids:
                self.model.set_speaker_mapping(spk_ids)
            elif self.local_rank == 0:
                print(
                    "[SASV] Warning: no speaker IDs collected; ArcFace will fall back to "
                    "lazy per-run speaker indexing (not resume-stable).",
                    flush=True,
                )
        
        if self.num_gpus > 1:
            find_unused = self._ddp_find_unused_parameters()
            use_static_graph = self._ddp_use_static_graph()
            if self.local_rank == 0:
                print(
                    f"DDP find_unused_parameters={find_unused}, static_graph={use_static_graph}",
                    flush=True,
                )
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=find_unused,
            )
            if use_static_graph:
                self.model._set_static_graph()

        self.logger = self._build_logger()
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
        
    def cleanup(self):
        """Clean up distributed process group to avoid resource leaks."""
        if self.num_gpus > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def _setup_output_dir(self) -> None:
        """Create timestamped run directory and write config_resolved.yaml."""
        base_output_dir = self.config.get("General", {}).get("output_dir", "./outputs")
        timestamp_str = "unknown"
        if self.local_rank == 0:
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
        if self.num_gpus > 1:
            obj_list = [timestamp_str]
            dist.broadcast_object_list(obj_list, src=0)
            timestamp_str = obj_list[0]
        self.output_dir = os.path.join(base_output_dir, f"run_{timestamp_str}")
        if self.local_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            config_to_save = self._resolve_config_for_save()
            print("--- Run Configuration ---", flush=True)
            import yaml
            print(yaml.dump(config_to_save, default_flow_style=False), flush=True)
            with open(os.path.join(self.output_dir, "config_resolved.yaml"), "w") as f:
                yaml.dump(config_to_save, f, default_flow_style=False)
            print("-------------------------", flush=True)

    @staticmethod
    def _is_fusion_gridsearch_run(run_type: str) -> bool:
        from .utils.sasv_threshold_gridsearch import is_fusion_gridsearch_run_type

        return is_fusion_gridsearch_run_type(run_type)

    @staticmethod
    def _is_score_calibration_run(run_type: str) -> bool:
        from .utils.sasv_score_calibration import is_score_calibration_run_type

        return is_score_calibration_run_type(run_type)

    def _run_fusion_gridsearch(self) -> None:
        from .utils.sasv_threshold_gridsearch import run_threshold_gridsearch_from_config

        paths = run_threshold_gridsearch_from_config(
            self.config,
            self.output_dir,
            self.device,
            local_rank=self.local_rank,
        )
        if self.local_rank == 0 and paths:
            print(
                "[runner] Fusion gridsearch done. See best_thresholds.json:\n"
                f"  {paths.get('best_json')}",
                flush=True,
            )
        if self.num_gpus > 1 and dist.is_initialized():
            dist.barrier()

    def _run_score_calibration(self) -> None:
        from .utils.sasv_score_calibration import run_score_calibration_from_config

        paths = run_score_calibration_from_config(
            self.config,
            self.output_dir,
            self.device,
            local_rank=self.local_rank,
        )
        if self.local_rank == 0 and paths:
            print(
                "[runner] Score calibration done. See score_calibration_metrics.json:\n"
                f"  {paths.get('metrics_json')}",
                flush=True,
            )
        if self.num_gpus > 1 and dist.is_initialized():
            dist.barrier()

    def _resolve_config_for_save(self) -> Dict[str, Any]:
        """Return a copy of config with effective defaults merged in, so config_resolved.yaml
        always reflects what will actually be used (e.g. filter_controversial, skeptic_batch_size).
        """
        import copy
        cfg = copy.deepcopy(self.config)
        runner = cfg.setdefault("Runner", {})
        trainer_type = runner.get("trainer", "SFT")
        if trainer_type == "GRPO":
            grpo = runner.setdefault("GRPO", {})
            grpo.setdefault("filter_controversial", False)
            grpo.setdefault("skeptic_batch_size", None)
            grpo.setdefault("skeptic_buffer_size", None)
            grpo.setdefault("task_type", runner.get("SFT", {}).get("task_type", "hard_label"))
            grpo.setdefault("prompt_type", "reasoning")
            grpo.setdefault("answer_labels", "antispoofing")
        return cfg

    def _count_unique_speakers_in_dataset(self, dataset_path: str) -> Dict[str, int]:
        """Count unique speaker IDs from SASV metadata file.

        Supports:
        - JSON list of samples
        - JSONL samples
        Expected fields are speaker_id inside reference_audios/query_audios.
        """
        speaker_ids = set()
        num_samples = 0
        num_missing = 0

        def _consume_item(item: Dict[str, Any]):
            nonlocal num_samples, num_missing
            num_samples += 1
            found_in_item = False

            if isinstance(item.get("reference_audios"), list):
                for ref in item["reference_audios"]:
                    sid = ref.get("speaker_id", "") if isinstance(ref, dict) else ""
                    if sid:
                        speaker_ids.add(str(sid))
                        found_in_item = True

            if isinstance(item.get("query_audios"), list):
                for qry in item["query_audios"]:
                    sid = qry.get("speaker_id", "") if isinstance(qry, dict) else ""
                    if sid:
                        speaker_ids.add(str(sid))
                        found_in_item = True

            sid_root = item.get("speaker_id", "")
            if sid_root:
                speaker_ids.add(str(sid_root))
                found_in_item = True

            if not found_in_item:
                num_missing += 1

        if not dataset_path or not os.path.exists(dataset_path):
            return {"num_speakers": 0, "num_samples": 0, "num_missing": 0}

        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        _consume_item(item)
        else:
            with open(dataset_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        _consume_item(item)

        if len(speaker_ids) == 0:
            try:
                with open(dataset_path, "r") as f:
                    raw = f.read()
                text_sids = set(re.findall(r'"speaker_id"\s*:\s*"([^"]+)"', raw))
                if text_sids:
                    speaker_ids.update(text_sids)
            except Exception:
                pass

        speaker_sample = sorted(list(speaker_ids))[:5]
        return {
            "num_speakers": len(speaker_ids),
            "num_samples": num_samples,
            "num_missing": num_missing,
            "speaker_sample": speaker_sample,
        }

    def _collect_sorted_speaker_ids(self, dataset_path: str) -> List[str]:
        if not dataset_path or not os.path.exists(dataset_path):
            return []
        pat = re.compile(rb'"speaker_id"\s*:\s*"([^"]+)"')
        sids = set()
        tail = b""
        with open(dataset_path, "rb") as f:
            while True:
                chunk = f.read(1 << 24)
                if not chunk:
                    break
                buf = tail + chunk
                for m in pat.finditer(buf):
                    sids.add(m.group(1).decode("utf-8", "ignore"))
                tail = buf[-128:]
        return sorted(sids)

    def _preprocess_sasv_speaker_num_classes(self) -> None:
        self._sasv_sorted_speaker_ids: List[str] = []

        model_cfg = self.config.get("Model", {})
        if model_cfg.get("model_name") not in (
            "sasv_salmon", "new_salmon", "new_salmon_fusion",
        ):
            return

        salmon_cfg = model_cfg.setdefault("additional_kwargs", {}).setdefault("salmon", {})
        data_cfg = self.config.get("Datasets", {})
        train_path = data_cfg.get("dataset_train_path", "")

        sorted_ids: List[str] = []
        if self.local_rank == 0:
            try:
                sorted_ids = self._collect_sorted_speaker_ids(train_path)
            except Exception as e:
                print(f"[SASV preprocess] failed to collect speaker_ids: {e}", flush=True)

        if self.num_gpus > 1 and dist.is_initialized():
            obj_list = [sorted_ids]
            dist.broadcast_object_list(obj_list, src=0)
            sorted_ids = obj_list[0]

        self._sasv_sorted_speaker_ids = sorted_ids
        num_speakers = len(sorted_ids)

        explicit_num = salmon_cfg.get("speaker_num_classes")
        required = num_speakers + 1 if num_speakers > 0 else None

        if explicit_num is not None and int(explicit_num) > 0:
            if required is not None and int(explicit_num) < required:
                if self.local_rank == 0:
                    print(
                        f"[SASV preprocess] explicit speaker_num_classes={int(explicit_num)} "
                        f"< {required} (num_speakers+1); bumping to {required}.",
                        flush=True,
                    )
                salmon_cfg["speaker_num_classes"] = required
        elif required is not None:
            salmon_cfg["speaker_num_classes"] = required

        if self.local_rank == 0:
            print(
                "[SASV preprocess] speaker_num_classes="
                f"{salmon_cfg.get('speaker_num_classes', 0)} "
                f"(num unique speakers={num_speakers}, +1 spoof class)",
                flush=True,
            )
            if sorted_ids:
                print(f"[SASV preprocess] sample speaker_ids: {sorted_ids[:5]}", flush=True)

    def _sasv_fusion_mode(self) -> str:
        model_name = self.config.get("Model", {}).get("model_name", "")
        if model_name not in ("sasv_ecapa_w2v_aasist", "sasv_w2v_aasist"):
            return "mlp"
        ak = self.config.get("Model", {}).get("additional_kwargs", {})
        section_key = "sasv_baseline" if model_name == "sasv_ecapa_w2v_aasist" else "w2v_aasist"
        return str(ak.get(section_key, {}).get("fusion_mode", "mlp")).lower()

    def _preprocess_sasv_fusion_checkpoint(self) -> None:
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name", "")
        if model_name not in ("sasv_ecapa_w2v_aasist", "sasv_w2v_aasist"):
            return

        run_type = self.config.get("Runner", {}).get("type", "train")
        fusion_mode = self._sasv_fusion_mode()
        ak = model_cfg.setdefault("additional_kwargs", {})
        section_key = "sasv_baseline" if model_name == "sasv_ecapa_w2v_aasist" else "w2v_aasist"
        baseline_cfg = ak.setdefault(section_key, {})
        fusion_ckpt = str(baseline_cfg.get("fusion_ckpt", "") or "").strip()
        fusion_tree = str(baseline_cfg.get("fusion_tree_path", "") or "").strip()

        if run_type == "train":
            if fusion_mode == "tree":
                baseline_cfg["fusion_ckpt"] = ""
                if self.local_rank == 0:
                    print(
                        "[SASV preprocess] fusion_mode=tree — will train sklearn decision tree "
                        "(fusion_ckpt ignored).",
                        flush=True,
                    )
                return
            if fusion_ckpt and self.local_rank == 0:
                print(
                    f"[SASV preprocess] Runner.type=train — ignoring fusion_ckpt "
                    f"({fusion_ckpt}); fusion MLP starts from scratch.",
                    flush=True,
                )
            baseline_cfg["fusion_ckpt"] = ""
            return

        if run_type == "test":
            if fusion_mode == "tree":
                if fusion_tree and self.local_rank == 0:
                    if os.path.isfile(fusion_tree):
                        print(
                            f"[SASV preprocess] fusion_mode=tree — will load {fusion_tree}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[SASV preprocess] Warning: fusion_tree_path not found: {fusion_tree}",
                            flush=True,
                        )
                return
            if fusion_ckpt and self.local_rank == 0:
                if os.path.isfile(fusion_ckpt):
                    print(
                        f"[SASV preprocess] Runner.type=test — will load fusion_ckpt: {fusion_ckpt}",
                        flush=True,
                    )
                else:
                    print(
                        f"[SASV preprocess] Warning: fusion_ckpt not found: {fusion_ckpt}",
                        flush=True,
                    )

    def _salmon_extra_cfg(self) -> Dict[str, Any]:
        return self.config.get("Model", {}).get("additional_kwargs", {}).get("salmon", {})

    def _ddp_use_static_graph(self) -> bool:
        """DDP static graph: required with Llama gradient checkpointing (see salmon_hierarchical)."""
        if "ddp_static_graph" in self.config.get("Model", {}):
            return bool(self.config["Model"]["ddp_static_graph"])
        model_name = self.config.get("Model", {}).get("model_name", "")
        if model_name not in ("new_salmon", "new_salmon_fusion", "sasv_salmon", "salmon"):
            return False
        return bool(self._salmon_extra_cfg().get("llama_gradient_checkpointing", True))

    def _ddp_find_unused_parameters(self) -> bool:
        """Track unused params for conditional SASV heads / multi-encoder paths."""
        model_cfg = self.config.get("Model", {})
        if "ddp_find_unused_parameters" in model_cfg:
            return bool(model_cfg["ddp_find_unused_parameters"])
        model_name = model_cfg.get("model_name", "")
        if model_name not in ("new_salmon", "new_salmon_fusion", "sasv_salmon", "salmon", "sasv_w2v_aasist", "sasv_ecapa_w2v_aasist"):
            return False
        if model_name in ("sasv_w2v_aasist", "sasv_ecapa_w2v_aasist"):
            return False
        if self._ddp_use_static_graph():
            return False
        return True

    def _build_model(self):
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name")
        if model_name == "dummy":
            return DummyModel(model_cfg)
        elif model_name == "salmon":
            from .models.salmon import SalmonModel
            return SalmonModel(model_cfg)
        elif model_name == "new_salmon":
            from .models.salmon_hierarchical import SalmonHierarchicalModel
            return SalmonHierarchicalModel(model_cfg)
        elif model_name == "new_salmon_fusion":
            from .models.salmon_hierarchical_fusion import SalmonHierarchicalFusionModel
            return SalmonHierarchicalFusionModel(model_cfg)
        elif model_name == "sasv_salmon":
            from .models.sasv_salmon import SASVSalmonModel
            return SASVSalmonModel(model_cfg)
        elif model_name == "qwen_audio":
            from .models.qwen_audio import QwenAudioModel
            return QwenAudioModel(model_cfg)
        elif model_name == "sasv_w2v_aasist":
            from .models.sasv_w2v_aasist_fusion import SASVW2vAasistFusionModel
            return SASVW2vAasistFusionModel(model_cfg)
        elif model_name == "sasv_ecapa_w2v_aasist":
            from .models.sasv_ecapa_w2v_aasist_fusion import SASVEcapaW2vAasistFusionModel
            return SASVEcapaW2vAasistFusionModel(model_cfg)
        raise ValueError(f"Unknown model name: {model_name}")

    def _load_sasv_wrapper_checkpoint(self, model: torch.nn.Module) -> None:
        """Apply full ``checkpoint['model']`` to SASV SALMONN wrapper modules.

        ``SALMONN.from_config`` only loads keys prefixed with ``model.`` into the inner
        SALMONN. Outer modules (``answer_head``, ``fusion_head``, ``speaker_projector``,
        ``arcface_head``, ``bonafide_spoof_head``, …) stay randomly initialized on a
        fresh process unless we load the full state dict saved by the trainer.
        """
        model_cfg = self.config.get("Model", {})
        ckpt_path = model_cfg.get("ckpt", "")
        model_name = model_cfg.get("model_name", "")
        if not ckpt_path or not isinstance(ckpt_path, str):
            return
        if not os.path.isfile(ckpt_path):
            return
        if model_name not in ("new_salmon", "new_salmon_fusion", "sasv_salmon"):
            return

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            if self.local_rank == 0:
                print(
                    f"[checkpoint] No 'model' state dict in {ckpt_path}; "
                    "skipping full wrapper load.",
                    flush=True,
                )
            return

        model_sd = model.state_dict()
        shape_mismatch = [
            k for k, v in state.items()
            if k in model_sd and hasattr(v, "shape") and tuple(v.shape) != tuple(model_sd[k].shape)
        ]
        if shape_mismatch:
            state = {k: v for k, v in state.items() if k not in shape_mismatch}

        incompatible = model.load_state_dict(state, strict=False)
        if self.local_rank == 0:
            n_missing = len(getattr(incompatible, "missing_keys", ()))
            n_unexpected = len(getattr(incompatible, "unexpected_keys", ()))
            print(
                f"[checkpoint] Full wrapper load from {ckpt_path} "
                f"(missing_keys={n_missing}, unexpected_keys={n_unexpected}, "
                f"shape_mismatch_skipped={len(shape_mismatch)})",
                flush=True,
            )
            if shape_mismatch:
                print(f"[checkpoint] Re-initialized (shape changed): {shape_mismatch}", flush=True)

    def _build_logger(self):
        logger_cfg = self.config.get("Logger", {})
        log_dir = os.path.join(self.output_dir, "logs")
        log_freq = logger_cfg.get("log_freq", 10)
        save_all_predictions = logger_cfg.get("save_all_predictions", True)
        
        runner_cfg = self.config.get("Runner", {})
        trainer_type = runner_cfg.get("trainer", "SFT")
        
        is_main_process = (self.local_rank == 0)

        if trainer_type == "GRPO":
            from .loggers.grpo_logger import GRPOLogger
            return GRPOLogger(
                log_dir, log_freq if is_main_process else 1000000,
                save_all_predictions=save_all_predictions,
            )
        if trainer_type in ("Distillation", "SASVDistillation"):
            from .loggers.distillation_logger import DistillationLogger
            return DistillationLogger(log_dir, log_freq if is_main_process else 1000000)

        task_type = runner_cfg.get("SFT", {}).get("task_type", "hard_label")
        if task_type == "reasoning":
            from .loggers.reasoning_logger import ReasoningLogger
            return ReasoningLogger(log_dir, log_freq if is_main_process else 1000000, save_all_predictions=save_all_predictions)
        from .loggers.hard_label_logger import HardLabelLogger
        return HardLabelLogger(log_dir, log_freq if is_main_process else 1000000, save_all_predictions=save_all_predictions)

    def _get_prompt_templates(self, task_type: str) -> Optional[List[str]]:
        """Load prompt templates from config based on task type.
        
        Supports two formats:
        - List of strings directly in config
        - Path to JSON file containing list of prompts (or dict with 'antispoofing' key)
        """
        prompts_cfg = self.config.get("Prompts", {})
        key = "reasoning_prompts" if task_type == "reasoning" else "hard_label_prompts"
        value = prompts_cfg.get(key)
        
        if value is None:
            return None
        
        # If it's already a list, return it
        if isinstance(value, list):
            return value
        
        # If it's a string, treat as file path and load JSON
        if isinstance(value, str):
            import json
            prompt_path = value
            if not os.path.isabs(prompt_path):
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), os.pardir)
                )
                candidate = os.path.join(project_root, prompt_path)
                if os.path.isfile(candidate):
                    prompt_path = candidate
            try:
                with open(prompt_path, 'r') as f:
                    prompts = json.load(f)
                    # Handle both list format and dict with keys like 'antispoofing' or 'sasv'
                    if isinstance(prompts, list):
                        return prompts
                    if isinstance(prompts, dict):
                        # Check for common keys in order of preference
                        for key in ["sasv", "antispoofing"]:
                            if key in prompts:
                                return prompts[key]
                        # If no known key found, return first value if dict has only one key
                        if len(prompts) == 1:
                            return list(prompts.values())[0]
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load prompts from {prompt_path}: {e}", flush=True)
        
        return None

    def _build_dataloaders(self):
        data_cfg = self.config.get("Datasets", {})
        runner_cfg = self.config.get("Runner", {})
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name")
        
        whisper_path = None
        model_path = None
        silence_delay_seconds = 0.5  # Default value
        if model_name in (
            "salmon", "sasv_salmon", "salmon_hierarchical", "new_salmon",
            "new_salmon_fusion", "sasv_w2v_aasist", "sasv_ecapa_w2v_aasist",
        ):
            salmon_cfg = model_cfg.get("additional_kwargs", {}).get("salmon", {})
            whisper_path = salmon_cfg.get("pretrained_ckpts", {}).get("whisper_path")
            # Get configurable silence delay for concatenating multiple audios
            silence_delay_seconds = salmon_cfg.get("silence_delay_seconds", 0.5)
        elif model_name == "qwen_audio":
            model_path = model_cfg.get("additional_kwargs", {}).get("qwen_audio", {}).get("model_path")
        
        trainer_type = runner_cfg.get("trainer", "SFT")
        iters_per_epoch = None
        iters_per_epoch_val = None
        train_iters_per_epoch = None  # for dataloader; may differ in skeptic mode
        accum_grad_iters = 1
        
        if trainer_type == "GRPO":
            grpo_cfg = runner_cfg.get("GRPO", {})
            task_type = grpo_cfg.get(
                "task_type",
                runner_cfg.get("SFT", {}).get("task_type", "hard_label"),
            )
            prompt_type = grpo_cfg.get("prompt_type", "reasoning")
            iters_per_epoch = grpo_cfg.get("iters_per_epoch")
            iters_per_epoch_val = grpo_cfg.get("iters_per_epoch_val")
            accum_grad_iters = grpo_cfg.get("accum_grad_iters", 1)
            # Skeptic mode: epoch = GRPO batches, so we need enough dataloader batches to reach that.
            # Pass None so sampler yields full dataset; we stop when grpo_iter >= iters_per_epoch.
            train_iters_per_epoch = None if grpo_cfg.get("filter_controversial") else iters_per_epoch
        elif trainer_type == "Distillation":
            task_type = runner_cfg.get("SFT", {}).get("task_type", "reasoning")
            if "GRPO" in runner_cfg:
                task_type = "reasoning"  # Distillation uses GRPO flow
            sft_cfg = runner_cfg.get("SFT", {})
            iters_per_epoch = sft_cfg.get("iters_per_epoch")
            iters_per_epoch_val = sft_cfg.get("iters_per_epoch_val")
            accum_grad_iters = sft_cfg.get("accum_grad_iters", 1)
            train_iters_per_epoch = iters_per_epoch
        elif trainer_type == "SASVDistillation":
            task_type = runner_cfg.get("SFT", {}).get("task_type", "hard_label")
            sft_cfg = runner_cfg.get("SFT", {})
            iters_per_epoch = sft_cfg.get("iters_per_epoch")
            iters_per_epoch_val = sft_cfg.get("iters_per_epoch_val")
            accum_grad_iters = sft_cfg.get("accum_grad_iters", 1)
            train_iters_per_epoch = iters_per_epoch
        else:
            task_type = runner_cfg.get("SFT", {}).get("task_type", "hard_label")
            iters_per_epoch = runner_cfg.get("SFT", {}).get("iters_per_epoch")
            iters_per_epoch_val = runner_cfg.get("SFT", {}).get("iters_per_epoch_val")
            accum_grad_iters = runner_cfg.get("SFT", {}).get("accum_grad_iters", 1)
            train_iters_per_epoch = iters_per_epoch

        # Use batch sizes from the active trainer (configs often have both SFT and GRPO sections)
        if trainer_type == "GRPO" and "GRPO" in runner_cfg:
            batch_size_train = runner_cfg["GRPO"].get("batch_size_train", 1)
            batch_size_eval = runner_cfg["GRPO"].get("batch_size_eval", 1)
        elif trainer_type == "Distillation" and "SFT" in runner_cfg:
            batch_size_train = runner_cfg["SFT"].get("batch_size_train", 1)
            batch_size_eval = runner_cfg["SFT"].get("batch_size_eval", 1)
        elif trainer_type == "SASVDistillation" and "SFT" in runner_cfg:
            batch_size_train = runner_cfg["SFT"].get("batch_size_train", 1)
            batch_size_eval = runner_cfg["SFT"].get("batch_size_eval", 1)
        elif "SFT" in runner_cfg:
            batch_size_train = runner_cfg["SFT"].get("batch_size_train", 1)
            batch_size_eval = runner_cfg["SFT"].get("batch_size_eval", 1)
        else:
            batch_size_train = 1
            batch_size_eval = 1
        
        # Load prompt templates based on task type (GRPO can use reasoning prompts with hard-label GT)
        if trainer_type == "GRPO":
            prompt_templates = self._get_prompt_templates(
                runner_cfg.get("GRPO", {}).get("prompt_type", "reasoning")
            )
        else:
            prompt_templates = self._get_prompt_templates(task_type)
            
        # train_samples_offset with backward compat for samples_offset
        train_samples_offset = data_cfg.get("train_samples_offset", data_cfg.get("samples_offset", 0))
        val_samples_offset = data_cfg.get("val_samples_offset", 0)
        test_samples_offset = data_cfg.get("test_samples_offset", 0)

        audio_cfg = self.config.get("Audio", {})
        reasoning_version = data_cfg.get("reasoning_version", "long")

        train_loader = get_dataloader(
            data_cfg.get("dataset_train_path", "dummy"), 
            batch_size_train, 
            shuffle=data_cfg.get("shuffle", False), 
            max_samples=data_cfg.get("max_train_samples"),
            task_type=task_type,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            iters_per_epoch=train_iters_per_epoch,
            prompt_templates=prompt_templates,
            samples_offset=train_samples_offset,
            silence_delay_seconds=silence_delay_seconds,
            audio_cfg=audio_cfg,
            is_train=True,
            split="train",
            reasoning_version=reasoning_version,
        )
        val_loader = get_dataloader(
            data_cfg.get("dataset_val_path", "dummy"), 
            batch_size_eval, 
            shuffle=False, 
            max_samples=data_cfg.get("max_valid_samples"),
            task_type=task_type,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            iters_per_epoch=iters_per_epoch_val,
            prompt_templates=prompt_templates,
            samples_offset=val_samples_offset,
            silence_delay_seconds=silence_delay_seconds,
            audio_cfg=audio_cfg,
            is_train=False,
            split="val",
            reasoning_version=reasoning_version,
        )
        test_loader = get_dataloader(
            data_cfg.get("dataset_test_path", "dummy"), 
            batch_size_eval, 
            shuffle=False, 
            max_samples=data_cfg.get("max_test_samples"),
            task_type=task_type,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            prompt_templates=prompt_templates,
            samples_offset=test_samples_offset,
            silence_delay_seconds=silence_delay_seconds,
            audio_cfg=audio_cfg,
            is_train=False,
            split="test",
            reasoning_version=reasoning_version,
        )
        return train_loader, val_loader, test_loader

    def _create_dataloader_from_path(
        self, path: str, task_type: str = "reasoning"
    ) -> DataLoader:
        """Create a dataloader from a dataset JSON path (e.g. intermediate_dataset_iter_N.json)."""
        data_cfg = self.config.get("Datasets", {})
        runner_cfg = self.config.get("Runner", {})
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name")
        whisper_path = None
        model_path = None
        silence_delay_seconds = 0.5
        if model_name in (
            "salmon", "sasv_salmon", "salmon_hierarchical", "new_salmon",
            "new_salmon_fusion", "sasv_w2v_aasist", "sasv_ecapa_w2v_aasist",
        ):
            salmon_cfg = model_cfg.get("additional_kwargs", {}).get("salmon", {})
            whisper_path = salmon_cfg.get("pretrained_ckpts", {}).get("whisper_path")
            silence_delay_seconds = salmon_cfg.get("silence_delay_seconds", 0.5)
        elif model_name == "qwen_audio":
            model_path = model_cfg.get("additional_kwargs", {}).get("qwen_audio", {}).get("model_path")
        batch_size_train = runner_cfg.get("SFT", {}).get("batch_size_train", 1)
        prompt_templates = self._get_prompt_templates(task_type)
        audio_cfg = self.config.get("Audio", {})
        reasoning_version = data_cfg.get("reasoning_version", "long")
        return get_dataloader(
            path,
            batch_size_train,
            shuffle=data_cfg.get("shuffle", False),
            max_samples=None,
            task_type=task_type,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            iters_per_epoch=runner_cfg.get("SFT", {}).get("iters_per_epoch"),
            prompt_templates=prompt_templates,
            samples_offset=0,
            silence_delay_seconds=silence_delay_seconds,
            audio_cfg=audio_cfg,
            is_train=True,
            reasoning_version=reasoning_version,
        )

    def _build_judge(self, grpo_cfg):
        judge_type = grpo_cfg.get("judge", "format")
        if judge_type == "format":
            judge_cfg = dict(grpo_cfg)
            if "answer_labels" not in judge_cfg and grpo_cfg.get("task_type") == "hard_label":
                prompts_cfg = self.config.get("Prompts", {})
                if prompts_cfg.get("hard_label_prompts") and "sasv" in str(
                    prompts_cfg.get("hard_label_prompts", "")
                ).lower():
                    judge_cfg.setdefault("answer_labels", "sasv")
            return FormatJudge(judge_cfg)
        elif judge_type == "openrouter":
            return OpenRouterJudge(grpo_cfg.get("LLM_judge", {}))
        elif judge_type == "local_llm":
            from .judges.local_llm_judge import LocalLLMJudge
            return LocalLLMJudge(grpo_cfg.get("LLM_judge", {}))
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

    def run(self):
        if getattr(self, "_fusion_gridsearch_mode", False):
            self._run_fusion_gridsearch()
            return

        if getattr(self, "_score_calibration_mode", False):
            self._run_score_calibration()
            return

        runner_cfg = self.config.get("Runner", {})
        run_type = runner_cfg.get("type", "train")

        if run_type == "train":
            trainer_type = runner_cfg.get("trainer", "SFT")
            amp_enabled = self.config.get("Model", {}).get("amp", True)
            dev_meta_path = self.config.get("Datasets", {}).get("dataset_val_path", "")
            
            if trainer_type == "SFT":
                sft_cfg = runner_cfg.get("SFT", {})
                trainer_config = {
                    **sft_cfg, 
                    "lr": sft_cfg.get("optimizator", {}).get("init_lr", 1e-4), 
                    "amp": amp_enabled,
                    "iters_per_epoch": sft_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": sft_cfg.get("accum_grad_iters", 1),
                    "meta_path": dev_meta_path,
                }
                trainer = SFTTrainer(trainer_config, self.model, self.train_loader, self.val_loader, self.logger, device=self.device, output_dir=self.output_dir)
                trainer.train()
                
            elif trainer_type == "GRPO":
                grpo_cfg = runner_cfg.get("GRPO", {})
                judge = self._build_judge(grpo_cfg)
                trainer_config = {
                    **grpo_cfg, 
                    "lr": grpo_cfg.get("optimizator", {}).get("init_lr", 1e-4), 
                    "amp": amp_enabled,
                    "iters_per_epoch": grpo_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": grpo_cfg.get("accum_grad_iters", 1),
                    "filter_controversial": grpo_cfg.get("filter_controversial", False),
                    "skeptic_batch_size": grpo_cfg.get("skeptic_batch_size"),
                    "skeptic_buffer_size": grpo_cfg.get("skeptic_buffer_size"),
                    "meta_path": dev_meta_path,
                }
                trainer = GRPOTrainer(trainer_config, self.model, self.train_loader, self.val_loader, self.logger, judge, device=self.device, output_dir=self.output_dir)
                resume_path = grpo_cfg.get("resume")
                if resume_path:
                    trainer.load_checkpoint(resume_path)
                trainer.train()

            elif trainer_type == "Distillation":
                distill_cfg = runner_cfg.get("Distillation", {})
                sft_cfg = runner_cfg.get("SFT", {})
                if not sft_cfg:
                    sft_cfg = {"optimizator": {"init_lr": 1e-4}, "num_epochs": 1}
                sft_trainer_config = {
                    **sft_cfg,
                    "lr": sft_cfg.get("optimizator", {}).get("init_lr", 1e-4),
                    "amp": amp_enabled,
                    "iters_per_epoch": sft_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": sft_cfg.get("accum_grad_iters", 1),
                    "meta_path": dev_meta_path,
                }
                sft_trainer = SFTTrainer(
                    sft_trainer_config, self.model, self.train_loader, self.val_loader,
                    self.logger, device=self.device, output_dir=self.output_dir
                )

                grpo_cfg = runner_cfg.get("GRPO", {})
                if not grpo_cfg:
                    grpo_cfg = {"optimizator": {"init_lr": 1e-4}, "num_epochs": 1, "judge": "format"}
                judge = self._build_judge(grpo_cfg)
                grpo_trainer_config = {
                    **grpo_cfg,
                    "lr": grpo_cfg.get("optimizator", {}).get("init_lr", 1e-4),
                    "amp": amp_enabled,
                    "filter_controversial": grpo_cfg.get("filter_controversial", False),
                    "skeptic_batch_size": grpo_cfg.get("skeptic_batch_size"),
                    "skeptic_buffer_size": grpo_cfg.get("skeptic_buffer_size"),
                    "meta_path": dev_meta_path,
                }
                grpo_trainer = GRPOTrainer(
                    grpo_trainer_config, self.model, self.train_loader, self.val_loader,
                    self.logger, judge, device=self.device, output_dir=self.output_dir
                )

                dataset_forming_epoch = None
                if distill_cfg.get("Filtering"):
                    from .epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch
                    from .trainers.distillation_trainer import create_forming_loader
                    forming_cfg = {
                        **distill_cfg,
                        "GRPO": grpo_cfg,
                    }
                    forming_batch_size = (
                        distill_cfg.get("Filtering", {})
                        .get("intermediate_dataset_forming", {})
                        .get("batch_size")
                    )
                    forming_loader = (
                        create_forming_loader(self.train_loader, forming_batch_size)
                        if forming_batch_size is not None
                        else self.train_loader
                    )
                    dataset_forming_epoch = DistillationDatasetFormingEpoch(
                        self.model, forming_loader, self.logger, judge, forming_cfg,
                        device=self.device, amp=amp_enabled
                    )

                create_dataloader_from_path = (
                    lambda p: self._create_dataloader_from_path(p, task_type="reasoning")
                )
                trainer = DistillationTrainer(
                    distill_cfg, sft_trainer, grpo_trainer,
                    dataset_forming_epoch=dataset_forming_epoch,
                    initial_train_loader=self.train_loader,
                    create_dataloader_from_path=create_dataloader_from_path,
                )
                trainer.train()

            elif trainer_type == "SASVDistillation":
                distill_cfg = runner_cfg.get("SASVDistillation", {})
                sft_cfg = runner_cfg.get("SFT", {})
                if not sft_cfg:
                    sft_cfg = {"optimizator": {"init_lr": 1e-4}, "num_epochs": 1}
                sft_trainer_config = {
                    **sft_cfg,
                    "lr": sft_cfg.get("optimizator", {}).get("init_lr", 1e-4),
                    "amp": amp_enabled,
                    "iters_per_epoch": sft_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": sft_cfg.get("accum_grad_iters", 1),
                    "meta_path": dev_meta_path,
                }
                sft_trainer = SFTTrainer(
                    sft_trainer_config, self.model, self.train_loader, self.val_loader,
                    self.logger, device=self.device, output_dir=self.output_dir
                )

                from .epochs.sasv_distillation_dataset_forming_epoch import SASVDistillationDatasetFormingEpoch
                from .trainers.sasv_distillation_trainer import create_forming_loader

                forming_batch_size = (
                    distill_cfg.get("Filtering", {}).get("batch_size")
                )
                forming_loader = (
                    create_forming_loader(self.train_loader, forming_batch_size)
                    if forming_batch_size is not None
                    else self.train_loader
                )

                dataset_forming_epoch = SASVDistillationDatasetFormingEpoch(
                    self.model, forming_loader, self.logger, distill_cfg,
                    device=self.device, amp=amp_enabled,
                )

                trainer = SASVDistillationTrainer(
                    distill_cfg, sft_trainer,
                    dataset_forming_epoch=dataset_forming_epoch,
                    initial_train_loader=self.train_loader,
                )
                trainer.train()

            elif trainer_type == "SASV":
                if self._sasv_fusion_mode() == "tree":
                    from .utils.sasv_tree_training import train_decision_tree_from_config

                    tree_path = train_decision_tree_from_config(
                        self.config,
                        self.output_dir,
                        self.device,
                        local_rank=self.local_rank,
                    )
                    if self.local_rank == 0:
                        print(
                            f"[runner] Decision tree training done. "
                            f"For test set fusion_tree_path: {tree_path}",
                            flush=True,
                        )
                    return

                if not any(p.requires_grad for p in self.model.parameters()):
                    if self.local_rank == 0:
                        model_name = self.config.get("Model", {}).get("model_name", "")
                        print(
                            "[runner] Model has no trainable parameters "
                            f"(model_name={model_name!r}; for sasv_w2v_aasist / sasv_ecapa_w2v_aasist use "
                            "Runner.type=test with fusion_mode=heuristic, or "
                            "fusion_mode=mlp|nonlinear|score_mlp / trainable_logit_bias=true "
                            "for training). "
                            "Running test/eval instead of SASV training.",
                            flush=True,
                        )
                    self._run_test()
                    return
                sft_cfg = runner_cfg.get("SFT", {})
                test_cfg = runner_cfg.get("Test", {})
                trainer_config = {
                    **sft_cfg,
                    "lr": sft_cfg.get("optimizator", {}).get("init_lr", 1e-4),
                    "amp": amp_enabled,
                    "iters_per_epoch": sft_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": sft_cfg.get("accum_grad_iters", 1),
                    "generation": test_cfg.get("generation", {"max_new_tokens": 1, "num_beams": 1, "do_sample": False}),
                    "decision_backend": test_cfg.get("decision_backend", "llm_only"),
                    "threshold_mode": test_cfg.get("threshold_mode", "fixed"),
                    "tau_sv": test_cfg.get("tau_sv"),
                    "tau_spf": test_cfg.get("tau_spf"),
                    "threshold_objective": test_cfg.get("threshold_objective", "min_a_dcf"),
                    "extract_confidence": test_cfg.get("extract_confidence", True),
                    "meta_path": dev_meta_path,
                }
                trainer = SASVTrainer(trainer_config, self.model, self.train_loader, self.val_loader, self.logger, device=self.device, output_dir=self.output_dir)
                trainer.train()

            elif trainer_type == "SASVHardMining":
                sft_cfg = runner_cfg.get("SFT", {})
                test_cfg = runner_cfg.get("Test", {})
                trainer_config = {
                    **sft_cfg,
                    "lr": sft_cfg.get("optimizator", {}).get("init_lr", 1e-4),
                    "amp": amp_enabled,
                    "iters_per_epoch": sft_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": sft_cfg.get("accum_grad_iters", 1),
                    "generation": test_cfg.get("generation", {"max_new_tokens": 1, "num_beams": 1, "do_sample": False}),
                    "decision_backend": test_cfg.get("decision_backend", "llm_only"),
                    "threshold_mode": test_cfg.get("threshold_mode", "fixed"),
                    "tau_sv": test_cfg.get("tau_sv"),
                    "tau_spf": test_cfg.get("tau_spf"),
                    "threshold_objective": test_cfg.get("threshold_objective", "min_a_dcf"),
                    "extract_confidence": test_cfg.get("extract_confidence", True),
                    "hard_mining": sft_cfg.get("hard_mining", {}),
                    "meta_path": dev_meta_path,
                }
                trainer = SASVHardMiningTrainer(trainer_config, self.model, self.train_loader, self.val_loader, self.logger, device=self.device, output_dir=self.output_dir)
                trainer.train()

        elif run_type == "test":
            self._run_test()

    def _run_test(self):
        """Run test/evaluation mode using Datasets.dataset_test_path."""
        runner_cfg = self.config.get("Runner", {})
        test_cfg = runner_cfg.get("Test", {})
        data_cfg = self.config.get("Datasets", {})
        
        # Get model format (what format the model was trained with)
        # Falls back to SFT.task_type if not specified
        model_format = test_cfg.get("model_format")
        if not model_format:
            model_format = runner_cfg.get("SFT", {}).get("task_type", "hard_label")
        
        # Dataset format (can differ from model format for cross-evaluation)
        dataset_format = test_cfg.get("dataset_format", model_format)
        
        # Get test dataset path and settings from Datasets section
        ds_path = test_cfg.get("dataset_path") or data_cfg.get("dataset_test_path")
        max_samples = test_cfg.get("max_samples", data_cfg.get("max_test_samples"))
        ds_name = os.path.basename(ds_path).replace(".json", "").replace(".jsonl", "") if ds_path else "test"
        
        if not ds_path or ds_path == "dummy":
            print("Error: Datasets.dataset_test_path must be set for test mode", flush=True)
            return
        
        # Validate: hard_label model can't be tested on reasoning datasets
        if model_format == "hard_label" and dataset_format == "reasoning":
            print(f"Error: hard_label model cannot be tested on reasoning dataset", flush=True)
            return
        
        print(f"\n{'='*50}", flush=True)
        print(f"Testing on: {ds_name}", flush=True)
        print(f"Model format: {model_format}, Dataset format: {dataset_format}", flush=True)
        print(f"{'='*50}", flush=True)
        
        test_loader = self._build_test_dataloader(ds_path, dataset_format, max_samples)
        test_logger = self._build_test_logger(dataset_format, ds_name)
        
        from .epochs.test_epoch import TestEpoch
        gen_cfg = test_cfg.get("generation", {"max_new_tokens": 256, "num_beams": 1, "do_sample": False})
        log_freq = self.config.get("Logger", {}).get("log_freq", 10)
        extract_confidence = test_cfg.get("extract_confidence", True)
        
        test_epoch = TestEpoch(
            self.model, 
            test_loader, 
            test_logger, 
            device=self.device,
            gen_cfg=gen_cfg,
            model_format=model_format,
            dataset_format=dataset_format,
            log_freq=log_freq,
            extract_confidence=extract_confidence
        )
        test_epoch.run(epoch_num=0)
        if self._is_main_process():
            self._write_test_summary_csv(test_logger)

    def _is_main_process(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def _outputs_root_dir(self) -> str:
        """Return the shared outputs/ directory for cross-run CSV summaries."""
        base_output_dir = self.config.get("General", {}).get("output_dir", "./outputs")
        base_output_dir = os.path.abspath(os.path.expanduser(base_output_dir))
        if os.path.basename(base_output_dir) == "outputs":
            return base_output_dir
        return os.path.dirname(base_output_dir)

    def _checkpoint_summary_identity(self, ckpt_path: str) -> Dict[str, str]:
        """Infer experiment/run from outputs/<experiment>/<run>/checkpoints paths."""
        best_ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path)) if ckpt_path else ""
        parts = os.path.normpath(best_ckpt_path).split(os.sep) if best_ckpt_path else []
        for i, part in enumerate(parts):
            if part == "outputs" and i + 2 < len(parts):
                maybe_run = parts[i + 2]
                if maybe_run.startswith("run_"):
                    return {
                        "experiment": parts[i + 1],
                        "run": maybe_run,
                        "best_ckpt_path": best_ckpt_path,
                    }

        base_output_dir = self.config.get("General", {}).get("output_dir", "./outputs")
        base_output_dir = os.path.abspath(os.path.expanduser(base_output_dir))
        experiment = os.path.basename(base_output_dir)
        if experiment == "outputs":
            experiment = self.config.get("Model", {}).get("model_name", "default")
        return {
            "experiment": experiment,
            "run": os.path.basename(self.output_dir),
            "best_ckpt_path": best_ckpt_path,
        }

    def _write_test_summary_csv(self, test_logger) -> None:
        """Upsert one test-run summary row into outputs/checkpoint_test_summary.csv."""
        metrics = getattr(test_logger, "last_epoch_metrics", None)
        if not isinstance(metrics, dict) or not metrics:
            print("Warning: no test metrics available for checkpoint summary CSV", flush=True)
            return

        ckpt_path = self.config.get("Model", {}).get("ckpt", "")
        row = self._checkpoint_summary_identity(ckpt_path)
        row["meta_path"] = self._resolve_test_meta_path()
        for key, value in metrics.items():
            row[key] = self._csv_metric_value(value)

        outputs_root = self._outputs_root_dir()
        os.makedirs(outputs_root, exist_ok=True)
        csv_path = os.path.join(outputs_root, "checkpoint_test_summary.csv")
        id_fields = ["experiment", "run", "best_ckpt_path", "meta_path"]
        preferred_metric_fields = [
            "accuracy",
            "accuracy_balanced",
            "accuracy_yes",
            "accuracy_no",
            "accuracy_gen",
            "t_eer",
            "min_a_dcf",
            "min_t_dcf",
        ]
        metric_fields = preferred_metric_fields
        required_fields = id_fields + metric_fields

        rows = []
        fieldnames = required_fields
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        key = (row["experiment"], row["run"], row["best_ckpt_path"])
        updated = False
        for idx, existing in enumerate(rows):
            existing_key = (
                existing.get("experiment", ""),
                existing.get("run", ""),
                existing.get("best_ckpt_path", ""),
            )
            if existing_key == key:
                rows[idx] = {**existing, **row}
                updated = True
                break
        if not updated:
            rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"Checkpoint test summary saved to {csv_path}", flush=True)

    @staticmethod
    def _csv_metric_value(value: Any) -> Any:
        if isinstance(value, (str, int, float)) or value is None:
            return value
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _normalize_meta_path(path_value: Any) -> str:
        if isinstance(path_value, str) and path_value:
            return os.path.abspath(os.path.expanduser(path_value))
        return ""

    def _resolve_test_meta_path(self) -> str:
        """Resolve dataset path used for test metrics."""
        data_cfg = self.config.get("Datasets", {})
        for cfg_key in ("dataset_test_path", "dataset_eval_path"):
            normalized = self._normalize_meta_path(data_cfg.get(cfg_key, ""))
            if normalized:
                return normalized

        dataset = getattr(self.test_loader, "dataset", None)
        for attr_name in ("meta_path", "dataset_path", "path", "file_path", "json_path"):
            normalized = self._normalize_meta_path(getattr(dataset, attr_name, ""))
            if normalized:
                return normalized
        return ""

    def _build_test_dataloader(self, dataset_path: str, dataset_format: str, max_samples: Optional[int] = None):
        """Build a dataloader for testing."""
        data_cfg = self.config.get("Datasets", {})
        runner_cfg = self.config.get("Runner", {})
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name")
        
        whisper_path = None
        model_path = None
        silence_delay_seconds = 0.5  # Default value
        if model_name in (
            "salmon", "sasv_salmon", "salmon_hierarchical", "new_salmon",
            "new_salmon_fusion", "sasv_w2v_aasist", "sasv_ecapa_w2v_aasist",
        ):
            salmon_cfg = model_cfg.get("additional_kwargs", {}).get("salmon", {})
            whisper_path = salmon_cfg.get("pretrained_ckpts", {}).get("whisper_path")
            # Get configurable silence delay for concatenating multiple audios
            silence_delay_seconds = salmon_cfg.get("silence_delay_seconds", 0.5)
        elif model_name == "qwen_audio":
            model_path = model_cfg.get("additional_kwargs", {}).get("qwen_audio", {}).get("model_path")
        
        test_cfg = runner_cfg.get("Test", {})
        batch_size = test_cfg.get("batch_size", 8)
        
        # Load prompt templates (Test.prompt_type overrides dataset_format, e.g. reasoning prompts for hard-label GT)
        prompt_type = test_cfg.get("prompt_type", dataset_format)
        prompt_templates = self._get_prompt_templates(prompt_type)
        
        test_samples_offset = data_cfg.get("test_samples_offset", 0)
        audio_cfg = self.config.get("Audio", {})
        reasoning_version = data_cfg.get("reasoning_version", "long")
        return get_dataloader(
            dataset_path, 
            batch_size, 
            shuffle=False, 
            max_samples=max_samples or data_cfg.get("max_test_samples"),
            task_type=dataset_format,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            prompt_templates=prompt_templates,
            samples_offset=test_samples_offset,
            silence_delay_seconds=silence_delay_seconds,
            audio_cfg=audio_cfg,
            is_train=False,
            split="test",
            reasoning_version=reasoning_version,
        )

    def _build_test_logger(self, dataset_format: str, dataset_name: str):
        """Build logger appropriate for dataset format."""
        log_dir = os.path.join(self.output_dir, "logs", f"test_{dataset_name}")
        log_freq = self.config.get("Logger", {}).get("log_freq", 10)
        is_main_process = (self.local_rank == 0)

        test_cfg = self.config.get("Runner", {}).get("Test", {})
        save_all = test_cfg.get("save_all_predictions", False)

        if dataset_format == "reasoning":
            from .loggers.reasoning_logger import ReasoningLogger
            return ReasoningLogger(log_dir, log_freq if is_main_process else 1000000, save_all_predictions=save_all)
        from .loggers.hard_label_logger import HardLabelLogger
        return HardLabelLogger(log_dir, log_freq if is_main_process else 1000000, save_all_predictions=save_all)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    runner = Runner(args.config)
    try:
        runner.run()
    finally:
        # Always cleanup distributed resources, even if training fails
        runner.cleanup()

if __name__ == "__main__":
    main()
