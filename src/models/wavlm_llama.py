import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Union, Optional

from .base import Model
from .SALMON.salmonn import SALMONN


class ArcFaceHead(nn.Module):

    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
            embeddings: (B, in_features) — L2-normalized speaker embeddings
            labels: (B,) — integer speaker class indices
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)  # (B, num_classes)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = torch.cos(theta + self.m * one_hot) * self.s

        loss = F.cross_entropy(logits, labels)
        return loss


class SASVSalmonModel(Model):
    """
        speaker_embedding_dim: int — speaker embedding dimension (default: 512)
        speaker_num_classes: int|null — total ArcFace classes (bonafide speakers + 1 spoof class)
        speaker_mlp_hidden_dims: list[int] — hidden dims for MLP projector
        arcface_scale: float — ArcFace scale (default: 64.0)
        arcface_margin: float — ArcFace margin (default: 0.5)
        arcface_lambda: float — weight for ArcFace loss (default: 0.1)
        ce1_lambda: float — weight for main LLM CE loss (default: 1.0)
        enable_speaker_classification: bool — enable/disable ArcFace
        enable_bonafide_spoof_classification: bool — enable/disable CE2 bonafide/spoof head
        ce2_lambda: float — weight for CE2 loss (default: 1.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Build inner SALMONN (same logic as SalmonModel)
        salmon_config = config.get("additional_kwargs", {}).get("salmon", {}).copy()
        if "pretrained_ckpts" in salmon_config:
            ckpts = salmon_config.pop("pretrained_ckpts")
            salmon_config.update(ckpts)

        torch_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16

        salmon_config.update({
            "lora": config.get("lora", {}).get("enabled", True),
            "lora_rank": config.get("lora", {}).get("rank", 8),
            "lora_alpha": config.get("lora", {}).get("alpha", 32),
            "lora_dropout": config.get("lora", {}).get("lora_dropout", 0.1),
            "ckpt": config.get("ckpt", ""),
            "torch_dtype": torch_dtype,
        })

        self.model = SALMONN.from_config(salmon_config)

        if hasattr(self.model.llama_model, "gradient_checkpointing_enable"):
            self.model.llama_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model.llama_model, "config"):
                self.model.llama_model.config.use_cache = False

        if torch_dtype == torch.float16:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

        # Q-Former time parameters (needed for segment boundary calculation)
        self._second_stride = float(getattr(self.model, "second_stride", 0.333333))
        self._target_sr = 16000  # Whisper input sample rate

        # ---- ArcFace components ----
        sal_cfg = config.get("additional_kwargs", {}).get("salmon", {})
        self.enable_speaker = sal_cfg.get("enable_speaker_classification", False)
        self.arcface_lambda = sal_cfg.get("arcface_lambda", 0.1)
        self.ce1_lambda = float(sal_cfg.get("ce1_lambda", 1.0))
        self.enable_ce2 = sal_cfg.get("enable_bonafide_spoof_classification", True)
        self.ce2_lambda = sal_cfg.get("ce2_lambda", 1.0)

        arcface_dim = sal_cfg.get("speaker_embedding_dim", 512)
        arcface_s = sal_cfg.get("arcface_scale", 64.0)
        arcface_m = sal_cfg.get("arcface_margin", 0.5)
        num_speakers = sal_cfg.get("speaker_num_classes") or 0
        mlp_hidden = sal_cfg.get("speaker_mlp_hidden_dims", [1024, 512])

        llm_hidden_size = self.model.llama_model.config.hidden_size
        self.arcface_embedding_layers = sal_cfg.get("arcface_embedding_layers", [-1])
        if not isinstance(self.arcface_embedding_layers, list) or len(self.arcface_embedding_layers) == 0:
            self.arcface_embedding_layers = [-1]
        self.arcface_layer_pooling = sal_cfg.get("arcface_layer_pooling", "learned_weighted_sum")
        self.arcface_layer_weights = nn.Parameter(
            torch.tensor(sal_cfg.get("arcface_layer_weights", [1.0] * len(self.arcface_embedding_layers)), dtype=torch.float32)
        )

        if self.enable_ce2:
            self.bonafide_spoof_head = nn.Linear(llm_hidden_size, 2)
            logging.info(
                f"Bonafide/spoof CE2 head enabled: {llm_hidden_size} -> 2 classes "
                f"(lambda={self.ce2_lambda})"
            )
        else:
            self.bonafide_spoof_head = None
            logging.info("Bonafide/spoof CE2 head disabled")

        if self.enable_speaker and num_speakers > 0:
            self.num_speakers = num_speakers
            self._spoof_class_idx = num_speakers - 1
            self._max_bonafide_speakers = max(0, num_speakers - 1)
            # Build MLP projector: llm_hidden -> mlp_hidden[0] -> ... -> arcface_dim
            layers: List[nn.Module] = []
            in_dim = llm_hidden_size
            for h in mlp_hidden:
                layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
                in_dim = h
            layers.append(nn.Linear(in_dim, arcface_dim))
            self.speaker_projector = nn.Sequential(*layers)

            self.arcface_head = ArcFaceHead(arcface_dim, num_speakers, s=arcface_s, m=arcface_m)
            logging.info(
                f"ArcFace head: {llm_hidden_size} -> {mlp_hidden} -> {arcface_dim} -> "
                f"{num_speakers} classes ({self._max_bonafide_speakers} bonafide + 1 spoof) "
                f"(s={arcface_s}, m={arcface_m}, "
                f"lambda={self.arcface_lambda})"
            )
        else:
            self.num_speakers = 0
            self._spoof_class_idx = None
            self._max_bonafide_speakers = 0
            self.speaker_projector = None
            self.arcface_head = None
            logging.info("ArcFace disabled")

        # Speaker ID vocabulary (str -> int), built at runtime
        self._speaker_id_to_idx: Dict[str, int] = {}
        self._next_speaker_idx = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_spoof_gt(gt: Any) -> bool:
        return str(gt).lower() in {"spoof", "gen"}

    @staticmethod
    def _is_bonafide_gt(gt: Any) -> bool:
        return not SASVSalmonModel._is_spoof_gt(gt)

    def _get_speaker_indices(
        self,
        speaker_ids: List[str],
        spoof_mask: Optional[List[bool]] = None,
    ) -> Optional[torch.Tensor]:
        """Map speaker IDs to ArcFace class indices.

        Bonafide samples use ordinary speaker IDs.
        Spoof samples are mapped to one shared spoof class.
        """
        if self.arcface_head is None or not speaker_ids:
            return None

        if spoof_mask is None:
            spoof_mask = [False] * len(speaker_ids)
        if len(spoof_mask) != len(speaker_ids):
            return None

        indices = []
        for sid, is_spoof in zip(speaker_ids, spoof_mask):
            if is_spoof:
                if self._spoof_class_idx is None:
                    return None
                indices.append(self._spoof_class_idx)
                continue
            if not sid:
                return None  # skip if any speaker_id is empty
            if sid not in self._speaker_id_to_idx:
                if self._next_speaker_idx >= self._max_bonafide_speakers:
                    return None  # exceeded capacity
                self._speaker_id_to_idx[sid] = self._next_speaker_idx
                self._next_speaker_idx += 1
            indices.append(self._speaker_id_to_idx[sid])
        return torch.tensor(indices, dtype=torch.long)

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from LLM."""
        if self.model.lora:
            return self.model.llama_model.model.model.embed_tokens(token_ids)
        return self.model.llama_model.model.embed_tokens(token_ids)

    def _samples_to_qformer_tokens(self, num_audio_samples: int) -> int:
        """Convert number of audio samples to number of Q-Former output tokens."""
        duration_sec = num_audio_samples / self._target_sr
        if self._second_stride <= 0:
            return 1
        return max(1, round(duration_sec / self._second_stride))

    def _extract_segment_embedding_per_sample(
        self,
        hidden_states: torch.Tensor,
        start_tokens: List[int],
        num_tokens: List[int],
    ) -> torch.Tensor:

        batch_size, seq_len, _ = hidden_states.shape
        pooled = []
        for b in range(batch_size):
            s = int(start_tokens[b])
            n = max(1, int(num_tokens[b]))
            e = min(s + n, seq_len)
            if e <= s:
                s = max(0, min(s, seq_len - 1))
                e = s + 1
            seg = hidden_states[b, s:e, :]
            pooled.append(seg.mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _fuse_arcface_hidden_states(self, hidden_states: Union[tuple, List[torch.Tensor]]) -> torch.Tensor:
        if not hidden_states:
            raise ValueError("LLM hidden_states are required for ArcFace fusion")
        selected = [hidden_states[idx] for idx in self.arcface_embedding_layers]
        if self.arcface_layer_pooling != "learned_weighted_sum":
            raise ValueError(f"Unsupported arcface_layer_pooling: {self.arcface_layer_pooling}")
        weights = F.softmax(self.arcface_layer_weights, dim=0).to(selected[0].dtype)
        fused = sum(w * h for w, h in zip(weights, selected))
        return fused

    @staticmethod
    def _empty_layer_cosine_stats(prefix: str) -> Dict[str, Optional[torch.Tensor]]:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }

    def _compute_pairwise_layer_cosine_for_span(
        self,
        hidden_states: Union[tuple, List[torch.Tensor]],
        span_starts: List[int],
        span_lengths: List[int],
        prefix: str,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Compute pairwise layer cosine stats for per-sample token spans."""
        if not hidden_states:
            return self._empty_layer_cosine_stats(prefix)
        selected = [hidden_states[idx] for idx in self.arcface_embedding_layers]
        if len(selected) < 2:
            return self._empty_layer_cosine_stats(prefix)

        batch_size = selected[0].shape[0]
        if len(span_starts) != batch_size or len(span_lengths) != batch_size:
            return self._empty_layer_cosine_stats(prefix)

        pairwise_cosines: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    layer_i = selected[i].float()
                    layer_j = selected[j].float()
                    per_sample_cosines: List[torch.Tensor] = []
                    seq_len = layer_i.shape[1]
                    for b in range(batch_size):
                        s = int(span_starts[b])
                        n = max(1, int(span_lengths[b]))
                        s = max(0, min(s, seq_len - 1))
                        e = min(s + n, seq_len)
                        if e <= s:
                            e = s + 1
                        cos_bt = F.cosine_similarity(
                            layer_i[b, s:e, :],
                            layer_j[b, s:e, :],
                            dim=-1,
                        ).mean()
                        per_sample_cosines.append(cos_bt)
                    if per_sample_cosines:
                        pairwise_cosines.append(torch.stack(per_sample_cosines).mean())

        if not pairwise_cosines:
            return self._empty_layer_cosine_stats(prefix)
        cos_tensor = torch.stack(pairwise_cosines)
        return {
            f"{prefix}_mean": cos_tensor.mean(),
            f"{prefix}_min": cos_tensor.min(),
            f"{prefix}_max": cos_tensor.max(),
        }

    def _compute_arcface_layer_cosine_stats(
        self,
        hidden_states: Union[tuple, List[torch.Tensor]],
        speech_atts: torch.Tensor,
        samples: Dict[str, Any],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Compute pairwise cosine stats over audio tokens (global/enroll/query)."""
        if not hidden_states:
            stats = self._empty_layer_cosine_stats("arcface_layer_cosine")
            stats.update(self._empty_layer_cosine_stats("arcface_layer_cosine_enroll"))
            stats.update(self._empty_layer_cosine_stats("arcface_layer_cosine_query"))
            return stats

        batch_n = int(speech_atts.shape[0])
        speech_start = 1  # after BOS
        speech_lengths = [int(speech_atts.shape[1])] * batch_n
        global_stats = self._compute_pairwise_layer_cosine_for_span(
            hidden_states,
            [speech_start] * batch_n,
            speech_lengths,
            "arcface_layer_cosine",
        )

        enroll_stats = self._empty_layer_cosine_stats("arcface_layer_cosine_enroll")
        enroll_num_samples = samples.get("enroll_num_samples")
        if enroll_num_samples is not None and len(enroll_num_samples) == batch_n:
            enroll_lengths = [self._samples_to_qformer_tokens(ns) for ns in enroll_num_samples]
            enroll_stats = self._compute_pairwise_layer_cosine_for_span(
                hidden_states,
                [speech_start] * batch_n,
                enroll_lengths,
                "arcface_layer_cosine_enroll",
            )

        query_stats = self._empty_layer_cosine_stats("arcface_layer_cosine_query")
        query_num_samples = samples.get("query_num_samples")
        if query_num_samples is not None and len(query_num_samples) == batch_n:
            query_lengths = [self._samples_to_qformer_tokens(ns) for ns in query_num_samples]
            query_starts = [
                speech_start + max(0, speech_lengths[b] - query_lengths[b]) for b in range(batch_n)
            ]
            query_stats = self._compute_pairwise_layer_cosine_for_span(
                hidden_states,
                query_starts,
                query_lengths,
                "arcface_layer_cosine_query",
            )

        result = {}
        result.update(global_stats)
        result.update(enroll_stats)
        result.update(query_stats)
        return result

    def _compute_segment_embeddings(
        self,
        hidden: torch.Tensor,
        speech_atts: torch.Tensor,
        samples: Dict[str, Any],
    ) -> Dict[str, Optional[torch.Tensor]]:
        speech_start = 1  # after BOS
        batch_n = hidden.shape[0]
        enroll_num_samples = samples.get("enroll_num_samples")
        query_num_samples = samples.get("query_num_samples")
        enroll_embed = None
        query_embed = None

        if enroll_num_samples is not None:
            enroll_tokens = [self._samples_to_qformer_tokens(ns) for ns in enroll_num_samples]
            enroll_starts = [speech_start] * batch_n
            enroll_embed = self._extract_segment_embedding_per_sample(hidden, enroll_starts, enroll_tokens)

        if query_num_samples is not None:
            query_tokens = [self._samples_to_qformer_tokens(ns) for ns in query_num_samples]
            speech_token_len = int(speech_atts.shape[1])
            query_starts = [speech_start + max(0, speech_token_len - qt) for qt in query_tokens]
            query_embed = self._extract_segment_embedding_per_sample(hidden, query_starts, query_tokens)

        return {"enroll_embed": enroll_embed, "query_embed": query_embed}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        # Adapt prompts
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        if "prompts" in samples and isinstance(samples["prompts"], list):
            samples["prompts"] = [p.replace("<Audio>", "<SpeechHere>") for p in samples["prompts"]]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        # Encode speech
        speech_embeds, speech_atts = self.model.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
        )

        # Prompt wrapping (if model has prompt_dict)
        if self.model.prompt_dict:
            task = list(set(samples["task"]))
            if len(task) > 1 or "QA" in task:
                self.model.multi_prompt = True
            if self.model.multi_prompt:
                import random
                prompt = [random.choice(self.model.prompt_dict[t]) for t in samples["task"]]
            else:
                import random
                prompt = random.choice(self.model.prompt_dict[samples["task"][0]])
            speech_embeds, speech_atts = self.model.prompt_wrap(
                speech_embeds, speech_atts, prompt, multi_prompt=self.model.multi_prompt
            )

        # Prepare target tokens (the 1-token answer: yes/no/gen + eos)
        text = [t + self.model.end_sym for t in samples["text"]]
        to_regress_tokens = self.model.llama_tokenizer(
            text, return_tensors="pt", padding="longest",
            truncation=True, max_length=self.model.max_txt_len,
            add_special_tokens=False,
        ).to(spectrogram.device)

        to_regress_embeds = self._embed_tokens(to_regress_tokens.input_ids)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones(
            [speech_atts.shape[0], speech_atts.shape[1] + 1],
            dtype=torch.long, device=spectrogram.device,
        ).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1], dtype=to_regress_tokens.input_ids.dtype,
            device=spectrogram.device,
        ) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self._embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # Determine whether we need hidden states for ArcFace / CE2
        need_arcface = (
            self.arcface_head is not None
            and (
                samples.get("enroll_speaker_ids") is not None
                or samples.get("query_speaker_ids") is not None
            )
        )
        need_ce2 = (
            self.bonafide_spoof_head is not None
            and (
                samples.get("enroll_num_samples") is not None
                or samples.get("query_num_samples") is not None
            )
        )
        need_hidden = need_arcface or need_ce2

        # Forward LLM
        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=need_hidden,
            )
            ce1_loss = outputs.loss

        result = {
            "loss": self.ce1_lambda * ce1_loss,
            "ce1_loss": ce1_loss,
            "ce_loss": ce1_loss,
        }

        # ---- ArcFace loss on enroll AND query ----
        # Enroll embedding: first N seconds (from enroll_num_samples)
        # Query embedding:  last  N seconds (from query_num_samples)
        if need_hidden:
            layer_cosine_stats = self._compute_arcface_layer_cosine_stats(
                outputs.hidden_states,
                speech_atts,
                samples,
            )
            result.update(layer_cosine_stats)
            hidden = self._fuse_arcface_hidden_states(outputs.hidden_states)  # (B, seq_len, H)
            batch_n = hidden.shape[0]
            enroll_num_samples = samples.get("enroll_num_samples")
            query_num_samples = samples.get("query_num_samples")
            segment_embeds = self._compute_segment_embeddings(hidden, speech_atts, samples)
            enroll_embed = segment_embeds["enroll_embed"]
            query_embed = segment_embeds["query_embed"]

            ce2_losses = []
            if need_ce2:
                if enroll_embed is not None:
                    enroll_logits = self.bonafide_spoof_head(enroll_embed)
                    enroll_targets = torch.zeros(batch_n, dtype=torch.long, device=spectrogram.device)
                    ce2_losses.append(F.cross_entropy(enroll_logits, enroll_targets))

                if query_embed is not None:
                    query_gt = samples.get("gt")
                    if isinstance(query_gt, list) and len(query_gt) == batch_n:
                        query_targets = torch.tensor(
                            [1 if self._is_spoof_gt(g) else 0 for g in query_gt],
                            dtype=torch.long,
                            device=spectrogram.device,
                        )
                        query_logits = self.bonafide_spoof_head(query_embed)
                        ce2_losses.append(F.cross_entropy(query_logits, query_targets))

                if ce2_losses:
                    ce2_loss = sum(ce2_losses) / len(ce2_losses)
                    result["ce2_loss"] = ce2_loss
                    result["loss"] = result["loss"] + self.ce2_lambda * ce2_loss

            arcface_losses = []

            # --- Enroll ArcFace ---
            enroll_spk_ids = samples.get("enroll_speaker_ids")
            if enroll_spk_ids is not None and enroll_num_samples is not None:
                enroll_labels = self._get_speaker_indices(enroll_spk_ids)
                if enroll_labels is not None and enroll_embed is not None:
                    enroll_labels = enroll_labels.to(spectrogram.device)
                    enroll_embed_arc = self.speaker_projector(enroll_embed)
                    arcface_losses.append(self.arcface_head(enroll_embed_arc, enroll_labels))

            # --- Query ArcFace ---
            query_spk_ids = samples.get("query_speaker_ids")
            if query_spk_ids is not None and query_num_samples is not None:
                query_gt = samples.get("gt")
                if isinstance(query_gt, list) and len(query_gt) == batch_n:
                    bonafide_indices = [i for i, g in enumerate(query_gt) if not self._is_spoof_gt(g)]
                    if bonafide_indices and query_embed is not None:
                        query_spk_ids_bonafide = [query_spk_ids[i] for i in bonafide_indices]
                        query_labels = self._get_speaker_indices(query_spk_ids_bonafide)
                        if query_labels is not None:
                            query_labels = query_labels.to(spectrogram.device)
                            bonafide_idx_t = torch.tensor(
                                bonafide_indices,
                                dtype=torch.long,
                                device=query_embed.device,
                            )
                            query_embed_bonafide = query_embed.index_select(0, bonafide_idx_t)
                            query_embed_arc = self.speaker_projector(query_embed_bonafide)
                            arcface_losses.append(self.arcface_head(query_embed_arc, query_labels))
                else:
                    # Fallback for batches without per-sample gt labels
                    query_labels = self._get_speaker_indices(query_spk_ids)
                    if query_labels is not None and query_embed is not None:
                        query_labels = query_labels.to(spectrogram.device)
                        query_embed_arc = self.speaker_projector(query_embed)
                        arcface_losses.append(self.arcface_head(query_embed_arc, query_labels))

            if arcface_losses:
                arcface_loss = sum(arcface_losses) / len(arcface_losses)
                result["arcface_loss"] = arcface_loss
                result["loss"] = result["loss"] + self.arcface_lambda * arcface_loss

        # Verbose: token accuracy
        if verbose:
            with torch.no_grad():
                nvocab = self.model.llama_model.config.vocab_size
                logits_slice = outputs.logits[:, empty_targets.size(1) - 1: -1, :]
                results_ids = logits_slice.contiguous().view(-1, nvocab).argmax(dim=-1)
                labels_flat = targets[:, empty_targets.size(1):].contiguous().view(-1)
                mask = labels_flat != -100
                correct = (results_ids[mask] == labels_flat[mask]).float().sum()
                total = mask.sum().item()
            result["correct"] = correct
            result["total"] = total

        return result

    def extract_arcface_embeddings(self, samples: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        if "prompts" in samples and isinstance(samples["prompts"], list):
            samples["prompts"] = [p.replace("<Audio>", "<SpeechHere>") for p in samples["prompts"]]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)
        speech_embeds, speech_atts = self.model.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
        )

        if self.model.prompt_dict and "task" in samples:
            task = list(set(samples["task"]))
            if len(task) > 1 or "QA" in task:
                self.model.multi_prompt = True
            if self.model.multi_prompt:
                import random
                prompt = [random.choice(self.model.prompt_dict[t]) for t in samples["task"]]
            else:
                import random
                prompt = random.choice(self.model.prompt_dict[samples["task"][0]])
            speech_embeds, speech_atts = self.model.prompt_wrap(
                speech_embeds, speech_atts, prompt, multi_prompt=self.model.multi_prompt
            )

        batch_size = speech_embeds.shape[0]
        text = samples.get("text")
        if not isinstance(text, list) or len(text) != batch_size:
            text = [""] * batch_size
        text = [t + self.model.end_sym for t in text]
        to_regress_tokens = self.model.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False,
        ).to(spectrogram.device)
        to_regress_embeds = self._embed_tokens(to_regress_tokens.input_ids)

        bos = torch.ones(
            [batch_size, 1], dtype=to_regress_tokens.input_ids.dtype, device=spectrogram.device
        ) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self._embed_tokens(bos)
        atts_bos = speech_atts[:, :1]
        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        hidden = self._fuse_arcface_hidden_states(outputs.hidden_states)
        segment_embeds = self._compute_segment_embeddings(hidden, speech_atts, samples)
        enroll_embed = segment_embeds["enroll_embed"]
        query_embed = segment_embeds["query_embed"]
        query_ce2_spoof_probs = None
        if query_embed is not None and self.bonafide_spoof_head is not None:
            query_ce2_logits = self.bonafide_spoof_head(query_embed)
            query_ce2_spoof_probs = F.softmax(query_ce2_logits.float(), dim=1)[:, 1]

        if enroll_embed is None or query_embed is None or self.speaker_projector is None:
            return {
                "enroll_embed": None,
                "query_embed": None,
                "cos_scores": None,
                "query_ce2_spoof_probs": query_ce2_spoof_probs,
            }

        enroll_embed = F.normalize(self.speaker_projector(enroll_embed), p=2, dim=1)
        query_embed = F.normalize(self.speaker_projector(query_embed), p=2, dim=1)
        cos_scores = (enroll_embed * query_embed).sum(dim=1)
        return {
            "enroll_embed": enroll_embed,
            "query_embed": query_embed,
            "cos_scores": cos_scores,
            "query_ce2_spoof_probs": query_ce2_spoof_probs,
        }

    def generate(self, samples: Dict[str, Any], generate_cfg: Dict[str, Any],
                 prompts: Optional[List[str]] = None, return_outputs: bool = False):
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        if prompts:
            prompts = [p.replace("<Audio>", "<SpeechHere>") for p in prompts]

        # Force max_new_tokens=1 for SASV (single token output)
        gen_cfg = dict(generate_cfg)
        gen_cfg["max_new_tokens"] = max(gen_cfg.get("max_new_tokens", 1), 1)

        num_return_sequences = gen_cfg.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            samples = {
                k: (v.repeat_interleave(num_return_sequences, dim=0) if isinstance(v, torch.Tensor)
                    else [x for x in v for _ in range(num_return_sequences)])
                for k, v in samples.items()
            }

        if return_outputs:
            texts = self.model.generate(samples, gen_cfg, prompts=prompts)
            completion_ids = []
            for t in texts:
                ids = self.model.llama_tokenizer(t, add_special_tokens=False).input_ids
                completion_ids.append(torch.tensor(ids))
            completion_ids = torch.nn.utils.rnn.pad_sequence(
                completion_ids, batch_first=True,
                padding_value=self.model.llama_tokenizer.pad_token_id,
            )
            logits = self.model.compute_logits(samples, completion_ids)
            return texts, completion_ids, logits

        return self.model.generate(samples, gen_cfg, prompts=prompts)

    def compute_logits_for_completions(self, samples, completion_ids):
        return self.model.compute_logits(samples, completion_ids)

    def get_tokenizer(self):
        return getattr(self.model, "llama_tokenizer", None)

    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict[str, Any]) -> List[str]:
        inputs = self.model.llama_tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.llama_model.generate(
            **inputs,
            max_new_tokens=generate_cfg.get("max_new_tokens", 10),
            do_sample=generate_cfg.get("do_sample", False),
        )
        return self.model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
