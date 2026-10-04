"""
SASV (Spoofing-Aware Speaker Verification) model based on SALMONN ALM.

Architecture:
  - SALMONN encodes concatenated (enroll + silence + query) audio
  - LLM predicts 1 token: "yes" / "no" / "gen"
  - ArcFace head on Q-Former hidden states for speaker discrimination
    (applied separately to enroll and query segments)
  - Combined loss: CE1 (token prediction) + CE2 (bonafide/spoof) + lambda * ArcFace (speaker)
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Union, Optional

from .base import Model
from .SALMON.salmonn import SALMONN
from .prompt_utils import wrap_speech_with_prompts
from .sasv_loss_utils import (
    build_full_trace_loss_weights,
    build_text_weight_tensors,
    parse_trace_class_weights,
    sample_trace_class_weights,
    weighted_causal_lm_loss,
)


class ArcFaceHead(nn.Module):
    """ArcFace (Additive Angular Margin) classification head for speaker embeddings."""

    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self._eps = 1e-6
        # Precompute margin constants for the acos-free formulation.
        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, in_features) — speaker embeddings (will be L2-normalized)
            labels: (B,) — integer speaker class indices
        Returns:
            ArcFace cross-entropy loss scalar

        Numerically stable, acos-free implementation computed in fp32:
            cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m),
            sin(theta) = sqrt(1 - cos(theta)^2).
        Running in fp32 avoids the inf gradients that ``acos`` produces near
        +/-1 under fp16/bf16 autocast.
        """
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        weight = F.normalize(self.weight.float(), p=2, dim=1)

        cosine = F.linear(embeddings, weight).clamp(-1.0 + self._eps, 1.0 - self._eps)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(self._eps))
        phi = cosine * self._cos_m - sine * self._sin_m  # cos(theta + m)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s

        loss = F.cross_entropy(logits, labels)
        return loss


class SASVSalmonModel(Model):
    """SASV model wrapping SALMONN with ArcFace speaker head.

    ArcFace is computed on BOTH enroll and query segments of the
    concatenated audio.  The segment boundaries are derived from
    ``enroll_num_samples`` / ``query_num_samples`` / ``silence_samples``
    passed in the batch dict by the collate function, together with
    the Q-Former window stride from the SALMONN config.

    Config keys (under Model.additional_kwargs.salmon):
        speaker_embedding_dim: int — speaker embedding dimension (default: 512)
        speaker_num_classes: int|null — total ArcFace classes (bonafide speakers + 1 spoof class)
        speaker_mlp_hidden_dims: list[int] — hidden dims for MLP projector
        arcface_scale: float — ArcFace scale (default: 64.0)
        arcface_margin: float — ArcFace margin (default: 0.5)
        arcface_lambda: float — weight for ArcFace loss (default: 0.1)
        enable_speaker_classification: bool — enable/disable ArcFace
        enable_bonafide_spoof_classification: bool — enable/disable CE2 bonafide/spoof head
        ce2_lambda: float — weight for CE2 loss (default: 1.0)
        wrap_collator_prompts: bool — when false, ignore collator ``prompts`` and only
            wrap speech via ``prompt_path``/``prompt_dict`` (pre-bugfix #1 behavior).
            Set false when evaluating checkpoints trained without instruction prompts.
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

        self.wrap_collator_prompts = bool(salmon_config.get("wrap_collator_prompts", True))

        salmon_config.update({
            "max_txt_len": config.get("max_txt_len", 1024),
            "lora": config.get("lora", {}).get("enabled", True),
            "lora_rank": config.get("lora", {}).get("rank", 8),
            "lora_alpha": config.get("lora", {}).get("alpha", 32),
            "lora_dropout": config.get("lora", {}).get("lora_dropout", 0.1),
            "ckpt": config.get("ckpt", ""),
            "torch_dtype": torch_dtype,
            "wrap_collator_prompts": self.wrap_collator_prompts,
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
        self.ce1_lambda = float(sal_cfg.get("ce1_lambda", 1.0))

        # ---- Token-level CE1 weighting (answer / format tokens + focal hard mining) ----
        # All default to a no-op (weights 1.0, gamma 0.0) so behaviour is unchanged
        # unless explicitly configured.  See sasv_loss_utils for the rationale.
        self.answer_loss_weight = float(sal_cfg.get("answer_loss_weight", 1.0))
        self.format_loss_weight = float(sal_cfg.get("format_loss_weight", 1.0))
        eos_w = sal_cfg.get("eos_loss_weight", None)
        self.eos_loss_weight = float(eos_w) if eos_w is not None else None
        self.answer_focal_gamma = float(sal_cfg.get("answer_focal_gamma", 0.0))
        self.token_weighting_enabled = (
            self.answer_loss_weight != 1.0
            or self.format_loss_weight != 1.0
            or (self.eos_loss_weight is not None and self.eos_loss_weight != 1.0)
            or self.answer_focal_gamma > 0.0
        )
        if self.token_weighting_enabled:
            logging.info(
                "CE1 token weighting enabled: answer=%.2f format=%.2f eos=%s focal_gamma=%.2f",
                self.answer_loss_weight,
                self.format_loss_weight,
                self.eos_loss_weight,
                self.answer_focal_gamma,
            )

        self.trace_class_weights = parse_trace_class_weights(sal_cfg.get("trace_class_weights"))
        if self.trace_class_weights is not None:
            logging.info(
                "trace_class_weights enabled: %s",
                self.trace_class_weights,
            )
        self.arcface_lambda = sal_cfg.get("arcface_lambda", 0.1)
        self.enable_ce2 = sal_cfg.get("enable_bonafide_spoof_classification", True)
        self.ce2_lambda = sal_cfg.get("ce2_lambda", 1.0)
        self.ce2_class_weights = sal_cfg.get("ce2_class_weights", None)
        self.asv_pair_lambda = float(sal_cfg.get("asv_pair_lambda", 0.0))
        self.asv_pair_scale = float(sal_cfg.get("asv_pair_scale", 10.0))
        self.asv_pair_spoof_weight = float(sal_cfg.get("asv_pair_spoof_weight", 0.25))
        self.asv_pair_include_spoof = bool(sal_cfg.get("asv_pair_include_spoof", False))

        arcface_dim = sal_cfg.get("speaker_embedding_dim", 512)
        arcface_s = sal_cfg.get("arcface_scale", 64.0)
        arcface_m = sal_cfg.get("arcface_margin", 0.5)
        num_speakers = sal_cfg.get("speaker_num_classes") or 0
        mlp_hidden = sal_cfg.get("speaker_mlp_hidden_dims", [1024, 512])

        llm_hidden_size = self.model.llama_model.config.hidden_size

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

        # Speaker ID vocabulary (str -> int). Built deterministically from the
        # dataset (see ``set_speaker_mapping``) so that the ArcFace weight columns
        # stay aligned with the same speakers across runs / checkpoint resumes.
        # Falls back to lazy runtime assignment if the mapping was never set.
        self._speaker_id_to_idx: Dict[str, int] = {}
        self._next_speaker_idx = 0
        self._speaker_mapping_frozen = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_speaker_mapping(self, speaker_ids: List[str]) -> Optional[Dict[str, int]]:
        """Build a deterministic speaker_id -> ArcFace-index mapping.

        The mapping is derived from the *sorted* set of bonafide speaker IDs, so
        it is identical on every process and every resume as long as the training
        metadata is unchanged. This keeps the trained ``arcface_head.weight``
        columns aligned with the same speakers (the previous lazy/per-run mapping
        silently desynced on resume and made the ArcFace loss diverge).
        """
        if self.arcface_head is None:
            return None
        uniq = sorted({str(s) for s in speaker_ids if s})
        cap = self._max_bonafide_speakers
        if len(uniq) > cap:
            logging.warning(
                "set_speaker_mapping: %d speakers exceed bonafide capacity %d; truncating "
                "(increase speaker_num_classes to len(speakers)+1)",
                len(uniq), cap,
            )
            uniq = uniq[:cap]
        self._speaker_id_to_idx = {sid: i for i, sid in enumerate(uniq)}
        self._next_speaker_idx = len(uniq)
        self._speaker_mapping_frozen = True
        logging.info(
            "Deterministic speaker mapping set: %d bonafide speakers "
            "(capacity=%d, spoof_class_idx=%s)",
            len(uniq), cap, self._spoof_class_idx,
        )
        return self._speaker_id_to_idx

    @staticmethod
    def _is_spoof_gt(gt: Any) -> bool:
        return str(gt).lower() in {"spoof", "gen"}

    @staticmethod
    def _is_bonafide_gt(gt: Any) -> bool:
        return not SASVSalmonModel._is_spoof_gt(gt)

    def _ce2_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Bonafide/spoof CE with optional class weights: class 0=bonafide, 1=spoof."""
        weight = None
        if self.ce2_class_weights is not None:
            if isinstance(self.ce2_class_weights, dict):
                bonafide_w = float(self.ce2_class_weights.get("bonafide", self.ce2_class_weights.get("bona", 1.0)))
                spoof_w = float(self.ce2_class_weights.get("spoof", self.ce2_class_weights.get("gen", 1.0)))
                weight = torch.tensor([bonafide_w, spoof_w], dtype=torch.float32, device=logits.device)
            elif isinstance(self.ce2_class_weights, (list, tuple)) and len(self.ce2_class_weights) == 2:
                weight = torch.tensor(self.ce2_class_weights, dtype=torch.float32, device=logits.device)
        return F.cross_entropy(logits.float(), targets, weight=weight)

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
                if self._speaker_mapping_frozen:
                    return None  # unknown speaker under a frozen deterministic mapping
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
        """Mean-pool hidden states over per-sample segments.

        Args:
            hidden_states: (B, seq_len, H) — LLM last hidden state
            start_tokens: list of start positions for each sample
            num_tokens: list of segment lengths for each sample
        Returns:
            (B, H) pooled embeddings
        """
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with CE + ArcFace loss on both enroll and query.

        Batch dict expected keys:
            spectrogram, raw_wav, text, task
            enroll_speaker_ids (list[str]): speaker IDs for enroll audios
            query_speaker_ids (list[str]): speaker IDs for query audios
            enroll_num_samples (list[int]): audio sample counts per enroll
            query_num_samples (list[int]): audio sample counts per query
            silence_samples (int): silence samples between enroll and query
        """
        # Adapt prompts
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        if "prompts" in samples and isinstance(samples["prompts"], list):
            samples["prompts"] = [p.replace("<Audio>", "<SpeechHere>") for p in samples["prompts"]]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        # Encode speech.
        # ``speech_embeds_raw`` are the Q-Former outputs *before* prompt wrapping:
        # their time axis maps directly to the concatenated [enroll | silence | query]
        # audio (token 0 = start of enroll, last tokens = end of query). The speaker /
        # Segment boundaries for CE2 heads (exclude silence gap)
        # (not prompt tokens) and their gradients flow into the encoder/Q-Former only —
        # never into the LLM weights.
        speech_embeds, speech_atts = self.model.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
        )
        speech_embeds_raw = speech_embeds
        speech_token_len_raw = int(speech_embeds_raw.shape[1])

        speech_embeds, speech_atts = wrap_speech_with_prompts(
            self.model, speech_embeds, speech_atts, samples, training=self.training
        )

        # Prepare target tokens (reasoning trace + <answer>yes/no/gen</answer> + eos)
        target_texts = list(samples["text"])  # before end_sym, for token weighting
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

        # Forward LLM (only CE1 — speaker/CE2 heads use the pre-LLM Q-Former features)
        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            ce1_loss = outputs.loss

        # Optional CE1 token / sample weighting.
        ce1_loss_unweighted = ce1_loss
        use_custom_ce1 = self.token_weighting_enabled or (
            self.training and self.trace_class_weights is not None
        )
        if use_custom_ce1:
            prefix_len = empty_targets.size(1)
            text_weights = None
            text_answer_mask = None
            if self.token_weighting_enabled:
                text_weights, text_answer_mask = build_text_weight_tensors(
                    self.model.llama_tokenizer,
                    target_texts,
                    self.model.end_sym,
                    to_regress_tokens.input_ids,
                    base_weight=1.0,
                    answer_weight=self.answer_loss_weight,
                    format_weight=self.format_loss_weight,
                    eos_weight=self.eos_loss_weight,
                )
                full_token_weights = torch.zeros_like(targets, dtype=torch.float32)
                full_answer_mask = torch.zeros_like(targets, dtype=torch.bool)
                full_token_weights[:, prefix_len:] = text_weights
                full_answer_mask[:, prefix_len:] = text_answer_mask
            else:
                full_token_weights = None
                full_answer_mask = None

            sample_weights = None
            if self.training and self.trace_class_weights is not None:
                gt_labels = samples.get("answer") or samples.get("gt") or []
                if not isinstance(gt_labels, list):
                    gt_labels = [gt_labels] * batch_size
                sample_weights = sample_trace_class_weights(gt_labels, self.trace_class_weights)

            full_weights = build_full_trace_loss_weights(
                targets,
                prefix_len,
                sample_weights=sample_weights,
                text_token_weights=full_token_weights,
            )
            ce1_loss = weighted_causal_lm_loss(
                outputs.logits,
                targets,
                full_weights,
                answer_mask=full_answer_mask,
                focal_gamma=self.answer_focal_gamma if self.token_weighting_enabled else 0.0,
            )

        result = {
            "loss": self.ce1_lambda * ce1_loss,
            "ce1_loss": ce1_loss,
            "ce_loss": ce1_loss,
            "ce1_loss_unweighted": ce1_loss_unweighted,
        }

        # ---- ArcFace / CE2 on enroll AND query (from Q-Former features) ----
        # Enroll embedding: first enroll_tokens of the concatenated stream
        # Query embedding:  last  query_tokens of the concatenated stream
        if need_hidden:
            speech_feats = speech_embeds_raw  # (B, S, H) — pre-prompt audio tokens
            speech_start = 0
            batch_n = speech_feats.shape[0]

            enroll_num_samples = samples.get("enroll_num_samples")
            query_num_samples = samples.get("query_num_samples")
            enroll_embed = None
            query_embed = None

            if enroll_num_samples is not None:
                enroll_tokens = [self._samples_to_qformer_tokens(ns) for ns in enroll_num_samples]
                enroll_starts = [speech_start] * batch_n
                enroll_embed = self._extract_segment_embedding_per_sample(
                    speech_feats, enroll_starts, enroll_tokens
                )

            if query_num_samples is not None:
                query_tokens = [self._samples_to_qformer_tokens(ns) for ns in query_num_samples]
                query_starts = [speech_start + max(0, speech_token_len_raw - qt) for qt in query_tokens]
                query_embed = self._extract_segment_embedding_per_sample(
                    speech_feats, query_starts, query_tokens
                )

            ce2_losses = []
            if need_ce2:
                if enroll_embed is not None:
                    enroll_logits = self.bonafide_spoof_head(enroll_embed)
                    enroll_targets = torch.zeros(batch_n, dtype=torch.long, device=spectrogram.device)
                    ce2_losses.append(self._ce2_loss(enroll_logits, enroll_targets))

                if query_embed is not None:
                    query_gt = samples.get("gt")
                    if isinstance(query_gt, list) and len(query_gt) == batch_n:
                        query_targets = torch.tensor(
                            [1 if self._is_spoof_gt(g) else 0 for g in query_gt],
                            dtype=torch.long,
                            device=spectrogram.device,
                        )
                        query_logits = self.bonafide_spoof_head(query_embed)
                        ce2_losses.append(self._ce2_loss(query_logits, query_targets))

                if ce2_losses:
                    ce2_loss = sum(ce2_losses) / len(ce2_losses)
                    result["ce2_loss"] = ce2_loss
                    result["loss"] = result["loss"] + self.ce2_lambda * ce2_loss

            if (
                self.asv_pair_lambda > 0
                and enroll_embed is not None
                and query_embed is not None
                and isinstance(samples.get("gt"), list)
                and len(samples["gt"]) == batch_n
            ):
                pair_labels = []
                pair_weights = []
                for g in samples["gt"]:
                    is_spoof = self._is_spoof_gt(g)
                    if is_spoof and not self.asv_pair_include_spoof:
                        pair_labels.append(0.0)
                        pair_weights.append(0.0)
                    else:
                        pair_labels.append(1.0 if str(g).lower() in {"yes", "verified"} else 0.0)
                        pair_weights.append(self.asv_pair_spoof_weight if is_spoof else 1.0)

                pair_weights_t = torch.tensor(pair_weights, dtype=torch.float32, device=spectrogram.device)
                if torch.any(pair_weights_t > 0):
                    pair_targets_t = torch.tensor(pair_labels, dtype=torch.float32, device=spectrogram.device)
                    pair_logits = (
                        F.cosine_similarity(
                            F.normalize(enroll_embed.float(), p=2, dim=1),
                            F.normalize(query_embed.float(), p=2, dim=1),
                            dim=1,
                        )
                        * self.asv_pair_scale
                    )
                    asv_pair_raw = F.binary_cross_entropy_with_logits(
                        pair_logits,
                        pair_targets_t,
                        reduction="none",
                    )
                    asv_pair_loss = (asv_pair_raw * pair_weights_t).sum() / pair_weights_t.sum().clamp_min(1.0)
                    result["asv_pair_loss"] = asv_pair_loss
                    result["loss"] = result["loss"] + self.asv_pair_lambda * asv_pair_loss

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

    # ------------------------------------------------------------------
    # Scoring for hard-pair mining (no ArcFace, returns per-sample scores)
    # ------------------------------------------------------------------

    def score_pairs(self, samples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Lightweight forward pass that returns per-sample difficulty scores.

        Used by :class:`HardPairMiner` for offline mining.  Does **not**
        compute ArcFace loss (no speaker-label mapping needed).

        Returns dict with optional keys (present when the corresponding head
        is enabled and the batch contains the required fields):
            enroll_embed   : (B, D)  — speaker-projected enroll embedding
            query_embed    : (B, D)  — speaker-projected query embedding
            cosine_sim     : (B,)    — cosine similarity(enroll, query)
            bonafide_prob  : (B,)    — P(bonafide) from CE2 head on query
            ce1_loss_per_sample : (B,) — per-sample token-prediction loss
        """
        # Adapt prompts
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        if "prompts" in samples and isinstance(samples["prompts"], list):
            samples["prompts"] = [p.replace("<Audio>", "<SpeechHere>") for p in samples["prompts"]]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        # Encode speech (keep pre-prompt Q-Former features for speaker segments)
        speech_embeds, speech_atts = self.model.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
        )
        speech_embeds_raw = speech_embeds
        speech_token_len_raw = int(speech_embeds_raw.shape[1])

        speech_embeds, speech_atts = wrap_speech_with_prompts(
            self.model, speech_embeds, speech_atts, samples, training=self.training
        )

        # Prepare target tokens
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

        # Forward LLM (CE1 logits only; speaker embeddings come from Q-Former features)
        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        result: Dict[str, torch.Tensor] = {}

        # --- Per-sample CE1 loss ---
        # Recompute CE per sample (the model returns the mean)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        per_sample_ce1 = torch.zeros(batch_size, device=spectrogram.device)
        for b in range(batch_size):
            mask = shift_labels[b] != -100
            if mask.any():
                per_sample_ce1[b] = F.cross_entropy(
                    shift_logits[b][mask], shift_labels[b][mask]
                )
        result["ce1_loss_per_sample"] = per_sample_ce1

        # --- Speaker embeddings & cosine similarity (from Q-Former features) ---
        speech_feats = speech_embeds_raw
        speech_start = 0

        enroll_num_samples = samples.get("enroll_num_samples")
        query_num_samples = samples.get("query_num_samples")

        enroll_embed = None
        query_embed = None

        if enroll_num_samples is not None:
            enroll_tokens = [self._samples_to_qformer_tokens(ns) for ns in enroll_num_samples]
            enroll_starts = [speech_start] * batch_size
            enroll_embed = self._extract_segment_embedding_per_sample(
                speech_feats, enroll_starts, enroll_tokens
            )

        if query_num_samples is not None:
            query_tokens = [self._samples_to_qformer_tokens(ns) for ns in query_num_samples]
            query_starts = [speech_start + max(0, speech_token_len_raw - qt) for qt in query_tokens]
            query_embed = self._extract_segment_embedding_per_sample(
                speech_feats, query_starts, query_tokens
            )

        # Project through speaker MLP if available
        if self.speaker_projector is not None:
            if enroll_embed is not None:
                enroll_proj = self.speaker_projector(enroll_embed)
                enroll_proj = F.normalize(enroll_proj, p=2, dim=1)
                result["enroll_embed"] = enroll_proj
            if query_embed is not None:
                query_proj = self.speaker_projector(query_embed)
                query_proj = F.normalize(query_proj, p=2, dim=1)
                result["query_embed"] = query_proj

            if enroll_embed is not None and query_embed is not None:
                result["cosine_sim"] = F.cosine_similarity(
                    result["enroll_embed"], result["query_embed"], dim=1
                )
        elif enroll_embed is not None and query_embed is not None:
            # Fallback: cosine on raw hidden states
            result["cosine_sim"] = F.cosine_similarity(
                F.normalize(enroll_embed, p=2, dim=1),
                F.normalize(query_embed, p=2, dim=1),
                dim=1,
            )

        # --- Bonafide probability from CE2 head ---
        if self.bonafide_spoof_head is not None and query_embed is not None:
            query_logits = self.bonafide_spoof_head(query_embed)
            probs = F.softmax(query_logits, dim=1)
            result["bonafide_prob"] = probs[:, 0]  # class 0 = bonafide

        return result

    def extract_arcface_embeddings(self, samples: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        """Compatibility interface for cosine-based SASV evaluation.

        The incoming branch used this method name for ArcFace cosine backends.
        Keep it backed by the first-branch Q-Former feature path so existing
        checkpoints keep the same embedding distribution.
        """
        pair_scores = self.score_pairs(samples)
        bonafide_prob = pair_scores.get("bonafide_prob")
        spoof_prob = 1.0 - bonafide_prob if bonafide_prob is not None else None
        return {
            "enroll_embed": pair_scores.get("enroll_embed"),
            "query_embed": pair_scores.get("query_embed"),
            "cos_scores": pair_scores.get("cosine_sim"),
            "query_ce2_spoof_probs": spoof_prob,
        }

    def _resolve_generation_prompts(
        self,
        samples: Dict[str, Any],
        prompts: Optional[List[str]],
    ) -> Optional[List[str]]:
        """Pick prompts for generate/logit scoring.

        Explicit ``prompts`` always win. Otherwise collator prompts are used only when
        ``wrap_collator_prompts`` is true (bugfix #1). When false, returns ``None`` so
        SALMONN sees ``[BOS, speech]`` — matching checkpoints trained without prompts.
        """
        if prompts is not None:
            resolved = prompts
        elif self.wrap_collator_prompts:
            resolved = samples.get("prompts")
        else:
            resolved = None
        if resolved:
            return [p.replace("<Audio>", "<SpeechHere>") for p in resolved]
        return None

    def generate(self, samples: Dict[str, Any], generate_cfg: Dict[str, Any],
                 prompts: Optional[List[str]] = None, return_outputs: bool = False,
                 return_logits: bool = True, return_generation_info: bool = False):
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        prompts = self._resolve_generation_prompts(samples, prompts)

        gen_cfg = dict(generate_cfg)
        gen_cfg["max_new_tokens"] = max(gen_cfg.get("max_new_tokens", 1), 1)
        num_return_sequences = gen_cfg.get("num_return_sequences", 1)

        if return_outputs:
            if return_generation_info:
                texts, generation_info = self.model.generate(
                    samples, gen_cfg, prompts=prompts, return_generation_info=True
                )
            else:
                texts = self.model.generate(samples, gen_cfg, prompts=prompts)
                generation_info = None
            completion_ids = []
            pad_token_id = self.model.llama_tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.model.llama_tokenizer.eos_token_id
            for t in texts:
                ids = self.model.llama_tokenizer(t, add_special_tokens=False).input_ids
                if len(ids) == 0:
                    ids = [pad_token_id]
                completion_ids.append(torch.tensor(ids, dtype=torch.long))
            completion_ids = torch.nn.utils.rnn.pad_sequence(
                completion_ids, batch_first=True,
                padding_value=pad_token_id,
            )
            if return_logits:
                if num_return_sequences > 1:
                    samples_for_logits = {
                        k: (v.repeat_interleave(num_return_sequences, dim=0) if isinstance(v, torch.Tensor)
                            else [x for x in v for _ in range(num_return_sequences)])
                        for k, v in samples.items()
                    }
                    prompts_for_logits = (
                        [p for p in prompts for _ in range(num_return_sequences)] if prompts else None
                    )
                else:
                    samples_for_logits = samples
                    prompts_for_logits = prompts
                logits = self.model.compute_logits(
                    samples_for_logits, completion_ids, prompts=prompts_for_logits
                )
            else:
                logits = None
            if return_generation_info:
                return texts, completion_ids, logits, generation_info
            return texts, completion_ids, logits

        if return_generation_info:
            return self.model.generate(
                samples, gen_cfg, prompts=prompts, return_generation_info=True
            )
        return self.model.generate(samples, gen_cfg, prompts=prompts)

    def compute_logits_for_completions(self, samples, completion_ids, prompts: Optional[List[str]] = None):
        prompts = self._resolve_generation_prompts(samples, prompts)
        return self.model.compute_logits(samples, completion_ids, prompts=prompts)

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
