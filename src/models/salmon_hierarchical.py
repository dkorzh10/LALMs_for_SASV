"""
SASV (Spoofing-Aware Speaker Verification) model based on SALMONN ALM.

Architecture:
  - SALMONN encodes concatenated (enroll + silence + query) audio
  - LLM hidden state at last speech position -> Linear(3): yes / no / gen
  - ArcFace head on Q-Former hidden states for speaker discrimination
    (applied separately to enroll and query segments)
  - Combined loss: CE1 (3-way yes/no/gen) + CE2 (bonafide/spoof) + lambda * ArcFace (speaker)
"""

import logging
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Union, Optional, Tuple

from .base import Model
from .SALMON.salmonn import SALMONN
from .prompt_utils import resolve_speech_prompts, wrap_speech_with_prompts


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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, in_features) — L2-normalized speaker embeddings
            labels: (B,) — integer speaker class indices
        Returns:
            ArcFace cross-entropy loss scalar
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)  # (B, num_classes)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # Add angular margin to target class
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = torch.cos(theta + self.m * one_hot) * self.s

        loss = F.cross_entropy(logits, labels)
        return loss


class BeheadedSASVSalmonModel(Model):
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
        ce2_apply_enroll: bool — if false, CE2 is only on query (enroll is always bonafide and can fight query)
        ce1_pooling: str — "audio_mean" (pool LLM states over the audio token span) or "last_token"
        use_class_weights: bool — if true and sasv_class_weights set, CE1 uses weighted cross-entropy
        sasv_class_weights: dict — e.g. {"yes": 1.25, "no": 1.0, "gen": 1.65} (stored on inner SALMONN)
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

        sal_cfg = config.get("additional_kwargs", {}).get("salmon", {})
        if sal_cfg.get("llama_gradient_checkpointing", True):
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
        self.enable_ce2 = sal_cfg.get("enable_bonafide_spoof_classification", True)
        self.ce2_lambda = sal_cfg.get("ce2_lambda", 1.0)
        self.ce2_apply_enroll = sal_cfg.get("ce2_apply_enroll", True)
        self.ce1_pooling = sal_cfg.get("ce1_pooling", "audio_mean")

        arcface_dim = sal_cfg.get("speaker_embedding_dim", 512)
        arcface_s = sal_cfg.get("arcface_scale", 64.0)
        arcface_m = sal_cfg.get("arcface_margin", 0.5)
        num_speakers = sal_cfg.get("speaker_num_classes") or 0
        mlp_hidden = sal_cfg.get("speaker_mlp_hidden_dims", [1024, 512])

        llm_hidden_size = self.model.llama_model.config.hidden_size

        self.answer_head = nn.Linear(llm_hidden_size, 3)
        logging.info(f"SASV answer head: {llm_hidden_size} -> 3 classes (yes / no / gen)")

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

    @classmethod
    def _is_bonafide_gt(cls, gt: Any) -> bool:
        return not cls._is_spoof_gt(gt)

    _ANSWER_CLASS_NAMES = ("yes", "no", "gen")

    @classmethod
    def _answer_class_from_label(cls, label: Any) -> int:
        raw = str(label).strip()
        s = raw.lower()
        m = re.search(
            r"<answer>\s*(yes|no|gen|spoof|verified|rejected)\s*</answer>",
            s,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            s = m.group(1).lower()
        if s == "verified":
            return 0
        if s == "rejected":
            return 1
        if s == "yes":
            return 0
        if s == "no":
            return 1
        if s in ("gen", "spoof"):
            return 2
        raise ValueError(
            f"Unknown SASV label {label!r}; expected one of {cls._ANSWER_CLASS_NAMES} "
            f"(or 'spoof' / verified / rejected)"
        )

    def _sasv_target_strings(self, samples: Dict[str, Any]) -> List[str]:
        """CE1 targets: prefer short ``answer``; else parse ``text`` (reasoning or hard_label)."""
        texts = samples.get("text")
        if not isinstance(texts, list):
            texts = [texts] if texts is not None else []
        n = len(texts)
        answers = samples.get("answer")
        if isinstance(answers, list) and len(answers) == n:
            out: List[str] = []
            for i in range(n):
                a = str(answers[i]).strip()
                if a:
                    out.append(a)
                else:
                    out.append(str(texts[i]).strip())
            return out
        return [str(t).strip() for t in texts]

    def _answer_targets_tensor(self, samples: Dict[str, Any], device: torch.device) -> torch.Tensor:
        labels = self._sasv_target_strings(samples)
        idx = [self._answer_class_from_label(t) for t in labels]
        return torch.tensor(idx, dtype=torch.long, device=device)

    @staticmethod
    def _gather_last_valid_hidden(
        hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Last valid token per row: hidden (B, T, H), mask (B, T) with 1 on valid prefix."""
        lengths = attention_mask.long().sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        b = hidden.shape[0]
        batch_idx = torch.arange(b, device=hidden.device)
        return hidden[batch_idx, lengths, :]

    def _ce1_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_layout: Dict[str, int],
    ) -> torch.Tensor:
        """Hidden state(s) for the 3-way SASV head: audio span mean or last valid token."""
        if self.ce1_pooling == "last_token":
            return self._gather_last_valid_hidden(hidden, attention_mask)
        audio_start = int(audio_layout["audio_start"])
        n_audio = max(1, int(audio_layout["n_audio"]))
        b, t, _ = hidden.shape
        end = min(audio_start + n_audio, t)
        start = max(0, min(audio_start, t - 1))
        if end <= start:
            return self._gather_last_valid_hidden(hidden, attention_mask)
        seg = hidden[:, start:end, :]
        mask = attention_mask[:, start:end].float().unsqueeze(-1)
        w = mask.squeeze(-1)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (seg * mask).sum(dim=1) / denom
        valid = w.sum(dim=1) > 0
        if bool(valid.all()):
            return pooled
        fallback = self._gather_last_valid_hidden(hidden, attention_mask)
        return torch.where(valid.unsqueeze(-1), pooled, fallback)

    def _answer_logits_from_hidden(self, hidden_last_speech: torch.Tensor) -> torch.Tensor:
        return self.answer_head(hidden_last_speech.float())

    def _ce1_class_weight_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        """``[yes, no, gen]`` weights for 3-way CE1; mirrors ``SALMONN.sasv_class_weights``."""
        inner = self.model
        if not getattr(inner, "use_class_weights", True):
            return None
        wd = getattr(inner, "sasv_class_weights", None)
        if not wd:
            return None
        yes = float(wd.get("yes", 1.0))
        no = float(wd.get("no", 1.0))
        gen = float(wd.get("gen", 1.0))
        return torch.tensor([yes, no, gen], device=device, dtype=torch.float32)

    def _prompt_prefix_token_width(
        self,
        prompt: Union[str, List[str]],
        multi_prompt: bool,
        device: torch.device,
    ) -> int:
        """Token width of the ``p_before`` segment (must match ``SALMONN.prompt_wrap`` batching)."""
        tok = self.model.llama_tokenizer
        if not multi_prompt:
            assert isinstance(prompt, str)
            p_before, _ = prompt.split("<SpeechHere>")
            t = tok(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            return int(t.input_ids.shape[1])
        assert isinstance(prompt, list)
        p_before_list = [p.split("<SpeechHere>")[0] for p in prompt]
        t = tok(p_before_list, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to(device)
        return int(t.input_ids.shape[1])

    def _encode_speech_for_sasv(
        self,
        samples: Dict[str, Any],
        override_prompts: Optional[Union[List[str], str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.device, Dict[str, int]]:
        """Encode audio (+ optional prompt wrap). ``override_prompts`` matches ``generate(..., prompts=...)``.

        Returns ``audio_layout``:
            ``audio_start`` — index of the first **audio** LLM token in ``[BOS | wrapped]`` (0 = BOS);
            ``n_audio`` — number of audio tokens inside ``wrapped`` (between p_before and p_after).
        """
        if override_prompts is None:
            bp = samples.get("prompts")
            if isinstance(bp, list) and len(bp) > 0:
                override_prompts = bp
            elif isinstance(bp, str) and bp:
                override_prompts = bp

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.model.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
        )
        n_audio = int(speech_embeds.shape[1])
        n_before = 0

        prompt_samples = dict(samples)
        if override_prompts is not None:
            prompt_samples["prompts"] = override_prompts

        resolved_prompt, use_multi = resolve_speech_prompts(
            self.model, prompt_samples, training=self.training
        )
        if resolved_prompt is not None:
            n_before = self._prompt_prefix_token_width(
                resolved_prompt, use_multi, spectrogram.device
            )

        speech_embeds, speech_atts = wrap_speech_with_prompts(
            self.model,
            speech_embeds,
            speech_atts,
            prompt_samples,
            training=self.training,
        )

        audio_start = 1 + n_before
        audio_layout = {"audio_start": audio_start, "n_audio": n_audio}
        return speech_embeds, speech_atts, spectrogram.device, audio_layout

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

    @staticmethod
    def _ddp_graph_anchor(
        device: torch.device,
        dtype: torch.dtype,
        *modules: Optional[nn.Module],
    ) -> torch.Tensor:
        """Zero loss term that touches trainable params (DDP static graph + conditional heads)."""
        terms: List[torch.Tensor] = []
        for mod in modules:
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad:
                    terms.append(p.sum().to(device=device, dtype=dtype) * 0.0)
        if not terms:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.stack(terms).sum()

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

        speech_embeds, speech_atts, device, audio_layout = self._encode_speech_for_sasv(samples)
        audio_start = audio_layout["audio_start"]
        n_audio_tokens = audio_layout["n_audio"]

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1], dtype=torch.long, device=device,
        ) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self._embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts], dim=1)

        # Determine whether we need hidden states for ArcFace / CE2 (CE1 always uses hidden states)
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

        # Forward LLM (BOS + speech only; 3-way head replaces full-vocab LM CE on answer tokens)
        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )

        answer_hidden = self._ce1_hidden(
            outputs.hidden_states[-1], attention_mask, audio_layout
        )
        answer_logits = self._answer_logits_from_hidden(answer_hidden)
        answer_targets = self._answer_targets_tensor(samples, device)
        ce1_w = self._ce1_class_weight_tensor(device)
        ce1_loss = F.cross_entropy(answer_logits, answer_targets, weight=ce1_w)

        result = {
            "loss": ce1_loss,
            "ce1_loss": ce1_loss,
            "ce_loss": ce1_loss,
        }

        # ---- ArcFace loss on enroll AND query ----
        # Enroll embedding: first N seconds (from enroll_num_samples)
        # Query embedding:  last  N seconds (from query_num_samples)
        if need_hidden:
            hidden = outputs.hidden_states[-1]  # (B, seq_len, H)
            # Audio lives at ``audio_start .. audio_start + n_audio_tokens - 1`` (not the whole ``speech_atts`` span).
            speech_start = int(audio_start)
            batch_n = hidden.shape[0]

            enroll_num_samples = samples.get("enroll_num_samples")
            query_num_samples = samples.get("query_num_samples")
            enroll_embed = None
            query_embed = None

            if enroll_num_samples is not None:
                enroll_tokens = [self._samples_to_qformer_tokens(ns) for ns in enroll_num_samples]
                enroll_starts = [speech_start] * batch_n
                enroll_embed = self._extract_segment_embedding_per_sample(
                    hidden, enroll_starts, enroll_tokens
                )

            if query_num_samples is not None:
                query_tokens = [self._samples_to_qformer_tokens(ns) for ns in query_num_samples]
                speech_token_len = int(n_audio_tokens)
                query_starts = [speech_start + max(0, speech_token_len - qt) for qt in query_tokens]
                query_embed = self._extract_segment_embedding_per_sample(
                    hidden, query_starts, query_tokens
                )

            ce2_losses = []
            if need_ce2:
                if self.ce2_apply_enroll and enroll_embed is not None:
                    enroll_logits = self.bonafide_spoof_head(enroll_embed)
                    enroll_targets = torch.zeros(batch_n, dtype=torch.long, device=device)
                    ce2_losses.append(F.cross_entropy(enroll_logits, enroll_targets))

                if query_embed is not None:
                    query_gt = samples.get("gt")
                    if isinstance(query_gt, list) and len(query_gt) == batch_n:
                        query_targets = torch.tensor(
                            [1 if self._is_spoof_gt(g) else 0 for g in query_gt],
                            dtype=torch.long,
                            device=device,
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
                    enroll_labels = enroll_labels.to(device)
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
                            query_labels = query_labels.to(device)
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
                        query_labels = query_labels.to(device)
                        query_embed_arc = self.speaker_projector(query_embed)
                        arcface_losses.append(self.arcface_head(query_embed_arc, query_labels))

            if arcface_losses:
                arcface_loss = sum(arcface_losses) / len(arcface_losses)
                result["arcface_loss"] = arcface_loss
                result["loss"] = result["loss"] + self.arcface_lambda * arcface_loss

        # DDP static graph: keep auxiliary / audio encoders in the graph every step.
        anchor_modules: List[Optional[nn.Module]] = []
        if self.bonafide_spoof_head is not None:
            anchor_modules.append(self.bonafide_spoof_head)
        if self.arcface_head is not None:
            anchor_modules.extend([self.arcface_head, self.speaker_projector])
        inner = self.model
        if getattr(inner, "use_speech_Qformer", False) and hasattr(inner, "speech_Qformer"):
            anchor_modules.append(inner.speech_Qformer)
        if getattr(inner, "speech_llama_proj", None) is not None:
            anchor_modules.append(inner.speech_llama_proj)
        if getattr(inner, "beats_path", None) and hasattr(inner, "beats"):
            anchor_modules.extend([inner.beats, inner.ln_audio])
        if any(p.requires_grad for p in inner.speech_encoder.parameters()):
            anchor_modules.append(inner.speech_encoder)
        result["loss"] = result["loss"] + self._ddp_graph_anchor(
            device, result["loss"].dtype, *anchor_modules
        )

        # Verbose: 3-way answer accuracy
        if verbose:
            with torch.no_grad():
                pred = answer_logits.argmax(dim=-1)
                correct = (pred == answer_targets).float().sum()
                total = int(batch_size)
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
            ce1_loss_per_sample : (B,) — per-sample 3-way yes/no/gen CE loss
        """
        # Adapt prompts
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        if "prompts" in samples and isinstance(samples["prompts"], list):
            samples["prompts"] = [p.replace("<Audio>", "<SpeechHere>") for p in samples["prompts"]]

        speech_embeds, speech_atts, device, audio_layout = self._encode_speech_for_sasv(samples)
        audio_start = audio_layout["audio_start"]
        n_audio_tokens = audio_layout["n_audio"]

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1], dtype=torch.long, device=device,
        ) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self._embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts], dim=1)

        # Forward LLM (always need hidden states for embeddings)
        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )

        result: Dict[str, torch.Tensor] = {}

        answer_hidden = self._ce1_hidden(
            outputs.hidden_states[-1], attention_mask, audio_layout
        )
        answer_logits = self._answer_logits_from_hidden(answer_hidden)
        answer_targets = self._answer_targets_tensor(samples, device)
        ce1_w = self._ce1_class_weight_tensor(device)
        result["ce1_loss_per_sample"] = F.cross_entropy(
            answer_logits, answer_targets, weight=ce1_w, reduction="none"
        )

        # --- Speaker embeddings & cosine similarity ---
        hidden = outputs.hidden_states[-1]
        speech_start = int(audio_start)

        enroll_num_samples = samples.get("enroll_num_samples")
        query_num_samples = samples.get("query_num_samples")

        enroll_embed = None
        query_embed = None

        if enroll_num_samples is not None:
            enroll_tokens = [self._samples_to_qformer_tokens(ns) for ns in enroll_num_samples]
            enroll_starts = [speech_start] * batch_size
            enroll_embed = self._extract_segment_embedding_per_sample(
                hidden, enroll_starts, enroll_tokens
            )

        if query_num_samples is not None:
            query_tokens = [self._samples_to_qformer_tokens(ns) for ns in query_num_samples]
            speech_token_len = int(n_audio_tokens)
            query_starts = [speech_start + max(0, speech_token_len - qt) for qt in query_tokens]
            query_embed = self._extract_segment_embedding_per_sample(
                hidden, query_starts, query_tokens
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

    def generate(self, samples: Dict[str, Any], generate_cfg: Dict[str, Any],
                 prompts: Optional[List[str]] = None, return_outputs: bool = False):
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        override_prompts: Optional[Union[List[str], str]] = None
        if prompts:
            override_prompts = [p.replace("<Audio>", "<SpeechHere>") for p in prompts]

        gen_cfg = dict(generate_cfg)

        num_return_sequences = gen_cfg.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            samples = {
                k: (v.repeat_interleave(num_return_sequences, dim=0) if isinstance(v, torch.Tensor)
                    else [x for x in v for _ in range(num_return_sequences)])
                for k, v in samples.items()
            }
            if isinstance(override_prompts, list):
                override_prompts = [p for p in override_prompts for _ in range(num_return_sequences)]

        speech_embeds, speech_atts, device, audio_layout = self._encode_speech_for_sasv(
            samples, override_prompts=override_prompts
        )
        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1], dtype=torch.long, device=device,
        ) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self._embed_tokens(bos)
        atts_bos = speech_atts[:, :1]
        inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts], dim=1)

        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        answer_hidden = self._ce1_hidden(
            outputs.hidden_states[-1], attention_mask, audio_layout
        )
        answer_logits = self._answer_logits_from_hidden(answer_hidden)

        temp = max(float(gen_cfg.get("temperature", 1.0)), 1e-8)
        if gen_cfg.get("do_sample", False):
            probs = F.softmax(answer_logits / temp, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            pred = answer_logits.argmax(dim=-1)

        texts = [self._ANSWER_CLASS_NAMES[int(i)] for i in pred.cpu()]

        if return_outputs:
            completion_ids = []
            for t in texts:
                ids = self.model.llama_tokenizer(t, add_special_tokens=False).input_ids
                completion_ids.append(torch.tensor(ids, dtype=torch.long, device=device))
            pad_id = self.model.llama_tokenizer.pad_token_id
            completion_ids = torch.nn.utils.rnn.pad_sequence(
                completion_ids, batch_first=True, padding_value=pad_id if pad_id is not None else 0,
            )
            return texts, completion_ids, answer_logits

        return texts

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


# Backward-compatible name for configs and imports (same implementation as SASVSalmonModel).
SalmonHierarchicalModel = BeheadedSASVSalmonModel
