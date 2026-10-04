"""
SASV SALMONN with cascade-style fusion head (ASV cosine + CM + optional CE1 hints).

Primary training loss is cross-entropy on fused 3-way logits (yes / no / gen), aligned with
ResNet-TDNN + CM fusion baselines. Optional auxiliary CE2 / ArcFace match the parent model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .salmon_hierarchical import BeheadedSASVSalmonModel


class BeheadedSASVSalmonFusionModel(BeheadedSASVSalmonModel):
    """SALMONN SASV with ``fusion_head`` over verifier-style scalar features.

    Config keys (under ``Model.additional_kwargs.salmon``), in addition to parent keys:

        fusion_hidden_dim: int — MLP hidden size (default 32)
        fusion_use_cm_logits: bool — append raw 2-way bonafide/spoof logits (default True)
        fusion_use_answer_logits: bool — append 3-way answer_head logits (default True)
        direct_ce1_lambda: float — optional weight for direct answer_head CE (default 0.0)
        fusion_primary: bool — log ``ce1_loss`` from fusion CE when True (default True)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        sal_cfg = config.get("additional_kwargs", {}).get("salmon", {})

        self.fusion_hidden_dim = int(sal_cfg.get("fusion_hidden_dim", 32))
        self.fusion_use_cm_logits = bool(sal_cfg.get("fusion_use_cm_logits", True))
        self.fusion_use_answer_logits = bool(sal_cfg.get("fusion_use_answer_logits", True))
        self.direct_ce1_lambda = float(sal_cfg.get("direct_ce1_lambda", 0.0))
        self.fusion_primary = bool(sal_cfg.get("fusion_primary", True))

        in_dim = self._fusion_input_dim()
        self.fusion_head = nn.Sequential(
            nn.Linear(in_dim, self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_hidden_dim, 3),
        )
        logging.info(
            "SASV fusion head: in_dim=%s hidden=%s -> 3 (direct_ce1_lambda=%s)",
            in_dim,
            self.fusion_hidden_dim,
            self.direct_ce1_lambda,
        )

    def _fusion_input_dim(self) -> int:
        dim = 1  # cosine_sim(enroll, query)
        if self.bonafide_spoof_head is not None:
            dim += 2  # P(bonafide), P(spoof) on query
            if self.fusion_use_cm_logits:
                dim += 2
        if self.fusion_use_answer_logits:
            dim += 3
        return dim

    def _extract_enroll_query_embeds(
        self,
        hidden: torch.Tensor,
        audio_start: int,
        n_audio_tokens: int,
        batch_n: int,
        samples: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        speech_start = int(audio_start)
        enroll_embed = None
        query_embed = None

        enroll_num_samples = samples.get("enroll_num_samples")
        query_num_samples = samples.get("query_num_samples")

        if enroll_num_samples is not None:
            enroll_tokens = [self._samples_to_qformer_tokens(ns) for ns in enroll_num_samples]
            enroll_starts = [speech_start] * batch_n
            enroll_embed = self._extract_segment_embedding_per_sample(
                hidden, enroll_starts, enroll_tokens
            )

        if query_num_samples is not None:
            query_tokens = [self._samples_to_qformer_tokens(ns) for ns in query_num_samples]
            speech_token_len = int(n_audio_tokens)
            query_starts = [
                speech_start + max(0, speech_token_len - qt) for qt in query_tokens
            ]
            query_embed = self._extract_segment_embedding_per_sample(
                hidden, query_starts, query_tokens
            )

        return enroll_embed, query_embed

    def _enroll_query_cosine_sim(
        self,
        enroll_embed: Optional[torch.Tensor],
        query_embed: Optional[torch.Tensor],
        device: torch.device,
        batch_n: int,
    ) -> torch.Tensor:
        if enroll_embed is None or query_embed is None:
            return torch.full((batch_n,), 0.5, device=device, dtype=torch.float32)

        if self.speaker_projector is not None:
            enroll_proj = F.normalize(self.speaker_projector(enroll_embed.float()), p=2, dim=1)
            query_proj = F.normalize(self.speaker_projector(query_embed.float()), p=2, dim=1)
        else:
            enroll_proj = F.normalize(enroll_embed.float(), p=2, dim=1)
            query_proj = F.normalize(query_embed.float(), p=2, dim=1)
        return F.cosine_similarity(enroll_proj, query_proj, dim=1)

    def _build_fusion_features(
        self,
        cosine_sim: torch.Tensor,
        query_embed: Optional[torch.Tensor],
        answer_logits: Optional[torch.Tensor],
        device: torch.device,
        batch_n: int,
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = [cosine_sim.unsqueeze(-1).float()]

        if self.bonafide_spoof_head is not None and query_embed is not None:
            cm_logits = self.bonafide_spoof_head(query_embed.float())
            feats.append(F.softmax(cm_logits, dim=-1))
            if self.fusion_use_cm_logits:
                feats.append(cm_logits)
        elif self.bonafide_spoof_head is not None:
            feats.append(torch.full((batch_n, 2), 0.5, device=device, dtype=torch.float32))
            if self.fusion_use_cm_logits:
                feats.append(torch.zeros(batch_n, 2, device=device, dtype=torch.float32))

        if self.fusion_use_answer_logits and answer_logits is not None:
            feats.append(answer_logits.float())

        return torch.cat(feats, dim=-1)

    def _fused_logits_from_batch(
        self,
        samples: Dict[str, Any],
        llm_outputs,
        attention_mask: torch.Tensor,
        audio_layout: Dict[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.device]:
        device = attention_mask.device
        batch_n = attention_mask.shape[0]
        audio_start = audio_layout["audio_start"]
        n_audio_tokens = audio_layout["n_audio"]

        answer_hidden = self._ce1_hidden(
            llm_outputs.hidden_states[-1], attention_mask, audio_layout
        )
        answer_logits = self._answer_logits_from_hidden(answer_hidden)

        hidden = llm_outputs.hidden_states[-1]
        enroll_embed, query_embed = self._extract_enroll_query_embeds(
            hidden, audio_start, n_audio_tokens, batch_n, samples
        )
        cosine_sim = self._enroll_query_cosine_sim(enroll_embed, query_embed, device, batch_n)
        fusion_feats = self._build_fusion_features(
            cosine_sim, query_embed, answer_logits, device, batch_n
        )
        return self.fusion_head(fusion_feats), answer_logits, enroll_embed, query_embed, device

    def _auxiliary_losses(
        self,
        result: Dict[str, torch.Tensor],
        samples: Dict[str, Any],
        device: torch.device,
        batch_n: int,
        enroll_embed: Optional[torch.Tensor],
        query_embed: Optional[torch.Tensor],
    ) -> None:
        ce2_losses: List[torch.Tensor] = []
        if self.bonafide_spoof_head is not None:
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

        arcface_losses: List[torch.Tensor] = []
        enroll_spk_ids = samples.get("enroll_speaker_ids")
        if enroll_spk_ids is not None and samples.get("enroll_num_samples") is not None:
            enroll_labels = self._get_speaker_indices(enroll_spk_ids)
            if enroll_labels is not None and enroll_embed is not None and self.arcface_head is not None:
                enroll_labels = enroll_labels.to(device)
                arcface_losses.append(
                    self.arcface_head(self.speaker_projector(enroll_embed), enroll_labels)
                )

        query_spk_ids = samples.get("query_speaker_ids")
        if query_spk_ids is not None and samples.get("query_num_samples") is not None:
            query_gt = samples.get("gt")
            if isinstance(query_gt, list) and len(query_gt) == batch_n:
                bonafide_indices = [i for i, g in enumerate(query_gt) if not self._is_spoof_gt(g)]
                if bonafide_indices and query_embed is not None and self.arcface_head is not None:
                    query_spk_ids_bonafide = [query_spk_ids[i] for i in bonafide_indices]
                    query_labels = self._get_speaker_indices(query_spk_ids_bonafide)
                    if query_labels is not None:
                        idx_t = torch.tensor(bonafide_indices, dtype=torch.long, device=device)
                        query_embed_bonafide = query_embed.index_select(0, idx_t)
                        arcface_losses.append(
                            self.arcface_head(
                                self.speaker_projector(query_embed_bonafide),
                                query_labels.to(device),
                            )
                        )
            elif self.arcface_head is not None:
                query_labels = self._get_speaker_indices(query_spk_ids)
                if query_labels is not None and query_embed is not None:
                    arcface_losses.append(
                        self.arcface_head(
                            self.speaker_projector(query_embed), query_labels.to(device)
                        )
                    )

        if arcface_losses:
            arcface_loss = sum(arcface_losses) / len(arcface_losses)
            result["arcface_loss"] = arcface_loss
            result["loss"] = result["loss"] + self.arcface_lambda * arcface_loss

    def _ddp_anchor_modules(self) -> List[Optional[nn.Module]]:
        anchor_modules: List[Optional[nn.Module]] = [self.fusion_head]
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
        return anchor_modules

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        if "prompts" in samples and isinstance(samples["prompts"], list):
            samples["prompts"] = [p.replace("<Audio>", "<SpeechHere>") for p in samples["prompts"]]

        speech_embeds, speech_atts, device, audio_layout = self._encode_speech_for_sasv(samples)
        batch_size = speech_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * (
            self.model.llama_tokenizer.bos_token_id
        )
        bos_embeds = self._embed_tokens(bos)
        atts_bos = speech_atts[:, :1]
        inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts], dim=1)

        with self.model.maybe_autocast():
            llm_outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )

        fused_logits, answer_logits, enroll_embed, query_embed, device = self._fused_logits_from_batch(
            samples, llm_outputs, attention_mask, audio_layout
        )
        answer_targets = self._answer_targets_tensor(samples, device)
        ce1_w = self._ce1_class_weight_tensor(device)

        fusion_loss = F.cross_entropy(fused_logits, answer_targets, weight=ce1_w)
        result: Dict[str, torch.Tensor] = {
            "loss": fusion_loss,
            "fusion_loss": fusion_loss,
            "fused_logits": fused_logits,
        }

        if self.fusion_primary:
            result["ce1_loss"] = fusion_loss
            result["ce_loss"] = fusion_loss
        else:
            direct_loss = F.cross_entropy(answer_logits, answer_targets, weight=ce1_w)
            result["ce1_loss"] = direct_loss
            result["ce_loss"] = direct_loss
            result["loss"] = direct_loss

        if self.direct_ce1_lambda > 0.0:
            direct_loss = F.cross_entropy(answer_logits, answer_targets, weight=ce1_w)
            result["direct_ce1_loss"] = direct_loss
            result["loss"] = result["loss"] + self.direct_ce1_lambda * direct_loss

        self._auxiliary_losses(result, samples, device, batch_size, enroll_embed, query_embed)
        result["loss"] = result["loss"] + self._ddp_graph_anchor(
            device, result["loss"].dtype, *self._ddp_anchor_modules()
        )

        if verbose:
            with torch.no_grad():
                pred = fused_logits.argmax(dim=-1)
                result["correct"] = (pred == answer_targets).float().sum()
                result["total"] = torch.tensor(batch_size, device=device)

        return result

    def generate(
        self,
        samples: Dict[str, Any],
        generate_cfg: Dict[str, Any],
        prompts: Optional[List[str]] = None,
        return_outputs: bool = False,
    ):
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        override_prompts: Optional[Union[List[str], str]] = None
        if prompts:
            override_prompts = [p.replace("<Audio>", "<SpeechHere>") for p in prompts]

        gen_cfg = dict(generate_cfg)
        num_return_sequences = gen_cfg.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            samples = {
                k: (
                    v.repeat_interleave(num_return_sequences, dim=0)
                    if isinstance(v, torch.Tensor)
                    else [x for x in v for _ in range(num_return_sequences)]
                )
                for k, v in samples.items()
            }
            if isinstance(override_prompts, list):
                override_prompts = [
                    p for p in override_prompts for _ in range(num_return_sequences)
                ]

        speech_embeds, speech_atts, device, audio_layout = self._encode_speech_for_sasv(
            samples, override_prompts=override_prompts
        )
        batch_size = speech_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * (
            self.model.llama_tokenizer.bos_token_id
        )
        bos_embeds = self._embed_tokens(bos)
        atts_bos = speech_atts[:, :1]
        inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts], dim=1)

        with self.model.maybe_autocast():
            llm_outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )

        fused_logits, _, _, _, _ = self._fused_logits_from_batch(
            samples, llm_outputs, attention_mask, audio_layout
        )

        temp = max(float(gen_cfg.get("temperature", 1.0)), 1e-8)
        if gen_cfg.get("do_sample", False):
            pred = torch.multinomial(F.softmax(fused_logits / temp, dim=-1), num_samples=1).squeeze(-1)
        else:
            pred = fused_logits.argmax(dim=-1)

        texts = [self._ANSWER_CLASS_NAMES[int(i)] for i in pred.cpu()]

        if return_outputs:
            completion_ids = []
            for t in texts:
                ids = self.model.llama_tokenizer(t, add_special_tokens=False).input_ids
                completion_ids.append(torch.tensor(ids, dtype=torch.long, device=device))
            pad_id = self.model.llama_tokenizer.pad_token_id
            completion_ids = torch.nn.utils.rnn.pad_sequence(
                completion_ids,
                batch_first=True,
                padding_value=pad_id if pad_id is not None else 0,
            )
            return texts, completion_ids, fused_logits

        return texts


SalmonHierarchicalFusionModel = BeheadedSASVSalmonFusionModel
