"""
SASV baseline: score fusion between wav2vec2 (ASV) and AASIST (CM).

- **wav2vec2 / ASV**: cosine similarity of SSL embeddings (enroll vs query).
- **AASIST / CM**: P(bonafide) from the antispoofing head on the query utterance.
- **Fusion head**: small MLP over [cosine_sim, P(bonafide), P(spoof)] -> yes / no / gen.

Integrates with :class:`~src.epochs.sasv_eval_epoch.SASVEvalEpoch` and
:class:`~src.epochs.test_epoch.TestEpoch` via ``generate`` (3-way logits) and
``score_pairs`` (cosine_sim + bonafide_prob).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .antispoofing import load_w2v_aasist_checkpoint
from .base import Model
from .sasv_score_fusion import build_score_fusion_head, stack_w2v_cm_scores


_ANSWER_CLASS_NAMES = ("yes", "no", "gen")
_DEFAULT_CM_SAMPLES = 64600


def _prep_cm_waveform(wav: torch.Tensor, max_len: int = _DEFAULT_CM_SAMPLES) -> torch.Tensor:
    """Center-crop or repeat-pad to fixed length + mean normalization (CM inference style)."""
    x = wav.float().flatten()
    if x.numel() >= max_len:
        start = (x.numel() - max_len) // 2
        x = x[start : start + max_len]
    else:
        reps = int(max_len / max(x.numel(), 1)) + 1
        x = x.repeat(reps)[:max_len]
    denom = torch.clamp(x.max() - x.min(), min=1e-12)
    return (x - x.mean()) / denom


class SASVW2vAasistFusionModel(Model):
    """Frozen wav2vec2+AASIST backbones with trainable SASV fusion head."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        ak = config.get("additional_kwargs", {}).get("w2v_aasist", {})

        self.antispoofing_repo = ak.get("antispoofing_repo", "")
        if not self.antispoofing_repo:
            raise ValueError(
                "SASVW2vAasistFusionModel requires additional_kwargs.w2v_aasist.antispoofing_repo"
            )
        ckpt = config.get("ckpt") or ak.get("cm_ckpt", "")
        if not ckpt:
            raise ValueError(
                "SASVW2vAasistFusionModel requires Model.ckpt or "
                "additional_kwargs.w2v_aasist.cm_ckpt"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cm_samples = int(ak.get("cm_samples", _DEFAULT_CM_SAMPLES))
        self.freeze_backbones = bool(ak.get("freeze_backbones", True))
        self.fusion_mode = str(ak.get("fusion_mode", "heuristic")).lower()
        self.fusion_hidden_dim = int(ak.get("fusion_hidden_dim", 32))
        self.fusion_dropout = float(ak.get("fusion_dropout", 0.1))

        self.cm_model, audio_cfg = load_w2v_aasist_checkpoint(
            ckpt,
            self.antispoofing_repo,
            self.device,
            xlsr_weights_path=ak.get("xlsr_weights_path", ""),
        )
        self.cm_samples = int(audio_cfg.get("samples", self.cm_samples))

        if self.freeze_backbones:
            self.cm_model.requires_grad_(False)

        if self.fusion_mode in ("mlp", "nonlinear", "score_mlp"):
            self.fusion_head: Optional[nn.Module] = build_score_fusion_head(
                ak, layout="w2v_cm"
            )
            self.logit_bias: Optional[nn.Parameter] = None
        else:
            self.fusion_head = None
            # Optional tiny calibration on top of heuristic fusion (3 scalars)
            if bool(ak.get("trainable_logit_bias", False)):
                self.logit_bias = nn.Parameter(torch.zeros(3))
            else:
                self.logit_bias = None

        sasv_w = ak.get("sasv_class_weights")
        self._class_weights: Optional[torch.Tensor] = None
        if isinstance(sasv_w, dict):
            self._class_weights = torch.tensor(
                [
                    float(sasv_w.get("yes", 1.0)),
                    float(sasv_w.get("no", 1.0)),
                    float(sasv_w.get("gen", 1.0)),
                ],
                dtype=torch.float32,
            )

        fusion_ckpt = ak.get("fusion_ckpt", "")
        if fusion_ckpt and self.fusion_head is not None and os.path.isfile(fusion_ckpt):
            raw = torch.load(fusion_ckpt, map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and "fusion_head" in raw:
                self.fusion_head.load_state_dict(raw["fusion_head"], strict=True)
                logging.info("Loaded fusion_head from %s", fusion_ckpt)
            elif isinstance(raw, dict) and "model" in raw:
                fusion_sd = {
                    k[len("fusion_head.") :]: v
                    for k, v in raw["model"].items()
                    if k.startswith("fusion_head.")
                }
                if fusion_sd:
                    self.fusion_head.load_state_dict(fusion_sd, strict=True)
                    logging.info("Loaded fusion_head from SASV checkpoint %s", fusion_ckpt)
            elif isinstance(raw, dict):
                fusion_sd = {
                    k[len("fusion_head.") :]: v
                    for k, v in raw.items()
                    if k.startswith("fusion_head.")
                }
                if fusion_sd:
                    self.fusion_head.load_state_dict(fusion_sd, strict=True)
                    logging.info("Loaded fusion_head weights from %s", fusion_ckpt)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            "SASVW2vAasistFusionModel: cm_ckpt=%s fusion_mode=%s freeze_backbones=%s "
            "trainable_params=%s",
            ckpt,
            self.fusion_mode,
            self.freeze_backbones,
            n_trainable,
        )

    @staticmethod
    def has_trainable_parameters(module: nn.Module) -> bool:
        return any(p.requires_grad for p in module.parameters())

    @staticmethod
    def _answer_class_from_label(label: Any) -> int:
        raw = str(label).strip()
        s = raw.lower()
        m = re.search(
            r"<answer>\s*(yes|no|gen|spoof|verified|rejected)\s*</answer>",
            s,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            s = m.group(1).lower()
        if s in ("verified", "yes"):
            return 0
        if s in ("rejected", "no"):
            return 1
        if s in ("gen", "spoof"):
            return 2
        raise ValueError(f"Unknown SASV label {label!r}")

    def _targets_tensor(self, samples: Dict[str, Any], device: torch.device) -> torch.Tensor:
        labels = samples.get("answer") or samples.get("gt") or samples.get("text")
        if not isinstance(labels, list):
            labels = [labels]
        idx = [self._answer_class_from_label(t) for t in labels]
        return torch.tensor(idx, dtype=torch.long, device=device)

    def _get_enroll_query_wavs(
        self, samples: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "enroll_wav" in samples and "query_wav" in samples:
            enroll = samples["enroll_wav"].to(self.device)
            query = samples["query_wav"].to(self.device)
            return enroll, query

        raise KeyError(
            "Batch must contain enroll_wav and query_wav (use model_name=sasv_w2v_aasist collate)"
        )

    @torch.inference_mode()
    def _ssl_embedding(self, wav_batch: torch.Tensor) -> torch.Tensor:
        """Mean-pooled normalized SSL embedding from wav2vec2 frontend [B, D]."""
        prepared = torch.stack(
            [_prep_cm_waveform(wav_batch[i], self.cm_samples) for i in range(wav_batch.size(0))],
            dim=0,
        )
        if self.freeze_backbones:
            self.cm_model.frontend.eval()
            with torch.no_grad():
                emb = self.cm_model.frontend.extract_feature(prepared)
        else:
            emb = self.cm_model.frontend.extract_feature(prepared)
        pooled = emb.mean(dim=1)
        return F.normalize(pooled.float(), p=2, dim=1)

    @torch.inference_mode()
    def _cm_probs(self, query_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bonafide / spoof probabilities from AASIST CM head on query [B]."""
        prepared = torch.stack(
            [_prep_cm_waveform(query_wav[i], self.cm_samples) for i in range(query_wav.size(0))],
            dim=0,
        )
        if self.freeze_backbones:
            self.cm_model.eval()
            with torch.no_grad():
                _, logits = self.cm_model(prepared)
        else:
            _, logits = self.cm_model(prepared)
        probs = F.softmax(logits.float(), dim=-1)
        return probs[:, 1], probs[:, 0]

    def _fusion_features(
        self,
        enroll_wav: torch.Tensor,
        query_wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enroll_emb = self._ssl_embedding(enroll_wav)
        query_emb = self._ssl_embedding(query_wav)
        cosine_sim = F.cosine_similarity(enroll_emb, query_emb, dim=1)
        bonafide_prob, spoof_prob = self._cm_probs(query_wav)
        feats = stack_w2v_cm_scores(cosine_sim, bonafide_prob, spoof_prob)
        return feats, cosine_sim, bonafide_prob

    def _heuristic_logits(
        self,
        cosine_sim: torch.Tensor,
        bonafide_prob: torch.Tensor,
        spoof_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Product-of-experts style fusion (no trainable params)."""
        eps = 1e-6
        cos = cosine_sim.clamp(eps, 1.0 - eps)
        bon = bonafide_prob.clamp(eps, 1.0 - eps)
        spf = spoof_prob.clamp(eps, 1.0 - eps)
        yes_logit = torch.log(cos) + torch.log(bon)
        no_logit = torch.log(1.0 - cos) + torch.log(bon)
        gen_logit = torch.log(spf)
        logits = torch.stack([yes_logit, no_logit, gen_logit], dim=-1)
        if self.logit_bias is not None:
            logits = logits + self.logit_bias
        return logits

    def _fused_logits(self, enroll_wav: torch.Tensor, query_wav: torch.Tensor) -> torch.Tensor:
        feats, cosine_sim, bonafide_prob = self._fusion_features(enroll_wav, query_wav)
        if self.fusion_head is not None:
            return self.fusion_head(feats.float())
        spoof_prob = feats[:, 2]
        return self._heuristic_logits(cosine_sim, bonafide_prob, spoof_prob)

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        enroll_wav, query_wav = self._get_enroll_query_wavs(samples)
        device = enroll_wav.device
        logits = self._fused_logits(enroll_wav, query_wav)
        targets = self._targets_tensor(samples, device)
        weight = self._class_weights.to(device) if self._class_weights is not None else None
        loss = F.cross_entropy(logits, targets, weight=weight)
        result: Dict[str, torch.Tensor] = {
            "loss": loss,
            "fusion_loss": loss,
            "ce1_loss": loss,
            "fused_logits": logits,
        }
        if verbose:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                result["correct"] = (pred == targets).float().sum()
                result["total"] = torch.tensor(logits.size(0), device=device)
        return result

    def score_pairs(self, samples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        enroll_wav, query_wav = self._get_enroll_query_wavs(samples)
        feats, cosine_sim, bonafide_prob = self._fusion_features(enroll_wav, query_wav)
        logits = self._fused_logits(enroll_wav, query_wav)
        return {
            "cosine_sim": cosine_sim,
            "bonafide_prob": bonafide_prob,
            "fused_logits": logits,
            "ce1_loss_per_sample": F.cross_entropy(
                logits,
                self._targets_tensor(samples, logits.device),
                reduction="none",
            ),
        }

    def generate(
        self,
        samples: Dict[str, Any],
        generate_cfg: Dict[str, Any],
        prompts: Optional[List[str]] = None,
        return_outputs: bool = False,
    ) -> Union[List[str], Tuple[List[str], torch.Tensor, torch.Tensor]]:
        enroll_wav, query_wav = self._get_enroll_query_wavs(samples)
        logits = self._fused_logits(enroll_wav, query_wav)
        temp = max(float(generate_cfg.get("temperature", 1.0)), 1e-8)
        if generate_cfg.get("do_sample", False):
            probs = F.softmax(logits / temp, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            pred = logits.argmax(dim=-1)
        texts = [_ANSWER_CLASS_NAMES[int(i)] for i in pred.cpu()]
        if not return_outputs:
            return texts
        batch_size = logits.size(0)
        completion_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=logits.device)
        return texts, completion_ids, logits

    def compute_logits_for_completions(
        self, samples: Dict[str, Any], completion_ids: torch.Tensor
    ) -> torch.Tensor:
        enroll_wav, query_wav = self._get_enroll_query_wavs(samples)
        return self._fused_logits(enroll_wav, query_wav)

    def generate_text_only(
        self, prompt_texts: List[str], generate_cfg: Dict[str, Any]
    ) -> List[str]:
        return ["no"] * len(prompt_texts)

    def get_tokenizer(self):
        return None
