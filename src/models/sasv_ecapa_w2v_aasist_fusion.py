"""
Classic SASV baseline: ECAPA2 + wav2vec2 (ASV) + AASIST (CM) with trainable MLP fusion.

- **ECAPA2 / ASV**: cosine similarity of ECAPA embeddings (enroll vs query).
- **wav2vec2 / ASV**: cosine similarity of SSL embeddings from the w2v frontend.
- **AASIST / CM**: P(bonafide) and P(spoof) on the query utterance.
- **Fusion head**: MLP over [ecapa_cos, w2v_cos, P(bonafide), P(spoof)] -> yes / no / gen.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .antispoofing import load_aasist3_checkpoint, load_ecapa2, load_w2v_aasist_checkpoint
from .base import Model
from .sasv_decision_tree_fusion import SASVDecisionTreeFusion
from .sasv_score_fusion import build_score_fusion_head, stack_ecapa_w2v_cm_scores
from .sasv_w2v_aasist_fusion import _ANSWER_CLASS_NAMES, _DEFAULT_CM_SAMPLES, _prep_cm_waveform


_DEFAULT_ECAPA_CROP_SECONDS = 5.0
_TRAINABLE_FUSION_MODES = frozenset({"mlp", "nonlinear", "score_mlp"})


def _prep_ecapa_waveform(wav: torch.Tensor, crop_seconds: float, target_sr: int = 16000) -> torch.Tensor:
    """Center-crop or zero-pad mono waveform to fixed ECAPA length [T]."""
    x = wav.float().flatten()
    max_len = int(crop_seconds * target_sr)
    if x.numel() >= max_len:
        start = (x.numel() - max_len) // 2
        x = x[start : start + max_len]
    else:
        x = F.pad(x, (0, max_len - x.numel()))
    return x


class SASVEcapaW2vAasistFusionModel(Model):
    """Frozen ECAPA2 + wav2vec2 + AASIST backbones with MLP or decision-tree fusion."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        ak = config.get("additional_kwargs", {}).get("sasv_baseline", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cm_samples = int(ak.get("cm_samples", _DEFAULT_CM_SAMPLES))
        self.ecapa_crop_seconds = float(ak.get("ecapa_crop_seconds", _DEFAULT_ECAPA_CROP_SECONDS))
        self.freeze_backbones = bool(ak.get("freeze_backbones", True))
        self.fusion_mode = str(ak.get("fusion_mode", "mlp")).lower()
        self.fusion_hidden_dim = int(ak.get("fusion_hidden_dim", 64))
        self.fusion_dropout = float(ak.get("fusion_dropout", 0.1))
        self.fusion_tree: Optional[SASVDecisionTreeFusion] = None

        self.ecapa = load_ecapa2(
            self.device,
            repo_id=ak.get("ecapa_repo", "Jenthe/ECAPA2"),
            filename=ak.get("ecapa_file", "ecapa2.pt"),
            weights_path=ak.get("ecapa_weights_path", ""),
        )

        self.cm_backend = str(ak.get("cm_backend", "lkb")).lower()
        ckpt = config.get("ckpt") or ak.get("cm_ckpt", "")
        if self.cm_backend == "aasist3":
            ckpt = ak.get("aasist3_cm_ckpt", "") or ckpt
        if self.cm_backend == "lkb" and not ckpt:
            raise ValueError(
                "SASVEcapaW2vAasistFusionModel (cm_backend=lkb) requires Model.ckpt or "
                "additional_kwargs.sasv_baseline.cm_ckpt"
            )

        if self.cm_backend == "aasist3":
            aasist3_repo = ak.get("aasist3_repo", "")
            if not aasist3_repo:
                raise ValueError(
                    "cm_backend=aasist3 requires additional_kwargs.sasv_baseline.aasist3_repo"
                )
            self.cm_model, audio_cfg = load_aasist3_checkpoint(
                ckpt,
                aasist3_repo,
                self.device,
                hf_model_id=ak.get("aasist3_hf_id", "MTUCI/AASIST3"),
                w2v_cache_dir=ak.get("aasist3_w2v_cache_dir", ""),
                load_pretrained=bool(ak.get("aasist3_load_pretrained", True)),
            )
        elif self.cm_backend == "lkb":
            self.antispoofing_repo = ak.get("antispoofing_repo", "")
            if not self.antispoofing_repo:
                raise ValueError(
                    "cm_backend=lkb requires additional_kwargs.sasv_baseline.antispoofing_repo"
                )
            self.cm_model, audio_cfg = load_w2v_aasist_checkpoint(
                ckpt,
                self.antispoofing_repo,
                self.device,
                xlsr_weights_path=ak.get("xlsr_weights_path", ""),
            )
        else:
            raise ValueError(
                f"Unknown cm_backend {self.cm_backend!r}; use 'lkb' or 'aasist3'"
            )
        self.cm_samples = int(audio_cfg.get("samples", self.cm_samples))

        if self.freeze_backbones:
            for p in self.ecapa.parameters():
                p.requires_grad_(False)
            self.cm_model.requires_grad_(False)

        if self.fusion_mode == "tree":
            self.fusion_head = None
            tree_path = str(ak.get("fusion_tree_path", "") or "").strip()
            if tree_path and os.path.isfile(tree_path):
                self.fusion_tree = SASVDecisionTreeFusion.load(tree_path)
                logging.info("Loaded decision tree fusion from %s", tree_path)
            else:
                self.fusion_tree = None
                if tree_path:
                    logging.warning(
                        "fusion_mode=tree but fusion_tree_path not found (%s); "
                        "inference will fail until the tree is trained",
                        tree_path,
                    )
        elif self.fusion_mode in _TRAINABLE_FUSION_MODES:
            self.fusion_head = build_score_fusion_head(
                ak, layout="ecapa_w2v_cm"
            ).to(self.device)
        else:
            raise ValueError(
                f"Unknown fusion_mode {self.fusion_mode!r}; "
                "use 'mlp', 'nonlinear', 'score_mlp', or 'tree'"
            )

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
            self._load_fusion_checkpoint(fusion_ckpt)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            "SASVEcapaW2vAasistFusionModel: cm_backend=%s fusion_mode=%s cm_ckpt=%s "
            "freeze_backbones=%s trainable_params=%s",
            self.cm_backend,
            self.fusion_mode,
            ckpt or "(hf only)",
            self.freeze_backbones,
            n_trainable,
        )

    def _load_fusion_checkpoint(self, fusion_ckpt: str) -> None:
        raw = torch.load(fusion_ckpt, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "fusion_head" in raw:
            self.fusion_head.load_state_dict(raw["fusion_head"], strict=True)
            logging.info("Loaded fusion_head from %s", fusion_ckpt)
            return
        state = raw.get("model", raw) if isinstance(raw, dict) else {}
        if isinstance(state, dict):
            fusion_sd = {
                k[len("fusion_head.") :]: v
                for k, v in state.items()
                if k.startswith("fusion_head.")
            }
            if fusion_sd:
                self.fusion_head.load_state_dict(fusion_sd, strict=True)
                logging.info("Loaded fusion_head from checkpoint %s", fusion_ckpt)

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
            "Batch must contain enroll_wav and query_wav "
            "(use model_name=sasv_ecapa_w2v_aasist collate)"
        )

    def _ecapa_embedding(self, wav_batch: torch.Tensor) -> torch.Tensor:
        """ECAPA2 embeddings [B, D]."""
        embs: List[torch.Tensor] = []
        ctx = torch.no_grad() if self.freeze_backbones else torch.enable_grad()
        with ctx:
            if self.freeze_backbones:
                self.ecapa.eval()
            for i in range(wav_batch.size(0)):
                prepared = _prep_ecapa_waveform(wav_batch[i], self.ecapa_crop_seconds)
                emb = self.ecapa(prepared.unsqueeze(0).float())
                embs.append(F.normalize(emb.float(), p=2, dim=-1).squeeze(0))
        return torch.stack(embs, dim=0)

    def _w2v_embedding(self, wav_batch: torch.Tensor) -> torch.Tensor:
        """Mean-pooled normalized wav2vec2 SSL embedding [B, D]."""
        prepared = torch.stack(
            [_prep_cm_waveform(wav_batch[i], self.cm_samples) for i in range(wav_batch.size(0))],
            dim=0,
        )
        ctx = torch.no_grad() if self.freeze_backbones else torch.enable_grad()
        with ctx:
            if self.freeze_backbones:
                self.cm_model.eval()
            if self.cm_backend == "aasist3":
                emb = self.cm_model.w2v_encoder(prepared)
            else:
                if self.freeze_backbones:
                    self.cm_model.frontend.eval()
                emb = self.cm_model.frontend.extract_feature(prepared)
        pooled = emb.mean(dim=1)
        return F.normalize(pooled.float(), p=2, dim=1)

    def _cm_probs(self, query_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bonafide / spoof probabilities from AASIST CM head on query [B]."""
        prepared = torch.stack(
            [_prep_cm_waveform(query_wav[i], self.cm_samples) for i in range(query_wav.size(0))],
            dim=0,
        )
        ctx = torch.no_grad() if self.freeze_backbones else torch.enable_grad()
        with ctx:
            if self.freeze_backbones:
                self.cm_model.eval()
            out = self.cm_model(prepared)
            logits = out[1] if isinstance(out, tuple) else out
        probs = F.softmax(logits.float(), dim=-1)
        return probs[:, 1], probs[:, 0]

    def _fusion_features(
        self,
        enroll_wav: torch.Tensor,
        query_wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enroll_ecapa = self._ecapa_embedding(enroll_wav)
        query_ecapa = self._ecapa_embedding(query_wav)
        ecapa_cos = F.cosine_similarity(enroll_ecapa, query_ecapa, dim=1)

        enroll_w2v = self._w2v_embedding(enroll_wav)
        query_w2v = self._w2v_embedding(query_wav)
        w2v_cos = F.cosine_similarity(enroll_w2v, query_w2v, dim=1)

        bonafide_prob, spoof_prob = self._cm_probs(query_wav)
        feats = stack_ecapa_w2v_cm_scores(ecapa_cos, w2v_cos, bonafide_prob, spoof_prob)
        return feats, ecapa_cos, w2v_cos, bonafide_prob

    def extract_fusion_features(
        self, enroll_wav: torch.Tensor, query_wav: torch.Tensor
    ) -> torch.Tensor:
        """Return fusion feature matrix [B, 4] for offline tree training."""
        feats, _, _, _ = self._fusion_features(enroll_wav, query_wav)
        return feats.float()

    def _fused_logits(self, enroll_wav: torch.Tensor, query_wav: torch.Tensor) -> torch.Tensor:
        feats, _, _, _ = self._fusion_features(enroll_wav, query_wav)
        if self.fusion_tree is not None:
            return self.fusion_tree.predict_logits_tensor(feats)
        if self.fusion_head is not None:
            return self.fusion_head(feats.float())
        raise RuntimeError(
            "fusion_mode=tree but no fusion_tree loaded; set fusion_tree_path or train a tree first"
        )

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
        feats, ecapa_cos, w2v_cos, bonafide_prob = self._fusion_features(enroll_wav, query_wav)
        logits = self._fused_logits(enroll_wav, query_wav)
        composite_cos = 0.5 * (ecapa_cos + w2v_cos)
        out: Dict[str, torch.Tensor] = {
            "ecapa_cosine_sim": ecapa_cos,
            "w2v_cosine_sim": w2v_cos,
            "cosine_sim": composite_cos,
            "bonafide_prob": bonafide_prob,
            "fused_logits": logits,
        }
        labels = samples.get("answer") or samples.get("gt")
        if labels is not None:
            try:
                out["ce1_loss_per_sample"] = F.cross_entropy(
                    logits,
                    self._targets_tensor(samples, logits.device),
                    reduction="none",
                )
            except ValueError:
                pass
        return out

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
