"""Sklearn decision tree fusion over ECAPA + w2v + CM features."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.tree import DecisionTreeClassifier, export_text
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sasv_decision_tree_fusion requires scikit-learn and joblib "
        "(see requirements.txt)."
    ) from exc


FEATURE_NAMES: Tuple[str, ...] = ("ecapa_cos", "w2v_cos", "bonafide_prob", "spoof_prob")
CLASS_NAMES: Tuple[str, ...] = ("yes", "no", "gen")
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class TreeTrainResult:
    train_accuracy: float
    val_accuracy: float
    train_balanced_accuracy: float
    val_balanced_accuracy: float
    val_report: str
    tree_depth: int
    n_leaves: int
    n_train: int
    n_val: int


def label_to_idx(label: Any) -> int:
    raw = str(label).strip().lower()
    if raw in ("verified", "yes"):
        return 0
    if raw in ("rejected", "no"):
        return 1
    if raw in ("gen", "spoof"):
        return 2
    raise ValueError(f"Unknown SASV label {label!r}")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accs: List[float] = []
    for cls in range(len(CLASS_NAMES)):
        mask = y_true == cls
        if mask.any():
            accs.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(accs)) if accs else 0.0


class SASVDecisionTreeFusion:
    """Wrapper around sklearn DecisionTreeClassifier for SASV fusion."""

    def __init__(self, classifier: DecisionTreeClassifier):
        self.clf = classifier

    @classmethod
    def train(
        cls,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        *,
        max_depth: Optional[int] = 8,
        min_samples_leaf: int = 20,
        min_samples_split: int = 40,
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
        random_state: int = 42,
    ) -> Tuple["SASVDecisionTreeFusion", TreeTrainResult]:
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state,
        )
        clf.fit(x_train, y_train)
        model = cls(clf)

        train_pred = clf.predict(x_train)
        train_acc = float(accuracy_score(y_train, train_pred))
        train_bal = balanced_accuracy(y_train, train_pred)

        if x_val is not None and y_val is not None and len(y_val) > 0:
            val_pred = clf.predict(x_val)
            val_acc = float(accuracy_score(y_val, val_pred))
            val_bal = balanced_accuracy(y_val, val_pred)
            val_report = classification_report(
                y_val,
                val_pred,
                target_names=list(CLASS_NAMES),
                digits=4,
                zero_division=0,
            )
            n_val = int(len(y_val))
        else:
            val_acc = 0.0
            val_bal = 0.0
            val_report = ""
            n_val = 0

        result = TreeTrainResult(
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            train_balanced_accuracy=train_bal,
            val_balanced_accuracy=val_bal,
            val_report=val_report,
            tree_depth=int(clf.get_depth()),
            n_leaves=int(clf.get_n_leaves()),
            n_train=int(len(y_train)),
            n_val=n_val,
        )
        return model, result

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        proba = self.clf.predict_proba(features)
        # sklearn columns follow sorted(classes_); remap to yes=0, no=1, gen=2
        out = np.zeros((features.shape[0], len(CLASS_NAMES)), dtype=np.float64)
        for col, cls_idx in enumerate(self.clf.classes_):
            out[:, int(cls_idx)] = proba[:, col]
        return out

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(features)
        return np.log(np.clip(proba, 1e-8, 1.0))

    def predict_logits_tensor(self, features: torch.Tensor) -> torch.Tensor:
        feats_np = features.detach().float().cpu().numpy()
        logits = self.predict_logits(feats_np)
        return torch.from_numpy(logits).to(device=features.device, dtype=torch.float32)

    def export_rules(self) -> str:
        return export_text(
            self.clf,
            feature_names=list(FEATURE_NAMES),
            class_names=list(CLASS_NAMES),
        )

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "classifier": self.clf,
            "feature_names": list(FEATURE_NAMES),
            "class_names": list(CLASS_NAMES),
            "metadata": metadata or {},
            "rules_text": self.export_rules(),
        }
        joblib.dump(payload, path)
        meta_path = path.replace(".joblib", ".meta.json")
        if meta_path == path:
            meta_path = path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_names": list(FEATURE_NAMES),
                    "class_names": list(CLASS_NAMES),
                    "metadata": metadata or {},
                    "tree_depth": int(self.clf.get_depth()),
                    "n_leaves": int(self.clf.get_n_leaves()),
                },
                f,
                indent=2,
            )
        rules_path = path.replace(".joblib", ".rules.txt")
        if rules_path == path:
            rules_path = path + ".rules.txt"
        with open(rules_path, "w", encoding="utf-8") as f:
            f.write(payload["rules_text"])

    @classmethod
    def load(cls, path: str) -> "SASVDecisionTreeFusion":
        payload = joblib.load(path)
        if isinstance(payload, DecisionTreeClassifier):
            return cls(payload)
        return cls(payload["classifier"])
