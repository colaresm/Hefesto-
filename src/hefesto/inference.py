from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
import pandas as pd

from hefesto.artifacts import CARD_FILENAME


@dataclass
class LoadedModel:
    model: Any
    metadata: dict[str, Any]

    @property
    def features(self) -> list[str]:
        feats = self.metadata.get("features")
        if feats:
            return list(feats)
        names = getattr(self.model, "feature_names_in_", None)
        if names is None:
            raise ValueError("The model does not record its input features.")
        return list(names)

    @property
    def target(self) -> str:
        return self.metadata.get("target", "prediction")

    @property
    def task(self) -> str:
        if "task" in self.metadata:
            return self.metadata["task"]
        return "classification" if hasattr(self.model, "predict_proba") else "regression"

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def name(self) -> str:
        return self.metadata.get("best_model", {}).get("name", type(self.model).__name__)


def load_model(source: str | Path | BinaryIO | bytes) -> LoadedModel:
    if isinstance(source, bytes):
        model = joblib.load(io.BytesIO(source))
        card_path = None
    elif isinstance(source, (str, Path)):
        model = joblib.load(source)
        card_path = Path(source).with_name(CARD_FILENAME)
    else:
        model = joblib.load(source)
        card_path = None
    metadata = getattr(model, "hefesto_metadata_", None) or {}
    if not metadata and card_path is not None and card_path.exists():
        metadata = json.loads(card_path.read_text(encoding="utf-8"))
    return LoadedModel(model=model, metadata=metadata)


def domain_report(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    domain = metadata.get("training_domain") or {}
    flags = pd.DataFrame(False, index=df.index, columns=list(domain))
    for feat, info in domain.items():
        if feat not in df:
            continue
        if info.get("type") == "numeric":
            values = pd.to_numeric(df[feat], errors="coerce")
            span = info["max"] - info["min"]
            tol = 1e-9 * (abs(span) or 1.0)
            flags[feat] = (values < info["min"] - tol) | (values > info["max"] + tol)
        else:
            flags[feat] = ~df[feat].astype(str).isin(info.get("categories", []))
    return flags


def predict(loaded: LoadedModel, df: pd.DataFrame) -> pd.DataFrame:
    missing = [f for f in loaded.features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing input column(s): {missing}")
    X = df[loaded.features].copy()
    for f, info in (loaded.metadata.get("training_domain") or {}).items():
        if info.get("type") == "numeric" and f in X:
            X[f] = pd.to_numeric(X[f], errors="coerce")
    out = df.copy()
    if loaded.is_classification:
        out[f"predicted_{loaded.target}"] = np.asarray(loaded.model.predict(X))
        if hasattr(loaded.model, "predict_proba"):
            proba = np.asarray(loaded.model.predict_proba(X), dtype=float)
            for j, cls in enumerate(loaded.model.classes_):
                out[f"proba_{cls}"] = proba[:, j]
    else:
        out[f"predicted_{loaded.target}"] = np.asarray(loaded.model.predict(X), dtype=float)
    flags = domain_report(X, loaded.metadata)
    if not flags.empty:
        out["out_of_domain"] = flags.any(axis=1)
        out["out_of_domain_features"] = flags.apply(
            lambda row: ", ".join(c for c, v in row.items() if v), axis=1
        )
    return out
