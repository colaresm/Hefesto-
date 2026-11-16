from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from hefesto.logging_utils import get_logger
from hefesto.metrics import get_metric
from hefesto.models import inner_pipeline

logger = get_logger(__name__)


@dataclass
class FeatureImportance:
    features: list[str]
    importances: np.ndarray
    std: np.ndarray | None
    method: str
    axis_label: str
    note: str = ""

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"feature": self.features, "importance": self.importances})
        if self.std is not None:
            df["std"] = self.std
        total = np.abs(self.importances).sum()
        df["relative_importance_pct"] = 100 * np.abs(self.importances) / total if total > 0 else 0.0
        return df.sort_values("importance", ascending=False, ignore_index=True)

    def top(self, n: int) -> pd.DataFrame:
        return self.to_frame().head(n)


def _permutation(
    model,
    X,
    y,
    metric_key: str,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
    eval_set: str,
    n_classes: int | None = None,
) -> FeatureImportance:
    metric = get_metric(metric_key)
    scorer = metric.sklearn_scorer
    if scorer == "roc_auc" and (n_classes or 2) > 2:
        scorer = "roc_auc_ovr"
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    mean, std = result.importances_mean, result.importances_std
    if metric_key == "mape":
        mean, std = mean * 100, std * 100
    verb = "decrease" if metric.greater_is_better else "increase"
    return FeatureImportance(
        features=list(X.columns),
        importances=mean,
        std=std,
        method="permutation",
        axis_label=f"Mean {verb} in {metric.label} when permuted ({eval_set})",
    )


def _native(model, numeric: list[str], categorical: list[str]):
    pipe = inner_pipeline(model)
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        raw = np.asarray(est.feature_importances_, dtype=float)
        label = "Native importance (impurity / gain, normalized)"
    elif hasattr(est, "coef_"):
        coef = np.atleast_2d(np.asarray(est.coef_, dtype=float))
        raw = np.abs(coef).sum(axis=0)
        label = "|Standardized coefficient|"
    else:
        return None

    widths = [1] * len(numeric)
    if categorical:
        encoder = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        widths += [len(c) for c in encoder.categories_]
    if sum(widths) != raw.size:
        return None
    agg, start = [], 0
    for w in widths:
        agg.append(raw[start : start + w].sum())
        start += w
    agg = np.asarray(agg)
    if agg.sum() > 0 and hasattr(est, "feature_importances_"):
        agg = agg / agg.sum()
    return FeatureImportance(
        features=[*numeric, *categorical],
        importances=agg,
        std=None,
        method="native",
        axis_label=label,
    )


def compute_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    numeric: list[str],
    categorical: list[str],
    method: str = "permutation",
    metric: str = "rmse",
    n_repeats: int = 15,
    random_state: int = 42,
    n_jobs: int = 1,
    eval_set: str = "hold-out set",
    n_classes: int | None = None,
) -> FeatureImportance:
    if method == "native":
        native = _native(model, numeric, categorical)
        if native is not None:
            return native
        logger.info("Estimator has no native importance; falling back to permutation importance.")
        fi = _permutation(model, X, y, metric, n_repeats, random_state, n_jobs, eval_set, n_classes)
        fi.note = "Estimator exposes no native importance; permutation importance shown instead."
        return fi
    return _permutation(model, X, y, metric, n_repeats, random_state, n_jobs, eval_set, n_classes)
