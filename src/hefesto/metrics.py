from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

TASKS: tuple[str, ...] = ("regression", "classification")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    func: Callable[..., float]
    greater_is_better: bool
    sklearn_scorer: str
    description: str
    task: str = "regression"
    needs_proba: bool = False

    @property
    def direction(self) -> str:
        return "maximize" if self.greater_is_better else "minimize"

    def is_better(self, candidate: float, reference: float) -> bool:
        if np.isnan(reference):
            return not np.isnan(candidate)
        return candidate > reference if self.greater_is_better else candidate < reference

    def score(self, y_true, y_pred, proba=None, classes=None) -> float:
        if self.task == "regression":
            return self.func(y_true, y_pred)
        return self.func(y_true, y_pred, proba, classes)


def _mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)


METRICS: dict[str, MetricSpec] = {
    "rmse": MetricSpec(
        "rmse",
        "RMSE",
        lambda y, p: float(root_mean_squared_error(y, p)),
        False,
        "neg_root_mean_squared_error",
        "Root mean squared error, in target units. Penalizes large errors.",
    ),
    "mae": MetricSpec(
        "mae",
        "MAE",
        lambda y, p: float(mean_absolute_error(y, p)),
        False,
        "neg_mean_absolute_error",
        "Mean absolute error, in target units. Robust to outliers.",
    ),
    "r2": MetricSpec(
        "r2",
        "R²",
        lambda y, p: float(r2_score(y, p)),
        True,
        "r2",
        "Coefficient of determination. 1.0 is a perfect fit.",
    ),
    "mape": MetricSpec(
        "mape",
        "MAPE (%)",
        _mape_percent,
        False,
        "neg_mean_absolute_percentage_error",
        "Mean absolute percentage error. Avoid when the target crosses zero.",
    ),
    "max_error": MetricSpec(
        "max_error",
        "Max error",
        lambda y, p: float(max_error(y, p)),
        False,
        "max_error",
        "Worst-case absolute error, in target units.",
    ),
}


def _roc_auc(y, pred, proba, classes) -> float:
    if proba is None:
        return float("nan")
    classes = list(classes)
    if len(classes) == 2:
        return float(roc_auc_score(np.asarray(y) == classes[1], proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes))


def _log_loss(y, pred, proba, classes) -> float:
    if proba is None:
        return float("nan")
    return float(log_loss(y, np.clip(proba, 1e-15, 1.0), labels=list(classes)))


def _on_labels(func, **kw) -> Callable[..., float]:
    return lambda y, pred, proba, classes: float(func(y, pred, **kw))


for _spec in (
    MetricSpec(
        "accuracy",
        "Accuracy",
        _on_labels(accuracy_score),
        True,
        "accuracy",
        "Share of correctly classified design points. Misleading on imbalanced classes.",
        "classification",
    ),
    MetricSpec(
        "balanced_accuracy",
        "Balanced accuracy",
        _on_labels(balanced_accuracy_score),
        True,
        "balanced_accuracy",
        "Mean recall over classes; robust to class imbalance.",
        "classification",
    ),
    MetricSpec(
        "f1_macro",
        "F1 (macro)",
        _on_labels(f1_score, average="macro", zero_division=0),
        True,
        "f1_macro",
        "Unweighted mean F1 over classes. Good default for imbalanced data.",
        "classification",
    ),
    MetricSpec(
        "precision_macro",
        "Precision (macro)",
        _on_labels(precision_score, average="macro", zero_division=0),
        True,
        "precision_macro",
        "Unweighted mean precision over classes (how many predicted are right).",
        "classification",
    ),
    MetricSpec(
        "recall_macro",
        "Recall (macro)",
        _on_labels(recall_score, average="macro", zero_division=0),
        True,
        "recall_macro",
        "Unweighted mean recall over classes (how many of each class are found).",
        "classification",
    ),
    MetricSpec(
        "roc_auc",
        "ROC AUC",
        _roc_auc,
        True,
        "roc_auc",
        "Ranking quality of the predicted probabilities (one-vs-rest macro for multiclass).",
        "classification",
        needs_proba=True,
    ),
    MetricSpec(
        "log_loss",
        "Log loss",
        _log_loss,
        False,
        "neg_log_loss",
        "Cross-entropy of the predicted probabilities; rewards calibrated confidence.",
        "classification",
        needs_proba=True,
    ),
):
    METRICS[_spec.key] = _spec

REPORT_METRICS: tuple[str, ...] = ("r2", "rmse", "mae", "mape", "max_error")
CLASSIFICATION_REPORT_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "roc_auc",
    "log_loss",
)
DEFAULT_METRIC: dict[str, str] = {"regression": "rmse", "classification": "f1_macro"}


def report_metrics(task: str = "regression") -> tuple[str, ...]:
    return CLASSIFICATION_REPORT_METRICS if task == "classification" else REPORT_METRICS


def metrics_for(task: str = "regression") -> dict[str, MetricSpec]:
    return {k: m for k, m in METRICS.items() if m.task == task}


def get_metric(key: str) -> MetricSpec:
    try:
        return METRICS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown metric '{key}'. Available: {sorted(METRICS)}") from exc


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    out: dict[str, float] = {}
    for key in REPORT_METRICS:
        try:
            out[key] = METRICS[key].func(y_true, y_pred)
        except ValueError:
            out[key] = float("nan")
    return out


def compute_classification_metrics(y_true, y_pred, proba=None, classes=None) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    classes = list(classes) if classes is not None else sorted(set(y_true) | set(y_pred))
    out: dict[str, float] = {}
    for key in CLASSIFICATION_REPORT_METRICS:
        try:
            out[key] = METRICS[key].score(y_true, y_pred, proba, classes)
        except ValueError:
            out[key] = float("nan")
    return out


def compute_task_metrics(task: str, y_true, y_pred, proba=None, classes=None) -> dict[str, float]:
    if task == "classification":
        return compute_classification_metrics(y_true, y_pred, proba, classes)
    return compute_metrics(y_true, y_pred)
