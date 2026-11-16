from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from hefesto.config import TrainingConfig
from hefesto.data import (
    class_labels,
    feature_domain,
    split_feature_types,
    validate_training_frame,
)
from hefesto.importance import FeatureImportance, compute_feature_importance
from hefesto.logging_utils import get_logger
from hefesto.metrics import compute_task_metrics, get_metric, report_metrics
from hefesto.models import BuildContext, ModelSpec, build_pipeline, get_model_spec

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ProgressEvent:
    stage: str
    model_key: str = ""
    model_name: str = ""
    model_index: int = 0
    n_models: int = 0
    trial: int = 0
    n_trials: int = 0
    value: float | None = None
    best_value: float | None = None
    message: str = ""

    @property
    def fraction(self) -> float:
        if self.stage == "done" or self.n_models == 0:
            return 1.0
        within = self.trial / self.n_trials if self.n_trials else 0.0
        if self.stage == "model_end":
            within = 1.0
        return min(1.0, (self.model_index + within) / self.n_models)


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class ModelResult:
    key: str
    name: str
    family: str
    status: str = "ok"
    error: str | None = None
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    cv_mean: float = float("nan")
    cv_std: float = float("nan")
    cv_metrics: dict[str, float] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    n_trials_complete: int = 0
    n_trials_pruned: int = 0
    n_trials_failed: int = 0
    tuning_time_s: float = 0.0
    estimator: Any = None
    oof_pred: np.ndarray | None = None
    test_pred: np.ndarray | None = None
    oof_proba: np.ndarray | None = None
    test_proba: np.ndarray | None = None
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def selection_score(self, metric: str, basis: str) -> float:
        if basis == "test":
            return self.test_metrics.get(metric, float("nan"))
        return self.cv_mean


@dataclass
class TrainingResult:
    config: TrainingConfig
    results: list[ModelResult]
    best_key: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame | None
    y_test: pd.Series | None
    numeric_features: list[str]
    categorical_features: list[str]
    importance: FeatureImportance | None
    domain: dict[str, dict]
    dataset_name: str
    started_at: datetime
    duration_s: float
    n_rows_dropped: int = 0
    classes: list | None = None

    @property
    def task(self) -> str:
        return self.config.task

    @property
    def is_classification(self) -> bool:
        return self.config.task == "classification"

    @property
    def best(self) -> ModelResult:
        return self.get(self.best_key)

    def get(self, key: str) -> ModelResult:
        for r in self.results:
            if r.key == key:
                return r
        raise KeyError(key)

    @property
    def has_test(self) -> bool:
        return self.X_test is not None and len(self.X_test) > 0

    def leaderboard(self) -> pd.DataFrame:
        metric = get_metric(self.config.metric)
        rows = []
        for r in self.results:
            row: dict[str, Any] = {
                "Model": r.name,
                "key": r.key,
                "Family": r.family,
                f"CV {metric.label} (mean)": r.cv_mean,
                f"CV {metric.label} (std)": r.cv_std,
            }
            for m in report_metrics(self.task):
                row[f"CV {get_metric(m).label}"] = r.cv_metrics.get(m, np.nan)
            if self.has_test:
                for m in report_metrics(self.task):
                    row[f"Test {get_metric(m).label}"] = r.test_metrics.get(m, np.nan)
            row.update(
                {
                    "Trials (complete)": r.n_trials_complete,
                    "Trials (pruned)": r.n_trials_pruned,
                    "Tuning time (s)": round(r.tuning_time_s, 2),
                    "Status": r.status if r.ok else f"failed: {r.error}",
                }
            )
            rows.append(row)
        df = pd.DataFrame(rows)
        score = [
            r.selection_score(self.config.metric, self.config.selection_basis) for r in self.results
        ]
        df["_score"] = score
        df = df.sort_values("_score", ascending=not metric.greater_is_better, na_position="last")
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df.drop(columns="_score").reset_index(drop=True)


class SurrogateTrainer:
    def __init__(self, config: TrainingConfig, progress_callback: ProgressCallback | None = None):
        self.config = config
        self.metric = get_metric(config.metric)
        self._callback = progress_callback
        self._specs: list[ModelSpec] = [get_model_spec(k, config.task) for k in config.models]
        self._clf = config.task == "classification"
        self._classes: np.ndarray | None = None

    def _emit(self, **kwargs) -> None:
        if self._callback is not None:
            self._callback(ProgressEvent(**kwargs))

    def _split(self, X: pd.DataFrame, y: pd.Series):
        cfg = self.config
        if cfg.test_size == 0:
            return X, None, y, None
        stratify = None
        if self._clf:
            if cfg.stratify_split and y.value_counts().min() >= 2:
                stratify = y
        elif cfg.stratify_split:
            n_test = max(1, int(round(cfg.test_size * len(y))))
            n_bins = int(min(10, n_test, len(y) // 4))
            if n_bins >= 2:
                bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
                if bins.value_counts().min() >= 2:
                    stratify = bins
        try:
            return train_test_split(
                X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify
            )
        except ValueError:
            logger.warning("Stratified split not possible; using a random split.")
            return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)

    def _cv_splits(self, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        cfg = self.config
        n = len(y)
        folds = min(cfg.cv_folds, n)
        if self._clf:
            folds = max(2, min(folds, int(y.value_counts().min())))
            if folds < cfg.cv_folds:
                logger.warning(
                    "Smallest class has %d rows: using %d stratified folds instead of %d.",
                    folds,
                    folds,
                    cfg.cv_folds,
                )
            if cfg.cv_repeats > 1:
                cv = RepeatedStratifiedKFold(
                    n_splits=folds, n_repeats=cfg.cv_repeats, random_state=cfg.random_state
                )
            else:
                cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.random_state)
            return list(cv.split(np.zeros(n), y))
        if cfg.cv_repeats > 1:
            cv = RepeatedKFold(
                n_splits=folds, n_repeats=cfg.cv_repeats, random_state=cfg.random_state
            )
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=cfg.random_state)
        return list(cv.split(np.zeros(n)))

    def _build(self, spec: ModelSpec, params: dict, ctx: BuildContext):
        return build_pipeline(
            spec, params, self._numeric, self._categorical, ctx, self.config.log_target
        )

    def _predict(self, model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        pred = np.asarray(model.predict(X))
        if not self._clf or not hasattr(model, "predict_proba"):
            return pred, None
        raw = np.asarray(model.predict_proba(X), dtype=float)
        if len(model.classes_) == len(self._classes):
            return pred, raw
        proba = np.zeros((len(X), len(self._classes)))
        cols = np.searchsorted(self._classes, model.classes_)
        proba[:, cols] = raw
        return pred, proba

    def _metrics(self, y, pred, proba=None) -> dict[str, float]:
        return compute_task_metrics(self.config.task, y, pred, proba, self._classes)

    def _tune(self, spec: ModelSpec, idx: int, X: pd.DataFrame, y: pd.Series, splits, ctx):
        cfg, metric = self.config, self.metric
        n_trials = cfg.n_trials if spec.tunable else 1
        sampler = optuna.samplers.TPESampler(
            seed=cfg.random_state, n_startup_trials=max(3, min(10, n_trials // 3))
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study = optuna.create_study(
            direction=metric.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"hefesto-{spec.key}",
        )

        def objective(trial: optuna.Trial) -> float:
            params = spec.sample_params(trial, ctx)
            trial.set_user_attr("model_params", params)
            scores: list[float] = []
            for step, (tr, va) in enumerate(splits):
                model = self._build(spec, params, ctx)
                model.fit(X.iloc[tr], y.iloc[tr])
                pred, proba = self._predict(model, X.iloc[va])
                if not self._clf and not np.all(np.isfinite(pred)):
                    raise ValueError("Model produced non-finite predictions.")
                scores.append(metric.score(y.iloc[va].to_numpy(), pred, proba, self._classes))
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("cv_std", float(np.std(scores)))
            return float(np.mean(scores))

        def on_trial_end(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
            try:
                best = st.best_value
            except ValueError:
                best = None
            self._emit(
                stage="trial",
                model_key=spec.key,
                model_name=spec.name,
                model_index=idx,
                n_models=len(self._specs),
                trial=tr.number + 1,
                n_trials=n_trials,
                value=tr.value,
                best_value=best,
            )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=cfg.timeout_per_model,
            callbacks=[on_trial_end],
            catch=(Exception,),
            gc_after_trial=False,
        )
        return study

    def _evaluate(self, spec: ModelSpec, params: dict, X, y, splits, ctx):
        n_first = self._n_folds
        if self._clf:
            oof = np.empty(len(y), dtype=y.dtype)
            oof_proba = np.full((len(y), len(self._classes)), np.nan)
        else:
            oof, oof_proba = np.full(len(y), np.nan), None
        fold_scores: list[float] = []
        for i, (tr, va) in enumerate(splits):
            model = self._build(spec, params, ctx)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred, proba = self._predict(model, X.iloc[va])
            fold_scores.append(self.metric.score(y.iloc[va].to_numpy(), pred, proba, self._classes))
            if i < n_first:
                oof[va] = pred
                if oof_proba is not None and proba is not None:
                    oof_proba[va] = proba
        if oof_proba is not None and np.isnan(oof_proba).all():
            oof_proba = None
        return np.asarray(fold_scores), oof, oof_proba

    def _run_model(
        self, spec: ModelSpec, idx: int, X, y, X_test, y_test, splits, ctx
    ) -> ModelResult:
        result = ModelResult(key=spec.key, name=spec.name, family=spec.family)
        self._emit(
            stage="model_start",
            model_key=spec.key,
            model_name=spec.name,
            model_index=idx,
            n_models=len(self._specs),
            n_trials=self.config.n_trials if spec.tunable else 1,
        )
        if spec.max_recommended_samples and len(y) > spec.max_recommended_samples:
            logger.warning("%s may be slow with %d samples.", spec.name, len(y))
        t0 = time.perf_counter()
        try:
            study = self._tune(spec, idx, X, y, splits, ctx)
            states = [t.state for t in study.trials]
            result.n_trials_complete = states.count(optuna.trial.TrialState.COMPLETE)
            result.n_trials_pruned = states.count(optuna.trial.TrialState.PRUNED)
            result.n_trials_failed = states.count(optuna.trial.TrialState.FAIL)
            result.trials = study.trials_dataframe(
                attrs=("number", "value", "state", "duration", "params")
            )
            if result.n_trials_complete == 0:
                failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                reason = (
                    failed[-1].system_attrs.get("fail_reason", "unknown") if failed else "timeout"
                )
                raise RuntimeError(f"No trial completed ({reason}).")
            params = study.best_trial.user_attrs["model_params"]
            result.best_params = params
            fold_scores, oof, oof_proba = self._evaluate(spec, params, X, y, splits, ctx)
            result.cv_scores, result.oof_pred, result.oof_proba = fold_scores, oof, oof_proba
            result.cv_mean, result.cv_std = float(fold_scores.mean()), float(fold_scores.std())
            if self._clf:
                result.cv_metrics = self._metrics(y.to_numpy(), oof, oof_proba)
            else:
                mask = ~np.isnan(oof)
                result.cv_metrics = self._metrics(y.to_numpy()[mask], oof[mask])

            final = self._build(spec, params, ctx)
            final.fit(X, y)
            result.estimator = final
            result.train_metrics = self._metrics(y, *self._predict(final, X))
            if X_test is not None and len(X_test):
                result.test_pred, result.test_proba = self._predict(final, X_test)
                result.test_metrics = self._metrics(y_test, result.test_pred, result.test_proba)
        except Exception as exc:
            logger.exception("Model %s failed", spec.name)
            result.status, result.error = "failed", str(exc).splitlines()[0][:200]
        result.tuning_time_s = time.perf_counter() - t0
        self._emit(
            stage="model_end",
            model_key=spec.key,
            model_name=spec.name,
            model_index=idx,
            n_models=len(self._specs),
            best_value=result.cv_mean,
            message=result.error or "",
        )
        logger.info(
            "%-28s CV %s = %.5g ± %.3g (%.1fs)",
            spec.name,
            self.metric.label,
            result.cv_mean,
            result.cv_std,
            result.tuning_time_s,
        )
        return result

    def fit(self, df: pd.DataFrame, dataset_name: str = "dataset") -> TrainingResult:
        cfg = self.config
        started, t0 = datetime.now(), time.perf_counter()
        validate_training_frame(df, cfg.target, cfg.features, cfg.cv_folds, cfg.task)

        data = df[[*cfg.features, cfg.target]].copy()
        n_before = len(data)
        data = data[data[cfg.target].notna()].reset_index(drop=True)
        n_dropped = n_before - len(data)
        if cfg.log_target and (data[cfg.target] <= 0).any():
            raise ValueError(
                f"Log transform requires a strictly positive target; '{cfg.target}' has values <= 0."
            )

        X = data[cfg.features]
        if self._clf:
            y = class_labels(data[cfg.target])
            self._classes = np.sort(y.unique())
            binary_only = [s.name for s in self._specs if s.binary_only]
            if binary_only and len(self._classes) > 2:
                raise ValueError(
                    f"{', '.join(binary_only)} supports binary targets only; "
                    f"'{cfg.target}' has {len(self._classes)} classes."
                )
        else:
            y = data[cfg.target].astype(float)
        self._numeric, self._categorical = split_feature_types(X, cfg.features)
        X_train, X_test, y_train, y_test = self._split(X, y)
        X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
        if X_test is not None:
            X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

        splits = self._cv_splits(y_train)
        self._n_folds = len(splits) // cfg.cv_repeats
        n_jobs = cfg.n_jobs
        ctx = BuildContext(
            random_state=cfg.random_state,
            n_jobs=n_jobs,
            n_samples=int(len(y_train) * (1 - 1 / cfg.cv_folds)),
            n_features=len(cfg.features),
        )
        logger.info(
            "Training on %d rows (%d hold-out), %d features, %d models, %d CV splits",
            len(y_train),
            0 if y_test is None else len(y_test),
            len(cfg.features),
            len(self._specs),
            len(splits),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results = [
                self._run_model(spec, i, X_train, y_train, X_test, y_test, splits, ctx)
                for i, spec in enumerate(self._specs)
            ]

        ok = [r for r in results if r.ok]
        if not ok:
            errors = "; ".join(f"{r.name}: {r.error}" for r in results)
            raise RuntimeError(f"Every model failed to train. {errors}")
        best = ok[0]
        for r in ok[1:]:
            if self.metric.is_better(
                r.selection_score(cfg.metric, cfg.selection_basis),
                best.selection_score(cfg.metric, cfg.selection_basis),
            ):
                best = r

        self._emit(
            stage="importance",
            model_key=best.key,
            model_name=best.name,
            n_models=len(self._specs),
            model_index=len(self._specs) - 1,
            message="Computing feature importance",
        )
        has_test = X_test is not None and len(X_test) > 0
        X_eval, y_eval = (X_test, y_test) if has_test else (X_train, y_train)
        importance = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                importance = compute_feature_importance(
                    best.estimator,
                    X_eval,
                    y_eval,
                    self._numeric,
                    self._categorical,
                    method=cfg.importance_method,
                    metric=cfg.metric,
                    n_repeats=cfg.permutation_repeats,
                    random_state=cfg.random_state,
                    n_jobs=1,
                    eval_set="hold-out set" if has_test else "training set",
                    n_classes=len(self._classes) if self._clf else None,
                )
        except Exception:
            logger.exception("Feature importance failed")

        duration = time.perf_counter() - t0
        self._emit(stage="done", n_models=len(self._specs), message=f"Finished in {duration:.1f}s")
        return TrainingResult(
            config=cfg,
            results=results,
            best_key=best.key,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            numeric_features=self._numeric,
            categorical_features=self._categorical,
            importance=importance,
            domain=feature_domain(X_train, cfg.features),
            dataset_name=dataset_name,
            started_at=started,
            duration_s=duration,
            n_rows_dropped=n_dropped,
            classes=self._classes.tolist() if self._clf else None,
        )
