from __future__ import annotations

import io
import json
import platform
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hefesto.logging_utils import get_logger
from hefesto.metrics import get_metric, report_metrics
from hefesto.plotting import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_table,
    plot_surrogate_report,
    save_figure,
    save_pdf,
)
from hefesto.training import TrainingResult

logger = get_logger(__name__)

MODEL_FILENAME = "best_model.joblib"
CARD_FILENAME = "model_card.json"
FIGURE_STEM = "surrogate_report"


@dataclass
class RunArtifacts:
    run_dir: Path
    model_path: Path
    card_path: Path
    figure_png: Path
    figure_pdf: Path
    leaderboard_csv: Path
    importance_csv: Path | None
    predictions_csv: Path
    extra_models: dict[str, Path] = field(default_factory=dict)
    panel_pngs: dict[str, Path] = field(default_factory=dict)

    def zip_bytes(self) -> bytes:
        return zip_directory(self.run_dir)


def _slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len] or "target"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (Path, datetime)):
        return str(obj)
    return obj


def library_versions() -> dict[str, str]:
    import importlib.metadata as md

    versions = {"python": platform.python_version()}
    for pkg in (
        "hefesto",
        "scikit-learn",
        "numpy",
        "pandas",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
    ):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return versions


def build_model_card(
    result: TrainingResult,
    units: dict[str, str] | None = None,
    data_preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg, best = result.config, result.best
    metric = get_metric(cfg.metric)
    return _jsonable(
        {
            "hefesto_model_card_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": result.dataset_name,
            "task": cfg.task,
            "target": cfg.target,
            "classes": result.classes,
            "target_units": (units or {}).get(cfg.target),
            "features": cfg.features,
            "numeric_features": result.numeric_features,
            "categorical_features": result.categorical_features,
            "feature_units": {
                f: (units or {}).get(f) for f in cfg.features if (units or {}).get(f)
            },
            "training_domain": result.domain,
            "data_preprocessing": data_preprocessing,
            "n_train": len(result.y_train),
            "n_test": len(result.y_test) if result.has_test else 0,
            "best_model": {
                "key": best.key,
                "name": best.name,
                "family": best.family,
                "hyperparameters": best.best_params,
                "selection_metric": metric.label,
                "selection_basis": cfg.selection_basis,
                "cv_score_mean": best.cv_mean,
                "cv_score_std": best.cv_std,
                "cv_metrics": best.cv_metrics,
                "train_metrics": best.train_metrics,
                "test_metrics": best.test_metrics,
            },
            "log_target": cfg.log_target,
            "validation": {
                "cv_folds": cfg.cv_folds,
                "cv_repeats": cfg.cv_repeats,
                "test_size": cfg.test_size,
                "stratified_split": cfg.stratify_split,
                "n_trials": cfg.n_trials,
                "random_state": cfg.random_state,
            },
            "feature_importance": (
                result.importance.to_frame().to_dict(orient="records")
                if result.importance
                else None
            ),
            "library_versions": library_versions(),
            "usage": (
                "import joblib, pandas as pd; model = joblib.load('best_model.joblib'); "
                "model.predict(pd.DataFrame(...)[model.hefesto_metadata_['features']])"
            ),
        }
    )


def confusion_inputs(result: TrainingResult, key: str | None = None) -> dict[str, Any]:
    r = result.get(key) if key else result.best
    if result.has_test:
        return dict(
            y_true=result.y_test.to_numpy(),
            y_pred=r.test_pred,
            classes=result.classes,
            split="Hold-out",
        )
    return dict(
        y_true=result.y_train.to_numpy(),
        y_pred=r.oof_pred,
        classes=result.classes,
        split="CV out-of-fold",
    )


def make_report_figure(result: TrainingResult, key: str | None = None, units: dict | None = None):
    r = result.get(key) if key else result.best
    cfg = result.config
    target_units = (units or {}).get(cfg.target)
    caption = (
        f"Target: {cfg.target}  |  {len(result.y_train)} training / "
        f"{len(result.y_test) if result.has_test else 0} hold-out design points  |  "
        f"{cfg.cv_folds}-fold CV"
        + (f" x {cfg.cv_repeats}" if cfg.cv_repeats > 1 else "")
        + f"  |  {cfg.n_trials} Optuna trials per model"
    )
    importance = result.importance if r.key == result.best_key else None
    return plot_surrogate_report(
        model_name=r.name,
        target=cfg.target,
        importance=importance,
        y_cv=result.y_train.to_numpy(),
        pred_cv=r.oof_pred,
        y_test=result.y_test.to_numpy() if result.has_test else None,
        pred_test=r.test_pred,
        top_n=cfg.top_n_features,
        units=target_units,
        subtitle=caption,
        confusion=confusion_inputs(result, r.key) if result.is_classification else None,
    )


def make_panel_figures(result: TrainingResult, units: dict | None = None) -> dict[str, Any]:
    cfg, best = result.config, result.best
    importance = plot_feature_importance(result.importance, cfg.top_n_features)
    if result.is_classification:
        return {
            "feature_importance": importance,
            "confusion_matrix": plot_confusion_matrix(**confusion_inputs(result)),
        }
    return {
        "feature_importance": importance,
        "actual_vs_predicted": plot_actual_vs_predicted(
            target=cfg.target,
            y_cv=result.y_train.to_numpy(),
            pred_cv=best.oof_pred,
            y_test=result.y_test.to_numpy() if result.has_test else None,
            pred_test=best.test_pred,
            units=(units or {}).get(cfg.target),
        ),
    }


def metrics_table(result: TrainingResult) -> tuple[pd.DataFrame, int | None]:
    board = result.leaderboard()
    metric = get_metric(result.config.metric)
    mean_col, std_col = f"CV {metric.label} (mean)", f"CV {metric.label} (std)"

    def fmt(v) -> str:
        return "-" if v is None or not np.isfinite(v) else f"{v:.4g}"

    out = pd.DataFrame({"Rank": board["Rank"].astype(str), "Model": board["Model"]})
    out[f"CV {metric.label} (mean ± std)"] = [
        "-" if not np.isfinite(m) else f"{m:.4g} ± {s:.2g}"
        for m, s in zip(board[mean_col], board[std_col], strict=True)
    ]
    prefixes = ["CV", "Test"] if result.has_test else ["CV"]
    for prefix in prefixes:
        for m in report_metrics(result.task):
            col = f"{prefix} {get_metric(m).label}"
            out[col.replace("Test ", "Hold-out ")] = board[col].map(fmt)
    failed = board["Status"] != "ok"
    out.loc[failed, out.columns[2:]] = "failed"
    best_rows = board.index[board["key"] == result.best_key].tolist()
    return out, (best_rows[0] if best_rows else None)


def make_metrics_figure(result: TrainingResult):
    cfg = result.config
    table, best_row = metrics_table(result)
    metric = get_metric(cfg.metric)
    note = (
        f"Ranked by {'mean CV' if cfg.selection_basis == 'cv' else 'hold-out'} {metric.label} "
        f"({'higher' if metric.greater_is_better else 'lower'} is better). CV metrics are computed "
        "on pooled out-of-fold predictions"
        + (
            f"; hold-out on {len(result.y_test)} untouched design points."
            if result.has_test
            else "."
        )
    )
    return plot_metrics_table(table, title="Model Metrics", highlight_row=best_row, note=note)


def save_run(
    result: TrainingResult,
    output_dir: str | Path | None = None,
    units: dict[str, str] | None = None,
    run_name: str | None = None,
    data_preprocessing: dict[str, Any] | None = None,
) -> RunArtifacts:
    cfg, best = result.config, result.best
    root = Path(output_dir or cfg.output_dir)
    name = run_name or f"{result.started_at:%Y%m%d-%H%M%S}_{_slug(cfg.target)}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    card = build_model_card(result, units, data_preprocessing)
    best.estimator.hefesto_metadata_ = card
    model_path = run_dir / MODEL_FILENAME
    joblib.dump(best.estimator, model_path, compress=3)
    card_path = run_dir / CARD_FILENAME
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(_jsonable(cfg.to_dict()), indent=2), encoding="utf-8"
    )

    fig = make_report_figure(result, units=units)
    png = Path(save_figure(fig, run_dir / FIGURE_STEM, ("png",))[0])
    table_fig = make_metrics_figure(result)
    pdf = Path(save_pdf([fig, table_fig], run_dir / f"{FIGURE_STEM}.pdf"))
    plt.close(fig)
    plt.close(table_fig)

    panel_pngs: dict[str, Path] = {}
    for panel_name, panel_fig in make_panel_figures(result, units).items():
        panel_pngs[panel_name] = Path(save_figure(panel_fig, run_dir / panel_name, ("png",))[0])
        plt.close(panel_fig)

    leaderboard_csv = run_dir / "leaderboard.csv"
    result.leaderboard().to_csv(leaderboard_csv, index=False)

    importance_csv = None
    if result.importance is not None:
        importance_csv = run_dir / "feature_importance.csv"
        result.importance.to_frame().to_csv(importance_csv, index=False)

    preds = [
        pd.DataFrame(
            {"split": "cv_out_of_fold", "actual": result.y_train, "predicted": best.oof_pred}
        )
    ]
    preds[0] = pd.concat([result.X_train, preds[0]], axis=1)
    if result.has_test:
        preds.append(
            pd.concat(
                [
                    result.X_test,
                    pd.DataFrame(
                        {
                            "split": "hold_out_test",
                            "actual": result.y_test,
                            "predicted": best.test_pred,
                        }
                    ),
                ],
                axis=1,
            )
        )
    if result.is_classification:
        probas = [best.oof_proba] + ([best.test_proba] if result.has_test else [])
        for frame, proba in zip(preds, probas, strict=True):
            frame["correct"] = frame["predicted"] == frame["actual"]
            if proba is not None:
                for j, cls in enumerate(result.classes):
                    frame[f"proba_{cls}"] = proba[:, j]
    predictions = pd.concat(preds, ignore_index=True)
    if not result.is_classification:
        predictions["residual"] = predictions["predicted"] - predictions["actual"]
    predictions_csv = run_dir / "predictions.csv"
    predictions.to_csv(predictions_csv, index=False)

    trials_dir = run_dir / "optuna_trials"
    trials_dir.mkdir(exist_ok=True)
    for r in result.results:
        if not r.trials.empty:
            r.trials.to_csv(trials_dir / f"{r.key}.csv", index=False)

    extra: dict[str, Path] = {}
    if cfg.save_all_models:
        models_dir = run_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for r in result.results:
            if r.ok:
                r.estimator.hefesto_metadata_ = {
                    **card,
                    "best_model": {
                        "key": r.key,
                        "name": r.name,
                        "hyperparameters": _jsonable(r.best_params),
                        "cv_metrics": _jsonable(r.cv_metrics),
                        "test_metrics": _jsonable(r.test_metrics),
                    },
                }
                extra[r.key] = models_dir / f"{r.key}.joblib"
                joblib.dump(r.estimator, extra[r.key], compress=3)
        best.estimator.hefesto_metadata_ = card

    logger.info("Run saved to %s", run_dir)
    return RunArtifacts(
        run_dir,
        model_path,
        card_path,
        png,
        pdf,
        leaderboard_csv,
        importance_csv,
        predictions_csv,
        extra,
        panel_pngs,
    )


def zip_directory(directory: str | Path) -> bytes:
    directory = Path(directory)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zf.write(path, Path(directory.name) / path.relative_to(directory))
    return buffer.getvalue()


def list_runs(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if (p / MODEL_FILENAME).exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
