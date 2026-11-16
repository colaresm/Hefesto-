import json
import re

import numpy as np
import optuna
import pandas as pd
import pytest

from hefesto import SurrogateTrainer, TrainingConfig
from hefesto.artifacts import metrics_table, save_run
from hefesto.cli import main
from hefesto.data import suggest_task
from hefesto.inference import load_model, predict
from hefesto.models import BuildContext, available_models, build_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def clf_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 150
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(-1, 1, n)
    material = rng.choice(["SAC305", "SnPb"], n)
    score = 3 * x1 + 2 * x2 + np.where(material == "SAC305", 0.5, 0.0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "material": material})
    df["status"] = np.where(score > 1.8, "fail", "pass")
    df["grade"] = np.digitize(score, [0.8, 2.4]).astype(float) + 1
    return df


def _config(tmp_path, target, models, **kw):
    base = dict(
        target=target,
        features=["x1", "x2", "material"],
        models=models,
        task="classification",
        n_trials=2,
        cv_folds=3,
        test_size=0.25,
        output_dir=tmp_path,
        n_jobs=1,
    )
    base.update(kw)
    return TrainingConfig(**base)


@pytest.mark.parametrize("key", sorted(available_models("classification")))
def test_every_classifier_samples_builds_fits_and_predicts(key, clf_df):
    spec = available_models("classification")[key]
    ctx = BuildContext(random_state=0, n_jobs=1, n_samples=len(clf_df), n_features=3)
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1)).ask()
    model = build_pipeline(spec, spec.sample_params(trial, ctx), ["x1", "x2"], ["material"], ctx)
    X, y = clf_df[["x1", "x2", "material"]], clf_df["status"]
    model.fit(X, y)
    assert set(model.predict(X)) <= {"fail", "pass"}
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(axis=1), 1)
    assert list(model.classes_) == ["fail", "pass"]


def test_config_metric_defaults_and_validation(tmp_path):
    assert _config(tmp_path, "status", ["random_forest"]).metric == "f1_macro"
    with pytest.raises(ValueError, match="not a classification metric"):
        _config(tmp_path, "status", ["random_forest"], metric="rmse")
    with pytest.raises(ValueError, match="regression only"):
        _config(tmp_path, "status", ["random_forest"], log_target=True)


def test_suggest_task(clf_df):
    assert suggest_task(clf_df["status"]) == "classification"
    assert suggest_task(clf_df["grade"]) == "classification"
    assert suggest_task(clf_df["x1"]) == "regression"


def test_binary_end_to_end_with_artifacts_and_inference(tmp_path, clf_df):
    cfg = _config(tmp_path, "status", ["logistic", "random_forest"], metric="roc_auc")
    result = SurrogateTrainer(cfg).fit(clf_df, "clf")
    assert result.is_classification and result.classes == ["fail", "pass"]
    best = result.best
    assert best.cv_metrics["accuracy"] > 0.8 and best.test_metrics["roc_auc"] > 0.8
    assert best.oof_proba.shape == (len(result.y_train), 2)
    assert "CV ROC AUC" in result.leaderboard().columns
    assert result.importance is not None

    art = save_run(result)
    assert set(art.panel_pngs) == {"feature_importance", "confusion_matrix"}
    assert all(p.exists() for p in art.panel_pngs.values())
    assert len(re.findall(rb"/Type\s*/Page(?!s)", art.figure_pdf.read_bytes())) == 2
    table, best_row = metrics_table(result)
    assert table.loc[best_row, "Model"] == best.name and "Hold-out F1 (macro)" in table
    preds = pd.read_csv(art.predictions_csv)
    assert {"actual", "predicted", "correct", "proba_fail", "proba_pass"} <= set(preds.columns)
    card = json.loads(art.card_path.read_text())
    assert card["task"] == "classification" and card["classes"] == ["fail", "pass"]

    loaded = load_model(art.model_path)
    assert loaded.is_classification
    out = predict(loaded, clf_df.head(5))
    assert set(out["predicted_status"]) <= {"fail", "pass"}
    assert np.allclose(out[["proba_fail", "proba_pass"]].sum(axis=1), 1)


def test_multiclass_integer_labels(tmp_path, clf_df):
    cfg = _config(tmp_path, "grade", ["random_forest", "xgboost"], metric="log_loss")
    result = SurrogateTrainer(cfg).fit(clf_df)
    assert result.classes == [1, 2, 3]
    assert result.best.test_proba.shape[1] == 3
    assert np.isfinite(result.best.cv_metrics["roc_auc"])

    with pytest.raises(ValueError, match="binary targets only"):
        SurrogateTrainer(_config(tmp_path, "grade", ["logistic"])).fit(clf_df)


def test_rare_class_is_rejected(tmp_path, clf_df):
    df = clf_df.copy()
    df.loc[0, "status"] = "unique"
    with pytest.raises(ValueError, match="at least 2 rows"):
        SurrogateTrainer(_config(tmp_path, "status", ["random_forest"])).fit(df)


def test_cli_classification_auto_task_and_outliers(tmp_path, clf_df, capsys):
    data = tmp_path / "clf.csv"
    clf_df.drop(columns="grade").to_csv(data, index=False)
    code = main(
        [
            "train",
            "--data",
            str(data),
            "--target",
            "status",
            "--features",
            "x1",
            "x2",
            "material",
            "--models",
            "logistic",
            "random_forest",
            "--trials",
            "2",
            "--folds",
            "3",
            "--remove-outliers",
            "iqr",
            "--output",
            str(tmp_path / "out"),
            "--n-jobs",
            "1",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "CV Accuracy" in out and "Best model" in out
    card = json.loads(next((tmp_path / "out").glob("*/model_card.json")).read_text())
    assert card["task"] == "classification"
    assert "status" not in card["data_preprocessing"]["outliers_removed"]["columns"]
