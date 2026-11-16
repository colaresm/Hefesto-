import json
import re

import joblib
import numpy as np
import pytest

from hefesto import SurrogateTrainer, TrainingConfig
from hefesto.artifacts import list_runs, metrics_table, save_run, zip_directory
from hefesto.data import load_dataset
from hefesto.inference import load_model, predict


def _config(tmp_path, **kw):
    base = dict(
        target="y",
        features=["x1", "x2", "material"],
        models=["ridge", "random_forest"],
        n_trials=3,
        cv_folds=3,
        test_size=0.25,
        output_dir=tmp_path,
        n_jobs=1,
    )
    base.update(kw)
    return TrainingConfig(**base)


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="cannot also be an input"):
        _config(tmp_path, features=["y", "x1"])
    with pytest.raises(ValueError, match="requires test_size"):
        _config(tmp_path, test_size=0, selection_basis="test")


def test_end_to_end_training_and_artifacts(tmp_path, mixed_df):
    events = []
    cfg = _config(tmp_path, importance_method="permutation", save_all_models=True)
    result = SurrogateTrainer(cfg, progress_callback=events.append).fit(mixed_df, "mixed")

    assert {r.key for r in result.results} == {"ridge", "random_forest"}
    assert all(r.ok for r in result.results)
    best = result.best
    others = [r for r in result.results if r.key != best.key]
    assert all(best.cv_mean <= r.cv_mean for r in others)
    assert best.oof_pred.shape == (len(result.y_train),) and not np.isnan(best.oof_pred).any()
    assert set(best.test_metrics) >= {"r2", "rmse", "mae"}
    assert result.importance is not None and set(result.importance.features) == set(cfg.features)
    assert events[-1].stage == "done" and any(e.stage == "trial" for e in events)

    board = result.leaderboard()
    assert board.iloc[0]["key"] == best.key and board["Rank"].tolist() == [1, 2]

    art = save_run(result, units={"y": "MPa"}, data_preprocessing={"outliers_removed": None})
    assert set(art.panel_pngs) == {"feature_importance", "actual_vs_predicted"}
    for path in (
        *art.panel_pngs.values(),
        art.model_path,
        art.card_path,
        art.figure_png,
        art.figure_pdf,
        art.leaderboard_csv,
        art.predictions_csv,
        art.importance_csv,
    ):
        assert path.exists() and path.stat().st_size > 0
    assert set(art.extra_models) == {"ridge", "random_forest"}
    pdf_pages = re.findall(rb"/Type\s*/Page(?!s)", art.figure_pdf.read_bytes())
    assert len(pdf_pages) == 2
    table, best_row = metrics_table(result)
    assert table.loc[best_row, "Model"] == best.name
    assert {"CV R²", "Hold-out RMSE"} <= set(table.columns)
    card = json.loads(art.card_path.read_text())
    assert card["data_preprocessing"] == {"outliers_removed": None}
    assert card["best_model"]["key"] == best.key and card["target_units"] == "MPa"
    assert list_runs(tmp_path) == [art.run_dir]
    assert len(zip_directory(art.run_dir)) > 0

    model = joblib.load(art.model_path)
    assert model.hefesto_metadata_["features"] == cfg.features
    np.testing.assert_allclose(model.predict(result.X_test), best.test_pred)

    loaded = load_model(art.model_path)
    new = mixed_df.head(3).copy()
    new.loc[0, "x1"] = 10.0
    out = predict(loaded, new)
    assert out["out_of_domain"].tolist() == [True, False, False]
    assert "x1" in out.loc[0, "out_of_domain_features"]


def test_cv_only_and_r2_metric(tmp_path, mixed_df):
    cfg = _config(tmp_path, test_size=0, metric="r2", models=["linear", "polynomial"], cv_repeats=2)
    result = SurrogateTrainer(cfg).fit(mixed_df)
    assert not result.has_test
    assert result.best.key == "polynomial"
    assert len(result.best.cv_scores) == 6
    assert result.get("linear").n_trials_complete == 1
    assert "training set" in result.importance.axis_label


def test_log_target_requires_positive(tmp_path, mixed_df):
    df = mixed_df.copy()
    df.loc[0, "y"] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        SurrogateTrainer(_config(tmp_path, log_target=True)).fit(df)


def test_native_importance_and_missing_target(tmp_path, mixed_df):
    df = mixed_df.copy()
    df.loc[:2, "y"] = np.nan
    cfg = _config(tmp_path, models=["random_forest"], importance_method="native")
    result = SurrogateTrainer(cfg).fit(df)
    assert result.n_rows_dropped == 3
    assert result.importance.method == "native"
    assert np.isclose(result.importance.importances.sum(), 1.0)


def test_workbench_file_round_trip(tmp_path, workbench_csv):
    df = load_dataset(workbench_csv.encode(), filename="dps.csv").df
    target = "Characteristic Life [cycles]"
    features = [c for c in df.columns[:10]]
    cfg = TrainingConfig(
        target=target,
        features=features,
        models=["gpr"],
        n_trials=2,
        cv_folds=3,
        log_target=True,
        output_dir=tmp_path,
        n_jobs=1,
    )
    result = SurrogateTrainer(cfg).fit(df, "dps.csv")
    assert result.best.test_metrics["r2"] > 0.8
