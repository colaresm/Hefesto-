import numpy as np
import optuna
import pytest

from hefesto.models import BuildContext, available_models, build_pipeline, get_model_spec

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.mark.parametrize("key", sorted(available_models()))
def test_every_model_samples_builds_fits_and_predicts(key, mixed_df):
    spec = available_models()[key]
    ctx = BuildContext(random_state=0, n_jobs=1, n_samples=len(mixed_df), n_features=3)
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1)).ask()
    params = spec.sample_params(trial, ctx)
    model = build_pipeline(spec, params, ["x1", "x2"], ["material"], ctx, log_target=True)
    X, y = mixed_df[["x1", "x2", "material"]], mixed_df["y"]
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.all(np.isfinite(pred))


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_spec("does_not_exist")


def test_target_transform_returns_original_units(mixed_df):
    spec = get_model_spec("linear")
    model = build_pipeline(spec, {}, ["x1", "x2"], [], BuildContext(), log_target=True)
    model.fit(mixed_df[["x1", "x2"]], mixed_df["y"])
    pred = model.predict(mixed_df[["x1", "x2"]])
    assert 3 < pred.mean() < 10
