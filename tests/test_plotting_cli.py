import matplotlib.pyplot as plt
import numpy as np

from hefesto.cli import main
from hefesto.importance import FeatureImportance
from hefesto.plotting import plot_surrogate_report
from hefesto.samples import make_bga_fatigue_doe, to_workbench_csv


def test_report_figure_has_two_titled_english_panels():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 10, 40)
    fi = FeatureImportance(
        ["a", "b", "c"],
        np.array([0.5, 0.2, -0.01]),
        np.array([0.05, 0.02, 0.01]),
        "permutation",
        "Mean increase in RMSE when permuted (hold-out set)",
    )
    fig = plot_surrogate_report(
        model_name="XGBoost",
        target="Warpage",
        importance=fi,
        y_cv=y,
        pred_cv=y + rng.normal(0, 0.1, 40),
        y_test=y[:8],
        pred_test=y[:8] * 1.01,
        units="um",
    )
    axes = fig.get_axes()
    assert len(axes) == 2
    assert axes[0].get_title().startswith("Feature Importance")
    assert axes[1].get_title() == "Actual vs. Predicted"
    assert axes[1].get_xlabel() == "Actual Warpage [um]"
    assert axes[1].get_ylabel() == "Predicted Warpage [um]"
    assert "Hefesto Surrogate Report" in fig.get_suptitle()
    plt.close(fig)


def test_cli_train_and_predict(tmp_path, capsys):
    data = tmp_path / "doe.csv"
    data.write_text(to_workbench_csv(make_bga_fatigue_doe(60, seed=1)))
    code = main(
        [
            "train",
            "--data",
            str(data),
            "--target",
            "Max Package Warpage [um]",
            "--exclude",
            "Plastic Work Density [MPa]",
            "Characteristic Life [cycles]",
            "--models",
            "ridge",
            "polynomial",
            "--trials",
            "2",
            "--folds",
            "3",
            "--output",
            str(tmp_path / "out"),
            "--n-jobs",
            "1",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Best model" in out
    model_path = next((tmp_path / "out").glob("*/best_model.joblib"))
    preds = tmp_path / "preds.csv"
    assert (
        main(["predict", "--model", str(model_path), "--data", str(data), "--output", str(preds)])
        == 0
    )
    assert preds.exists()
    assert main(["list-models"]) == 0
