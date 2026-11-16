import numpy as np
import pandas as pd
import pytest

from hefesto.outliers import detect_outliers, remove_outliers


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(10, 2, 200), "label": "x"})
    df.loc[3, "a"] = 25.0
    df.loc[7, "b"] = -40.0
    df.loc[11, "a"] = np.nan
    return df


@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_detects_injected_outliers(df_with_outliers, method):
    report = detect_outliers(df_with_outliers, ["a", "b", "label"], method)
    assert report.columns == ["a", "b"]
    assert report.row_mask[[3, 7]].all() and not report.row_mask[11]
    assert report.has_outliers and report.counts["a"] >= 1 and report.counts["b"] >= 1
    table = report.table()
    assert list(table.columns) == ["Column", "Outliers", "% of rows", "Lower bound", "Upper bound"]

    cleaned = remove_outliers(df_with_outliers, report)
    assert len(cleaned) == len(df_with_outliers) - report.n_outlier_rows
    assert 3 not in cleaned.index and 7 not in cleaned.index and 11 in cleaned.index
    assert report.to_dict()["rows_flagged"] == report.n_outlier_rows


def test_threshold_and_degenerate_columns():
    df = pd.DataFrame({"c": [1.0] * 9 + [5.0], "d": np.arange(10.0)})
    df.loc[9, "d"] = 30.0
    assert detect_outliers(df, ["d"], "iqr").counts["d"] == 1
    assert not detect_outliers(df, ["d"], "iqr", threshold=100).has_outliers
    assert detect_outliers(df, ["c"], "iqr").counts["c"] == 0
    assert detect_outliers(df, ["c"], "zscore").counts["c"] == 0
    with pytest.raises(ValueError, match="Unknown outlier method"):
        detect_outliers(df, ["c"], "nope")
