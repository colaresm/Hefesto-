import pandas as pd
import pytest

from hefesto.data import (
    clean_dataframe,
    feature_domain,
    load_dataset,
    simplify_ansys_names,
    split_feature_types,
    suggest_target,
    validate_training_frame,
)


def test_workbench_export_is_cleaned(workbench_csv, bga_df):
    loaded = load_dataset(workbench_csv.encode(), filename="dps.csv")
    df, report = loaded.df, loaded.report
    assert "Name" in report.dropped_columns
    assert report.comment_lines == 3
    assert df.shape == (len(bga_df), bga_df.shape[1])
    assert "Die Thickness [mm]" in df.columns
    assert report.units["Characteristic Life [cycles]"] == "cycles"
    assert df["Characteristic Life [cycles]"].isna().sum() == 2
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)


def test_keep_parameter_ids(workbench_csv):
    df = load_dataset(workbench_csv.encode(), filename="dps.csv", simplify_names=False).df
    assert df.columns[0].startswith("P1 - ")


def test_units_row_semicolon_and_decimal_comma():
    text = (
        "Name;P1 - Thickness;P2 - Load;P3 - Stress\n"
        "Units;mm;N;MPa\n"
        "DP 0;0,5;10;101,5\n"
        "DP 1;0,6;12;Update failed\n"
        "DP 2;0,7;14;140,25\n"
        "DP 3;0,8;16;160,0\n"
    )
    loaded = load_dataset(text.encode(), filename="t.csv")
    df = loaded.df
    assert list(df.columns) == ["Thickness", "Load", "Stress"]
    assert df["Thickness"].tolist() == [0.5, 0.6, 0.7, 0.8]
    assert pd.isna(df.loc[1, "Stress"])
    assert loaded.report.units == {"Thickness": "mm", "Load": "N", "Stress": "MPa"}


def test_constant_and_empty_columns_dropped():
    raw = pd.DataFrame(
        {"a": [1, 2, 3], "const": [7, 7, 7], "empty": [None] * 3, "y": [1.0, 4.0, 9.0]}
    )
    loaded = clean_dataframe(raw)
    assert list(loaded.df.columns) == ["a", "y"]
    assert set(loaded.report.dropped_columns) == {"const", "empty"}
    kept = clean_dataframe(raw.drop(columns="empty"), drop_constant=False)
    assert "const" in kept.df.columns


def test_simplify_names_keeps_uniqueness():
    mapping = simplify_ansys_names(["P1 - Stress", "P2 - Stress", "Other"])
    assert mapping == {"P1 - Stress": "Stress"}


def test_helpers(mixed_df):
    assert suggest_target(mixed_df) == "y"
    num, cat = split_feature_types(mixed_df, ["x1", "material", "x2"])
    assert num == ["x1", "x2"] and cat == ["material"]
    dom = feature_domain(mixed_df, ["x1", "material"])
    assert dom["x1"]["type"] == "numeric" and dom["material"]["categories"] == ["SAC305", "SnPb"]


def test_validate_training_frame(mixed_df):
    validate_training_frame(mixed_df, "y", ["x1"], cv_folds=5)
    with pytest.raises(ValueError, match="must be numeric"):
        validate_training_frame(mixed_df, "material", ["x1"], cv_folds=5)
    with pytest.raises(ValueError, match="at least"):
        validate_training_frame(mixed_df.head(8), "y", ["x1"], cv_folds=5)
