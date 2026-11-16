from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from hefesto.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".csv", ".txt", ".dat", ".xlsx", ".xls", ".parquet")

_IDENTIFIER_NAMES = {
    "name",
    "id",
    "index",
    "dp",
    "design point",
    "design_point",
    "designpoint",
    "unnamed: 0",
    "run",
    "sample",
}
_DP_LABEL = re.compile(r"^\s*(dp|design\s*point)\s*[-_ ]?\d+\s*$", re.IGNORECASE)
_PARAM_PREFIX = re.compile(r"^\s*P\d+\s*[-:]\s*")
_UNITS_IN_NAME = re.compile(r"\[([^\]]+)\]\s*$")
_FAILED_TOKENS = {"", "nan", "n/a", "na", "none", "null", "failed", "update failed", "error", "-"}


@dataclass
class DataReport:
    source: str = ""
    n_rows_raw: int = 0
    n_cols_raw: int = 0
    comment_lines: int = 0
    units: dict[str, str] = field(default_factory=dict)
    dropped_columns: dict[str, str] = field(default_factory=dict)
    dropped_rows: int = 0
    renamed_columns: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> list[str]:
        lines = [f"Loaded '{self.source}' ({self.n_rows_raw} rows x {self.n_cols_raw} columns)."]
        if self.comment_lines:
            lines.append(f"Skipped {self.comment_lines} comment/header line(s).")
        if self.units:
            lines.append(f"Detected units for {len(self.units)} column(s).")
        for col, reason in self.dropped_columns.items():
            lines.append(f"Dropped column '{col}': {reason}.")
        if self.dropped_rows:
            lines.append(f"Dropped {self.dropped_rows} empty or failed row(s).")
        if self.renamed_columns:
            lines.append(f"Simplified {len(self.renamed_columns)} ANSYS parameter name(s).")
        lines.extend(self.notes)
        return lines


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    report: DataReport


def _read_bytes(source: str | Path | BinaryIO | bytes) -> tuple[bytes, str]:
    if isinstance(source, bytes):
        return source, "uploaded"
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes(), path.name
    name = getattr(source, "name", "uploaded")
    data = source.read()
    return (data.encode() if isinstance(data, str) else data), Path(str(name)).name


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            if "\x00" not in text:
                return text
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _read_delimited(text: str, report: DataReport) -> pd.DataFrame:
    lines = text.splitlines()
    kept = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    report.comment_lines = sum(1 for ln in lines if ln.lstrip().startswith("#"))
    if not kept:
        raise ValueError("The file has no tabular content.")
    sample = "\n".join(kept[:50])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    if sep == " ":
        return pd.read_csv(io.StringIO("\n".join(kept)), sep=r"\s+", engine="python")
    return pd.read_csv(io.StringIO("\n".join(kept)), sep=sep, skipinitialspace=True)


def read_table(source: str | Path | BinaryIO | bytes, filename: str | None = None) -> LoadedDataset:
    raw, inferred_name = _read_bytes(source)
    name = filename or inferred_name
    ext = Path(name).suffix.lower()
    report = DataReport(source=name)
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(raw))
    elif ext == ".parquet":
        df = pd.read_parquet(io.BytesIO(raw))
    else:
        df = _read_delimited(_decode(raw), report)
    report.n_rows_raw, report.n_cols_raw = df.shape
    return LoadedDataset(df=df, report=report)


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    as_text = series.astype("string").str.strip()
    as_text = as_text.mask(as_text.str.lower().isin(_FAILED_TOKENS))
    converted = pd.to_numeric(as_text, errors="coerce")
    if converted.isna().sum() > as_text.isna().sum():
        alt = as_text.str.replace(r"\s", "", regex=True).str.replace(",", ".", regex=False)
        converted_alt = pd.to_numeric(alt, errors="coerce")
        if converted_alt.notna().sum() > converted.notna().sum():
            converted = converted_alt
    return converted.astype(float)


def _is_mostly_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    values = series.dropna()
    if not pd.api.types.is_numeric_dtype(values):
        values = values[~values.astype(str).str.strip().str.lower().isin(_FAILED_TOKENS)]
    if values.empty:
        return False
    return _to_numeric(values).notna().mean() >= threshold


def _extract_units_row(df: pd.DataFrame, report: DataReport) -> pd.DataFrame:
    if df.empty:
        return df
    first = df.iloc[0]
    first_cell = str(first.iloc[0]).strip().lower()
    body = df.iloc[1:]
    numeric_body_cols = [c for c in df.columns if _is_mostly_numeric(body[c])] if len(body) else []
    looks_like_units = first_cell in {"units", "unit", "[units]"}
    if not looks_like_units and numeric_body_cols:
        non_numeric = sum(
            pd.isna(_to_numeric(pd.Series([first[c]])).iloc[0]) and pd.notna(first[c])
            for c in numeric_body_cols
        )
        looks_like_units = non_numeric >= max(1, int(0.6 * len(numeric_body_cols)))
    if looks_like_units:
        for col in df.columns:
            value = first[col]
            if pd.notna(value) and str(value).strip() and str(value).strip().lower() != first_cell:
                report.units[str(col)] = str(value).strip()
        report.notes.append("Detected and removed a units row.")
        return df.iloc[1:].reset_index(drop=True)
    return df


def _is_identifier(col: str, series: pd.Series) -> bool:
    lowered = str(col).strip().lower()
    if lowered in _IDENTIFIER_NAMES or lowered.startswith("unnamed:"):
        if not pd.api.types.is_numeric_dtype(series):
            return True
        values = series.dropna().to_numpy()
        return len(values) > 0 and np.array_equal(
            values, np.arange(values[0], values[0] + len(values))
        )
    if not pd.api.types.is_numeric_dtype(series):
        values = series.dropna().astype(str)
        return len(values) > 0 and values.str.match(_DP_LABEL).mean() > 0.9
    return False


def simplify_ansys_names(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in columns:
        new = _PARAM_PREFIX.sub("", str(col)).strip() or str(col)
        if new in used:
            new = str(col).strip()
        used.add(new)
        if new != col:
            mapping[col] = new
    return mapping


def clean_dataframe(
    df: pd.DataFrame,
    report: DataReport | None = None,
    simplify_names: bool = True,
    drop_constant: bool = True,
) -> LoadedDataset:
    report = report or DataReport(source="dataframe", n_rows_raw=len(df), n_cols_raw=df.shape[1])
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    for c in empty_cols:
        report.dropped_columns[c] = "column is empty"
    df = df.drop(columns=empty_cols)

    df = _extract_units_row(df, report)

    for col in list(df.columns):
        if _is_identifier(col, df[col]):
            report.dropped_columns[col] = "identifier / design-point label"
            df = df.drop(columns=col)

    for col in df.columns:
        if _is_mostly_numeric(df[col]):
            df[col] = _to_numeric(df[col])

    n_before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        df = df[~df[numeric_cols].isna().all(axis=1)].reset_index(drop=True)
    report.dropped_rows += n_before - len(df)

    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1] if drop_constant else []
    for c in constant:
        report.dropped_columns[c] = "constant value (carries no information)"
    df = df.drop(columns=constant)

    for col in df.columns:
        match = _UNITS_IN_NAME.search(col)
        if match and col not in report.units:
            report.units[col] = match.group(1)

    if simplify_names:
        mapping = simplify_ansys_names(list(df.columns))
        if mapping:
            df = df.rename(columns=mapping)
            report.renamed_columns = mapping
            report.units = {mapping.get(k, k): v for k, v in report.units.items()}

    missing = int(df.isna().sum().sum())
    if missing:
        report.notes.append(
            f"{missing} missing value(s) remain; rows with a missing target are dropped at training "
            "time and missing inputs are median/mode-imputed inside the model pipeline."
        )
    if df.empty or df.shape[1] == 0:
        raise ValueError("After cleaning, the table has no usable rows or columns.")
    logger.info("Cleaned dataset: %d rows x %d columns", *df.shape)
    return LoadedDataset(df=df, report=report)


def load_dataset(
    source: str | Path | BinaryIO | bytes,
    filename: str | None = None,
    simplify_names: bool = True,
    drop_constant: bool = True,
) -> LoadedDataset:
    loaded = read_table(source, filename=filename)
    return clean_dataframe(
        loaded.df, loaded.report, simplify_names=simplify_names, drop_constant=drop_constant
    )


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    categorical = [f for f in features if f not in numeric]
    return numeric, categorical


def suggest_target(df: pd.DataFrame) -> str | None:
    cols = numeric_columns(df)
    return cols[-1] if cols else None


def feature_domain(df: pd.DataFrame, features: list[str]) -> dict[str, dict]:
    domain: dict[str, dict] = {}
    for f in features:
        s = df[f]
        if pd.api.types.is_numeric_dtype(s):
            domain[f] = {
                "type": "numeric",
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
            }
        else:
            domain[f] = {"type": "categorical", "categories": sorted(map(str, s.dropna().unique()))}
    return domain


MAX_CLASSES = 20


def is_class_label_column(s: pd.Series, max_classes: int = MAX_CLASSES) -> bool:
    values = s.dropna()
    n_unique = values.nunique()
    if n_unique < 2 or n_unique > max_classes:
        return False
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return True
    return bool(np.all(np.mod(values.astype(float), 1) == 0))


def suggest_task(s: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return "classification"
    return "classification" if is_class_label_column(s, max_classes=10) else "regression"


def class_labels(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s) and np.all(np.mod(s, 1) == 0):
        return s.astype("int64")
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.asarray(s).dtype)
    return s.astype(str).astype(object)


def validate_training_frame(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    cv_folds: int,
    task: str = "regression",
) -> None:
    missing = [c for c in [target, *features] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the dataset: {missing}")
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(
            f"The target '{target}' must be numeric for regression. "
            "Use the classification task for categorical labels."
        )
    n = int(df[target].notna().sum())
    if n < max(10, 2 * cv_folds):
        raise ValueError(
            f"Only {n} rows with a valid target; at least {max(10, 2 * cv_folds)} are required "
            f"for {cv_folds}-fold cross-validation."
        )
    if task == "classification":
        counts = df[target].dropna().value_counts()
        if len(counts) < 2:
            raise ValueError(f"The target '{target}' has a single class; at least two are needed.")
        if len(counts) > 50:
            raise ValueError(
                f"The target '{target}' has {len(counts)} distinct values - too many classes. "
                "Is it a continuous response (use regression)?"
            )
        rare = counts[counts < 2]
        if not rare.empty:
            raise ValueError(
                "Every class needs at least 2 rows for a stratified split; too few rows for "
                + ", ".join(f"'{k}' ({v})" for k, v in rare.items())
                + ". Remove or merge these classes."
            )
