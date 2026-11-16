from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

METHODS: dict[str, dict[str, Any]] = {
    "iqr": {"label": "IQR (Tukey fences)", "threshold": 1.5, "param": "k × IQR"},
    "zscore": {"label": "Robust z-score (median / MAD)", "threshold": 3.5, "param": "|z| limit"},
}


@dataclass
class OutlierReport:
    method: str
    threshold: float
    columns: list[str]
    bounds: dict[str, tuple[float, float]]
    counts: dict[str, int]
    row_mask: pd.Series = field(repr=False)

    @property
    def n_rows(self) -> int:
        return len(self.row_mask)

    @property
    def n_outlier_rows(self) -> int:
        return int(self.row_mask.sum())

    @property
    def has_outliers(self) -> bool:
        return self.n_outlier_rows > 0

    def table(self) -> pd.DataFrame:
        rows = [
            {
                "Column": c,
                "Outliers": self.counts[c],
                "% of rows": 100.0 * self.counts[c] / max(self.n_rows, 1),
                "Lower bound": self.bounds[c][0],
                "Upper bound": self.bounds[c][1],
            }
            for c in self.columns
        ]
        df = pd.DataFrame(
            rows, columns=["Column", "Outliers", "% of rows", "Lower bound", "Upper bound"]
        )
        return df.sort_values("Outliers", ascending=False, kind="stable").reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "columns": self.columns,
            "rows_flagged": self.n_outlier_rows,
            "rows_screened": self.n_rows,
            "per_column": {
                c: {"outliers": self.counts[c], "bounds": list(self.bounds[c])}
                for c in self.columns
                if self.counts[c]
            },
        }


def _bounds(s: pd.Series, method: str, k: float) -> tuple[float, float]:
    s = s.dropna().astype(float)
    if s.empty:
        return (-np.inf, np.inf)
    if method == "iqr":
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return (-np.inf, np.inf)
        return (float(q1 - k * iqr), float(q3 + k * iqr))
    if method == "zscore":
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        if mad == 0:
            return (-np.inf, np.inf)
        half_width = k * mad / 0.6745
        return (med - half_width, med + half_width)
    raise ValueError(f"Unknown outlier method '{method}'. Available: {sorted(METHODS)}")


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "iqr",
    threshold: float | None = None,
) -> OutlierReport:
    if method not in METHODS:
        raise ValueError(f"Unknown outlier method '{method}'. Available: {sorted(METHODS)}")
    k = float(METHODS[method]["threshold"] if threshold is None else threshold)
    columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    bounds: dict[str, tuple[float, float]] = {}
    counts: dict[str, int] = {}
    mask = pd.Series(False, index=df.index)
    for c in columns:
        lo, hi = _bounds(df[c], method, k)
        col_mask = (df[c] < lo) | (df[c] > hi)
        bounds[c], counts[c] = (lo, hi), int(col_mask.sum())
        mask |= col_mask
    return OutlierReport(method, k, columns, bounds, counts, mask)


def remove_outliers(df: pd.DataFrame, report: OutlierReport) -> pd.DataFrame:
    return df.loc[~report.row_mask.reindex(df.index, fill_value=False)].copy()
