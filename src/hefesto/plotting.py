from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from hefesto.importance import FeatureImportance  # noqa: E402
from hefesto.metrics import compute_classification_metrics, compute_metrics  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
IDEAL_LINE = "#52514e"

_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "font.family": "DejaVu Sans",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _fmt(value: float) -> str:
    if value == 0 or not np.isfinite(value):
        return "0"
    if abs(value) >= 1e4:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def _wrap(label: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(str(label), width)) or str(label)


def _plot_importance(ax, importance: FeatureImportance | None, top_n: int) -> None:
    if importance is None:
        ax.text(
            0.5,
            0.5,
            "Feature importance unavailable",
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            transform=ax.transAxes,
        )
        ax.set_title("Feature Importance")
        ax.set_axis_off()
        return
    df = importance.top(top_n).iloc[::-1]
    y = np.arange(len(df))
    xerr = df["std"].to_numpy() if "std" in df else None
    ax.barh(
        y, df["importance"], height=0.62, color=SERIES_1, edgecolor="white", linewidth=1.5, zorder=3
    )
    if xerr is not None:
        ax.errorbar(
            df["importance"],
            y,
            xerr=xerr,
            fmt="none",
            ecolor=TEXT_SECONDARY,
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
    span = float(np.nanmax(np.abs(df["importance"] + (xerr if xerr is not None else 0)))) or 1.0
    for yi, val, err in zip(
        y, df["importance"], xerr if xerr is not None else np.zeros(len(df)), strict=True
    ):
        ax.text(
            val + (err if val >= 0 else -err) + 0.015 * span * (1 if val >= 0 else -1),
            yi,
            _fmt(val),
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5,
            color=TEXT_SECONDARY,
            zorder=5,
        )
    ax.set_yticks(y, [_wrap(f) for f in df["feature"]])
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8, zorder=2)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.12 * span)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(_wrap(importance.axis_label, 60))
    method = "Permutation" if importance.method == "permutation" else "Native"
    shown = f"Top {len(df)}" if len(importance.features) > len(df) else "All"
    ax.set_title(f"Feature Importance ({method}, {shown} Inputs)")


def _plot_parity(ax, y_cv, p_cv, y_test, p_test, target: str, units: str | None) -> None:
    label_units = f" [{units}]" if units and f"[{units}]" not in target else ""
    arrays = [np.asarray(a, dtype=float) for a in (y_cv, p_cv) if a is not None]
    if y_test is not None and p_test is not None:
        arrays += [np.asarray(y_test, dtype=float), np.asarray(p_test, dtype=float)]
    lo = min(np.nanmin(a) for a in arrays)
    hi = max(np.nanmax(a) for a in arrays)
    pad = 0.05 * (hi - lo or abs(hi) or 1.0)
    lo, hi = lo - pad, hi + pad

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color=IDEAL_LINE,
        linewidth=1.2,
        label="Ideal (y = x)",
        zorder=2,
    )
    band = np.linspace(lo, hi, 50)
    ax.fill_between(
        band,
        band * 0.9,
        band * 1.1,
        color=GRID,
        alpha=0.6,
        linewidth=0,
        label="±10% band",
        zorder=1,
    )

    text_lines = []
    if y_cv is not None and p_cv is not None:
        mask = ~np.isnan(p_cv)
        ax.scatter(
            np.asarray(y_cv)[mask],
            np.asarray(p_cv)[mask],
            s=34,
            color=SERIES_1,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label="Cross-validation (out-of-fold)",
        )
        m = compute_metrics(np.asarray(y_cv)[mask], np.asarray(p_cv)[mask])
        text_lines.append(f"CV:        R² = {m['r2']:.4f}   RMSE = {m['rmse']:.4g}")
    if y_test is not None and p_test is not None and len(y_test):
        ax.scatter(
            y_test,
            p_test,
            s=44,
            color=SERIES_2,
            marker="D",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="Hold-out test",
        )
        m = compute_metrics(y_test, p_test)
        text_lines.append(f"Hold-out: R² = {m['r2']:.4f}   RMSE = {m['rmse']:.4g}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(_wrap(f"Actual {target}{label_units}", 60))
    ax.set_ylabel(_wrap(f"Predicted {target}{label_units}", 50))
    ax.set_title("Actual vs. Predicted")
    ax.legend(loc="lower right")
    if text_lines:
        ax.text(
            0.03,
            0.97,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="DejaVu Sans Mono",
            color=TEXT_PRIMARY,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=GRID),
        )


def _plot_confusion(ax, y_true, y_pred, classes, split: str) -> None:
    classes = list(classes)
    index = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p_ in zip(np.asarray(y_true), np.asarray(y_pred), strict=True):
        cm[index[t], index[p_]] += 1
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    font = 11 if k <= 4 else 9 if k <= 8 else 7
    for i in range(k):
        for j in range(k):
            dark = row_share[i, j] > 0.55
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{100 * row_share[i, j]:.0f}%",
                ha="center",
                va="center",
                fontsize=font,
                color="white" if dark else TEXT_PRIMARY,
                fontweight="bold" if i == j else "normal",
            )
    labels = [_wrap(str(c), 14) for c in classes]
    ax.set_xticks(range(k), labels, rotation=30 if k > 4 else 0, ha="right" if k > 4 else "center")
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    m = compute_classification_metrics(y_true, y_pred, classes=classes)
    ax.set_title(f"Confusion Matrix ({split})")
    ax.text(
        0.5,
        -0.16 if k <= 4 else -0.24,
        f"Accuracy = {m['accuracy']:.4f}   F1 (macro) = {m['f1_macro']:.4f}   "
        f"Balanced accuracy = {m['balanced_accuracy']:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        family="DejaVu Sans Mono",
        color=TEXT_PRIMARY,
    )


def plot_surrogate_report(
    *,
    model_name: str,
    target: str,
    importance: FeatureImportance | None,
    y_cv=None,
    pred_cv=None,
    y_test=None,
    pred_test=None,
    top_n: int = 20,
    units: str | None = None,
    subtitle: str | None = None,
    confusion: dict | None = None,
) -> Figure:
    height = _importance_height(importance, top_n)
    with plt.rc_context(_RC):
        fig, (ax_imp, ax_par) = plt.subplots(
            1,
            2,
            figsize=(16, height),
            gridspec_kw={"width_ratios": [1.15, 1.0]},
            layout="constrained",
        )
        _plot_importance(ax_imp, importance, top_n)
        if confusion is not None:
            _plot_confusion(ax_par, **confusion)
            kind = "Classifier"
        else:
            _plot_parity(ax_par, y_cv, pred_cv, y_test, pred_test, target, units)
            kind = "Surrogate"
        title = f"Hefesto {kind} Report — {model_name}"
        fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT_PRIMARY)
        caption = subtitle or f"Target: {target}"
        fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=9.5, color=TEXT_SECONDARY)
    return fig


def _importance_height(importance: FeatureImportance | None, top_n: int) -> float:
    n_bars = min(top_n, len(importance.features)) if importance else 5
    return float(np.clip(0.42 * n_bars + 2.6, 6.2, 12.0))


def plot_feature_importance(importance: FeatureImportance | None, top_n: int = 20) -> Figure:
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(
            figsize=(9, _importance_height(importance, top_n)), layout="constrained"
        )
        _plot_importance(ax, importance, top_n)
    return fig


def plot_actual_vs_predicted(
    *,
    target: str,
    y_cv=None,
    pred_cv=None,
    y_test=None,
    pred_test=None,
    units: str | None = None,
) -> Figure:
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
        _plot_parity(ax, y_cv, pred_cv, y_test, pred_test, target, units)
    return fig


def plot_confusion_matrix(y_true, y_pred, classes, split: str) -> Figure:
    size = float(np.clip(1.1 * len(classes) + 3.5, 6.0, 14.0))
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
        _plot_confusion(ax, y_true, y_pred, classes, split)
    return fig


def plot_metrics_table(
    table: pd.DataFrame,
    *,
    title: str = "Model Metrics",
    highlight_row: int | None = None,
    note: str | None = None,
) -> Figure:
    n_rows, n_cols = table.shape
    width = float(np.clip(1.25 * n_cols + 2.0, 11.0, 22.0))
    height = 1.6 + 0.42 * (n_rows + 1) + (0.4 if note else 0)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_axis_off()
        tbl = ax.table(
            cellText=table.to_numpy(),
            colLabels=[_wrap(c, 14) for c in table.columns],
            cellLoc="center",
            colLoc="center",
            loc="upper center",
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        model_col = list(table.columns).index("Model") if "Model" in table.columns else None
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.8)
            if row == 0:
                cell.set_facecolor("#1f1e1c")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
                cell.set_height(cell.get_height() * 1.6)
            else:
                cell.set_facecolor(SURFACE if row % 2 else "white")
                cell.get_text().set_color(TEXT_PRIMARY)
                if highlight_row is not None and row - 1 == highlight_row:
                    cell.set_facecolor("#fde8dc")
                    cell.get_text().set_fontweight("bold")
            if col == model_col:
                cell.get_text().set_ha("left")
                cell.PAD = 0.03
        if model_col is not None:
            widths = np.ones(n_cols)
            widths[model_col] = 2.4
            widths /= widths.sum()
            for (_, col), cell in tbl.get_celld().items():
                cell.set_width(widths[col])
        ax.set_title(title, pad=14)
        if note:
            fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)
        fig.tight_layout(rect=(0.02, 0.05 if note else 0.02, 0.98, 0.98))
    return fig


def save_pdf(figures: list[Figure], path) -> str:
    path = str(path)
    with plt.rc_context(_RC), PdfPages(path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
    return path


def save_figure(
    fig: Figure, path_without_suffix, formats: tuple[str, ...] = ("png", "pdf")
) -> list:
    paths = []
    with plt.rc_context(_RC):
        for fmt in formats:
            path = f"{path_without_suffix}.{fmt}"
            fig.savefig(path, format=fmt)
            paths.append(path)
    return paths
