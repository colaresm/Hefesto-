import dataclasses

import altair as alt
import pandas as pd
import streamlit as st
from ui_common import (
    BLUE,
    EXAMPLES_DIR,
    GRAY,
    GRID,
    ORANGE,
    STATE_ARTIFACTS,
    STATE_DATASET,
    STATE_DATASET_NAME,
    STATE_DATASET_ORIGINAL,
    STATE_OUTLIERS,
    STATE_RESULT,
    STATE_SUGGESTED_FEATURES,
    STATE_SUGGESTED_TARGET,
    chart_config,
)

from hefesto.data import SUPPORTED_EXTENSIONS, LoadedDataset, load_dataset, numeric_columns
from hefesto.outliers import METHODS, detect_outliers, remove_outliers
from hefesto.samples import INPUTS, make_bga_fatigue_doe, to_workbench_csv

SAMPLE_FILE = "bga_solder_fatigue_doe.csv"

st.title("Data")
st.caption(
    "Load the design-of-experiments table exported from ANSYS (or any tabular simulation output)."
)


def _store(loaded, name: str, suggested_target=None, suggested_features=None) -> None:
    st.session_state[STATE_DATASET] = loaded
    st.session_state[STATE_DATASET_ORIGINAL] = loaded
    st.session_state.pop(STATE_OUTLIERS, None)
    st.session_state[STATE_DATASET_NAME] = name
    st.session_state[STATE_SUGGESTED_TARGET] = suggested_target
    st.session_state[STATE_SUGGESTED_FEATURES] = suggested_features
    st.session_state.pop(STATE_RESULT, None)
    st.session_state.pop(STATE_ARTIFACTS, None)
    for key in (
        "train_task",
        "train_target",
        "train_features",
        "_features_for_target",
        "train_outlier_columns",
    ):
        st.session_state.pop(key, None)


src_col, opt_col = st.columns([2, 1])
with opt_col:
    st.markdown("##### Parsing options")
    simplify = st.toggle(
        "Simplify ANSYS parameter names",
        value=True,
        help="Turns 'P3 - Total Deformation [mm]' into 'Total Deformation [mm]'.",
    )
with src_col:
    upload = st.file_uploader(
        "Simulation table",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        help="CSV/TXT (any delimiter, '#' comments allowed), Excel or Parquet.",
    )
    b1, b2 = st.columns([1, 2])
    use_sample = b1.button("Use demo dataset", icon=":material/science:")
    b2.caption(
        "Synthetic BGA solder-joint thermal-cycling DOE (200 LHS design points, "
        "ANSYS Workbench-like layout). For demonstration only."
    )

try:
    if upload is not None and st.session_state.get("_last_upload") != (
        upload.name,
        upload.size,
        simplify,
    ):
        loaded = load_dataset(upload.getvalue(), filename=upload.name, simplify_names=simplify)
        _store(loaded, upload.name)
        st.session_state["_last_upload"] = (upload.name, upload.size, simplify)
        st.rerun()
    if use_sample:
        sample_path = EXAMPLES_DIR / SAMPLE_FILE
        raw = (
            sample_path.read_bytes()
            if sample_path.exists()
            else to_workbench_csv(make_bga_fatigue_doe()).encode()
        )
        loaded = load_dataset(raw, filename=SAMPLE_FILE, simplify_names=simplify)
        cols = list(loaded.df.columns)
        features = [c for c in cols if any(c.split(" [")[0].endswith(k) for k in INPUTS)]
        target = next((c for c in cols if c.startswith(("Characteristic Life", "P12"))), None)
        _store(loaded, SAMPLE_FILE, target, features)
        st.session_state["_last_upload"] = None
        st.rerun()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the file: {exc}", icon=":material/error:")
    st.stop()

ds = st.session_state.get(STATE_DATASET)
if ds is None:
    st.info("Upload a file or use the demo dataset to continue.", icon=":material/upload_file:")
    st.stop()

original: LoadedDataset = st.session_state.get(STATE_DATASET_ORIGINAL) or ds
st.session_state[STATE_DATASET_ORIGINAL] = original
removal = st.session_state.get(STATE_OUTLIERS)
df: pd.DataFrame = ds.df
num_cols = numeric_columns(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Design points", f"{len(df):,}")
m2.metric("Columns", df.shape[1])
m3.metric("Numeric columns", len(num_cols))
m4.metric("Missing values", int(df.isna().sum().sum()))

with st.expander("Cleaning report", expanded=True, icon=":material/rule:"):
    for line in ds.report.summary():
        st.markdown(f"- {line}")
    extra_units = {k: v for k, v in ds.report.units.items() if f"[{v}]" not in k}
    if extra_units:
        st.caption("Units: " + " Â· ".join(f"**{k}**: {v}" for k, v in extra_units.items()))

st.subheader("Outlier check")
orig_df = original.df
orig_num = numeric_columns(orig_df)
st.session_state.setdefault("train_outlier_method", "iqr")
o1, o2, o3 = st.columns([1.2, 0.8, 3])
o_method = o1.radio(
    "Method",
    list(METHODS),
    format_func=lambda k: METHODS[k]["label"],
    key="train_outlier_method",
    help="IQR: outside [Q1 âˆ’ kÂ·IQR, Q3 + kÂ·IQR]. Robust z-score: |0.6745Â·(x âˆ’ median)/MAD| > k.",
)
if st.session_state.get("train_outlier_threshold_for") != o_method:
    st.session_state["train_outlier_threshold"] = float(METHODS[o_method]["threshold"])
    st.session_state["train_outlier_threshold_for"] = o_method
o_threshold = o2.number_input(
    f"Threshold ({METHODS[o_method]['param']})",
    min_value=0.5,
    max_value=20.0,
    step=0.5,
    key="train_outlier_threshold",
)
stale = [c for c in st.session_state.get("train_outlier_columns", []) if c not in orig_num]
if stale or "train_outlier_columns" not in st.session_state:
    suggested = [
        c
        for c in [
            st.session_state.get(STATE_SUGGESTED_TARGET),
            *(st.session_state.get(STATE_SUGGESTED_FEATURES) or []),
        ]
        if c in orig_num
    ]
    st.session_state["train_outlier_columns"] = suggested or orig_num
screen_cols = o3.multiselect(
    "Columns to screen",
    orig_num,
    key="train_outlier_columns",
    help="Usually the design inputs you will train on, plus the target for regression. "
    "Do not screen class-label columns: a rare class is not an outlier.",
)
outliers = detect_outliers(orig_df, screen_cols, o_method, o_threshold)
flagged = outliers.table()
flagged = flagged[flagged["Outliers"] > 0]

if not screen_cols:
    st.info("Select at least one numeric column to screen.", icon=":material/info:")
elif not outliers.has_outliers:
    st.success(
        f"No outliers found in {len(screen_cols)} column(s) with {METHODS[o_method]['label']} "
        f"(threshold {o_threshold:g}).",
        icon=":material/check_circle:",
    )
else:
    share = 100 * outliers.n_outlier_rows / max(outliers.n_rows, 1)
    st.warning(
        f"**{outliers.n_outlier_rows}** of {outliers.n_rows} design points ({share:.1f}%) have at "
        f"least one outlier in **{len(flagged)}** of {len(screen_cols)} screened column(s).",
        icon=":material/warning:",
    )
    st.dataframe(
        flagged.style.format(
            {"% of rows": "{:.1f}", "Lower bound": "{:.5g}", "Upper bound": "{:.5g}"}
        ),
        hide_index=True,
        width="stretch",
        height=min(35 * (len(flagged) + 1) + 3, 320),
    )
    with st.expander("Flagged design points", icon=":material/table_rows:"):
        st.dataframe(orig_df.loc[outliers.row_mask], width="stretch", height=280)


def _set_dataset(loaded: LoadedDataset, applied_removal: dict | None) -> None:
    st.session_state[STATE_DATASET] = loaded
    st.session_state[STATE_OUTLIERS] = applied_removal
    st.session_state.pop(STATE_RESULT, None)
    st.session_state.pop(STATE_ARTIFACTS, None)


applied = removal is not None
b1, b2, b3 = st.columns([1.3, 1.1, 3])
if outliers.has_outliers and b1.button(
    "Re-apply with these settings"
    if applied
    else f"Remove {outliers.n_outlier_rows} outlier row(s)",
    type="primary",
    icon=":material/filter_alt:",
):
    note = (
        f"Removed {outliers.n_outlier_rows} outlier row(s) with {METHODS[o_method]['label']} "
        f"(threshold {o_threshold:g}) on {len(screen_cols)} column(s)."
    )
    report = dataclasses.replace(original.report, notes=[*original.report.notes, note])
    _set_dataset(LoadedDataset(remove_outliers(orig_df, outliers), report), outliers.to_dict())
    st.rerun()
if applied and b2.button("Restore original data", icon=":material/undo:"):
    _set_dataset(original, None)
    st.rerun()
if applied:
    b3.info(
        f"Training will use **{len(df)}** design points: {removal['rows_flagged']} outlier row(s) "
        f"removed ({METHODS[removal['method']]['label']}, threshold {removal['threshold']:g}). "
        "Compare the distributions in the **Distributions** tab.",
        icon=":material/filter_alt:",
    )

tab_preview, tab_stats, tab_corr, tab_dist = st.tabs(
    ["Preview", "Statistics", "Correlations", "Distributions"]
)
with tab_preview:
    st.dataframe(df, width="stretch", height=380)
with tab_stats:
    st.dataframe(df.describe(include="all").T, width="stretch")
with tab_corr:
    if len(num_cols) < 2:
        st.info("At least two numeric columns are needed.")
    else:
        method = st.radio(
            "Method", ["pearson", "spearman"], horizontal=True, format_func=str.capitalize
        )
        corr = df[num_cols].corr(method=method)
        long = corr.reset_index(names="row").melt(id_vars="row", var_name="column", value_name="r")
        heat = (
            alt.Chart(long)
            .mark_rect(stroke="white", strokeWidth=2)
            .encode(
                x=alt.X(
                    "column:N",
                    sort=num_cols,
                    title=None,
                    axis=alt.Axis(labelAngle=-40, labelLimit=180),
                ),
                y=alt.Y("row:N", sort=num_cols, title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color(
                    "r:Q",
                    title="r",
                    scale=alt.Scale(domain=[-1, 0, 1], range=[BLUE, "#f0efec", "#e34948"]),
                ),
                tooltip=["row", "column", alt.Tooltip("r:Q", format=".3f")],
            )
        )
        text = (
            alt.Chart(long)
            .mark_text(fontSize=10)
            .encode(
                x=alt.X("column:N", sort=num_cols),
                y=alt.Y("row:N", sort=num_cols),
                text=alt.Text("r:Q", format=".2f"),
                color=alt.condition("abs(datum.r) > 0.6", alt.value("white"), alt.value("#1f1e1c")),
            )
        )
        size = max(360, 38 * len(num_cols))
        st.altair_chart(chart_config((heat + text).properties(height=size)), width="stretch")
with tab_dist:
    if num_cols:
        default_col = next((c for c in flagged["Column"] if c in num_cols), num_cols[-1])
        col = st.selectbox("Column", num_cols, index=num_cols.index(default_col))
        if applied:
            values = pd.concat(
                [
                    orig_df[[col]].dropna().assign(Data="Original"),
                    df[[col]].dropna().assign(Data="Outliers removed"),
                ],
                ignore_index=True,
            ).rename(columns={col: "value"})
            hist = (
                alt.Chart(values)
                .mark_bar(opacity=0.55, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=col),
                    y=alt.Y("count()", stack=None, title="Design points"),
                    color=alt.Color(
                        "Data:N",
                        scale=alt.Scale(
                            domain=["Original", "Outliers removed"], range=[GRAY, BLUE]
                        ),
                        title=None,
                    ),
                    tooltip=["Data:N", alt.Tooltip("count()", title="Design points")],
                )
            )
        else:
            values = df[[col]].dropna().rename(columns={col: "value"})
            hist = (
                alt.Chart(values)
                .mark_bar(
                    color=BLUE,
                    cornerRadiusTopLeft=4,
                    cornerRadiusTopRight=4,
                    stroke=GRID,
                    strokeWidth=0,
                )
                .encode(
                    x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=col),
                    y=alt.Y("count()", title="Design points"),
                    tooltip=[alt.Tooltip("count()", title="Design points")],
                )
            )
        layers = [hist]
        lo, hi = outliers.bounds.get(col, (None, None))
        vmin, vmax = values["value"].min(), values["value"].max()
        inside = [b for b in (lo, hi) if b is not None and vmin <= b <= vmax]
        if inside:
            layers.append(
                alt.Chart(pd.DataFrame({"bound": inside}))
                .mark_rule(color=ORANGE, strokeDash=[6, 4], strokeWidth=1.6)
                .encode(
                    x="bound:Q",
                    tooltip=[alt.Tooltip("bound:Q", title="Outlier bound", format=".5g")],
                )
            )
        st.altair_chart(chart_config(alt.layer(*layers).properties(height=320)), width="stretch")
        if col in outliers.bounds:
            st.caption(
                f"Dashed orange lines: outlier bounds ({METHODS[o_method]['label']}, threshold "
                f"{o_threshold:g}) Â· {outliers.counts[col]} value(s) outside them in this column."
            )

        def _describe(s: pd.Series) -> dict:
            s = s.dropna()
            return {
                "Count": len(s),
                "Mean": s.mean(),
                "Std": s.std(),
                "Min": s.min(),
                "Median": s.median(),
                "Max": s.max(),
                "Skewness": s.skew(),
                "Kurtosis": s.kurt(),
            }

        stats = {"Original": _describe(orig_df[col])}
        if applied:
            before, after = stats["Original"], _describe(df[col])
            stats["Outliers removed"] = after
            stats["Change (%)"] = {
                k: 100 * (after[k] - before[k]) / abs(before[k]) if before[k] else float("nan")
                for k in before
            }
        st.dataframe(pd.DataFrame(stats).T.style.format("{:.5g}", na_rep="-"), width="stretch")

st.page_link("views/train.py", label="Continue to Train", icon=":material/arrow_forward:")
