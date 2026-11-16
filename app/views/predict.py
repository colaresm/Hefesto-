from pathlib import Path

import pandas as pd
import streamlit as st
from ui_common import OUTPUT_DIR, STATE_ARTIFACTS

from hefesto.artifacts import list_runs
from hefesto.data import load_dataset
from hefesto.inference import LoadedModel, load_model, predict

st.title("Predict")
st.caption(
    "Evaluate new design points with a trained surrogate. Inputs outside the training "
    "domain are flagged - surrogates should interpolate, not extrapolate."
)

sources = []
if st.session_state.get(STATE_ARTIFACTS) is not None:
    sources.append("Model from this session")
runs = list_runs(OUTPUT_DIR)
if runs:
    sources.append("Saved run")
sources.append("Upload .joblib")
source = st.radio("Model source", sources, horizontal=True)

loaded: LoadedModel | None = None
try:
    if source == "Model from this session":
        loaded = load_model(st.session_state[STATE_ARTIFACTS].model_path)
    elif source == "Saved run":
        run = st.selectbox("Run", runs, format_func=lambda p: Path(p).name)
        loaded = load_model(Path(run) / "best_model.joblib")
    else:
        st.warning(
            "Only load model files you trust: joblib files can execute arbitrary code.",
            icon=":material/shield:",
        )
        up = st.file_uploader("Model file", type=["joblib", "pkl"])
        if up is not None:
            loaded = load_model(up.getvalue())
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

if loaded is None:
    st.stop()

meta = loaded.metadata
info = meta.get("best_model", {})
score_key, score_label = ("accuracy", "accuracy") if loaded.is_classification else ("r2", "R²")
score = (info.get("test_metrics") or {}).get(score_key)
if score is None:
    score = (info.get("cv_metrics") or {}).get(score_key)
with st.container(border=True):
    c1, c2, c3 = st.columns([2, 2, 1])
    c1.markdown(f"**Model**  \n{loaded.name}")
    c2.markdown(f"**Target**  \n{loaded.target}")
    c3.markdown(f"**Validated {score_label}**  \n" + (f"{score:.4f}" if score is not None else "-"))
    st.caption(f"{len(loaded.features)} input(s): " + ", ".join(loaded.features))
if not meta:
    st.info("This model has no Hefesto metadata; the training domain is unknown.")

domain = meta.get("training_domain", {})
tab_single, tab_batch = st.tabs(["Single design point", "Batch (file)"])

with tab_single:
    values = {}
    cols = st.columns(3)
    for i, feat in enumerate(loaded.features):
        d = domain.get(feat, {})
        with cols[i % 3]:
            if d.get("type") == "categorical":
                values[feat] = st.selectbox(feat, d.get("categories", []))
            else:
                lo, hi, med = d.get("min"), d.get("max"), d.get("median", 0.0)
                help_txt = f"Training range: [{lo:.6g}, {hi:.6g}]" if lo is not None else None
                values[feat] = st.number_input(feat, value=float(med), format="%.6g", help=help_txt)
    if st.button("Predict", type="primary", icon=":material/bolt:"):
        out = predict(loaded, pd.DataFrame([values]))
        pred = out[f"predicted_{loaded.target}"].iloc[0]
        if loaded.is_classification:
            st.metric(f"Predicted {loaded.target}", str(pred))
            proba_cols = [c for c in out.columns if c.startswith("proba_")]
            if proba_cols:
                probs = out[proba_cols].iloc[0].rename(lambda c: c.removeprefix("proba_"))
                st.bar_chart(probs.rename("Probability"), horizontal=True, height=180)
        else:
            st.metric(f"Predicted {loaded.target}", f"{pred:.6g}")
        if out.get("out_of_domain", pd.Series([False])).iloc[0]:
            st.warning(
                f"Extrapolation: {out['out_of_domain_features'].iloc[0]} outside the "
                "training range. Treat this prediction with caution.",
                icon=":material/warning:",
            )

with tab_batch:
    up = st.file_uploader(
        "Design points to evaluate", type=["csv", "txt", "xlsx", "xls", "parquet"], key="batch_file"
    )
    simplify = st.toggle("Simplify ANSYS parameter names", value=True, key="batch_simplify")
    if up is not None:
        try:
            data = load_dataset(
                up.getvalue(), filename=up.name, simplify_names=simplify, drop_constant=False
            ).df
            out = predict(loaded, data)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        n_ood = int(out.get("out_of_domain", pd.Series(dtype=bool)).sum())
        if n_ood:
            st.warning(
                f"{n_ood} of {len(out)} row(s) lie outside the training domain.",
                icon=":material/warning:",
            )
        st.dataframe(out, width="stretch", height=380)
        st.download_button(
            "Download predictions (.csv)",
            out.to_csv(index=False).encode(),
            file_name="hefesto_predictions.csv",
            mime="text/csv",
            type="primary",
            icon=":material/download:",
        )
