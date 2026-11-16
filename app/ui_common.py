from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import streamlit as st

from hefesto import __version__

OUTPUT_DIR = Path(os.getenv("HEFESTO_OUTPUT_DIR", "outputs"))
EXAMPLES_DIR = Path(
    os.getenv("HEFESTO_EXAMPLES_DIR", Path(__file__).resolve().parents[1] / "examples")
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#52514e"
GRID = "#e1e0d9"
EMBER = "#d9480f"

STATE_DATASET = "dataset"
STATE_DATASET_ORIGINAL = "dataset_original"
STATE_OUTLIERS = "outlier_removal"
STATE_DATASET_NAME = "dataset_name"
STATE_SUGGESTED_FEATURES = "suggested_features"
STATE_SUGGESTED_TARGET = "suggested_target"
STATE_RESULT = "training_result"
STATE_ARTIFACTS = "run_artifacts"

_CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1350px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .hefesto-hero {
      background: linear-gradient(120deg, #1f1e1c 0%, #3b2a20 55%, #d9480f 140%);
      color: #fcfcfb; border-radius: 1rem; padding: 1.6rem 2rem; margin-bottom: 1.4rem;
  }
  .hefesto-hero h1 {color: #fcfcfb; margin: 0 0 .3rem 0; font-size: 2.1rem;}
  .hefesto-hero p {color: #e9e4dc; margin: 0; font-size: 1.02rem;}
  .hefesto-hero .tag {display:inline-block; background: rgba(252,252,251,.14);
      border: 1px solid rgba(252,252,251,.25); border-radius: 999px; padding: .1rem .65rem;
      font-size: .78rem; margin-right: .35rem; margin-top: .7rem;}
  .hefesto-card {border: 1px solid #e1e0d9; border-radius: .9rem; padding: 1rem 1.1rem;
      background: #ffffff; height: 100%;}
  .hefesto-card h4 {margin: 0 0 .35rem 0; font-size: 1.02rem;}
  .hefesto-card p {margin: 0; color: #52514e; font-size: .92rem;}
  .hefesto-step {color: #d9480f; font-weight: 700; font-size: .8rem; letter-spacing: .06em;}
  div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e1e0d9;
      border-radius: .8rem; padding: .7rem .9rem;}
  section[data-testid="stSidebar"] {border-right: 1px solid #e1e0d9;}
</style>
"""


def apply_style() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tags: list[str] | None = None) -> None:
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in (tags or []))
    st.markdown(
        f'<div class="hefesto-hero"><h1>{title}</h1><p>{subtitle}</p>{tag_html}</div>',
        unsafe_allow_html=True,
    )


def card(step: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="hefesto-card"><div class="hefesto-step">{step}</div><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def sidebar_status() -> None:
    with st.sidebar:
        st.markdown("### Session")
        ds = st.session_state.get(STATE_DATASET)
        if ds is not None:
            st.success(
                f"Dataset: **{st.session_state.get(STATE_DATASET_NAME, 'dataset')}**  \n"
                f"{ds.df.shape[0]} rows × {ds.df.shape[1]} columns",
                icon=":material/table:",
            )
            removal = st.session_state.get(STATE_OUTLIERS)
            if removal:
                st.caption(f"{removal['rows_flagged']} outlier row(s) removed")
        else:
            st.info("No dataset loaded yet.", icon=":material/upload_file:")
        result = st.session_state.get(STATE_RESULT)
        if result is not None:
            st.success(f"Best model: **{result.best.name}**", icon=":material/emoji_events:")
        st.divider()
        st.caption(f"Hefesto v{__version__} · Outputs: `{OUTPUT_DIR}`")


def require_dataset():
    ds = st.session_state.get(STATE_DATASET)
    if ds is None:
        st.warning("Load a dataset first.", icon=":material/warning:")
        st.page_link("views/data.py", label="Go to Data", icon=":material/database:")
        st.stop()
    return ds


def require_result():
    result = st.session_state.get(STATE_RESULT)
    if result is None:
        st.warning("No training run in this session yet.", icon=":material/warning:")
        st.page_link("views/train.py", label="Go to Train", icon=":material/model_training:")
        st.stop()
    return result


def chart_config(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            gridColor=GRID,
            domainColor=GRID,
            labelColor=GRAY,
            titleColor="#1f1e1c",
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor=GRAY, titleColor="#1f1e1c", orient="top", labelLimit=0)
    )
