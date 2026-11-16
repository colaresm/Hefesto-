from __future__ import annotations

import streamlit as st

from hefesto.logging_utils import configure_logging

configure_logging()

st.set_page_config(
    page_title="Hefesto · Surrogate Model Forge",
    page_icon=":material/local_fire_department:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui_common import apply_style, sidebar_status  # noqa: E402

apply_style()

for _key in [k for k in st.session_state if str(k).startswith("train_")]:
    st.session_state[_key] = st.session_state[_key]

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/data.py", title="Data", icon=":material/database:"),
    st.Page("views/train.py", title="Train", icon=":material/model_training:"),
    st.Page("views/results.py", title="Results", icon=":material/insights:"),
    st.Page("views/predict.py", title="Predict", icon=":material/online_prediction:"),
]
navigation = st.navigation(pages)
sidebar_status()
navigation.run()
