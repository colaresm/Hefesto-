import streamlit as st
from ui_common import card, hero

from hefesto.models import available_models

hero(
    "Hefesto · Surrogate Model Forge",
    "Turn ANSYS simulation DOEs into fast, validated regression surrogates - with cross-validation, "
    "Bayesian hyperparameter optimization and publication-ready reports.",
    ["Semiconductor packaging", "ANSYS Workbench / PyMAPDL", "Regression", "Optuna HPO"],
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    card(
        "STEP 1",
        "Load simulation data",
        "Upload a CSV, Excel or Parquet table. ANSYS parameter exports are cleaned automatically: "
        "comment lines, design-point labels, units and failed points.",
    )
with c2:
    card(
        "STEP 2",
        "Pick models & protocol",
        f"Choose among {len(available_models())} regressors - from polynomial response surfaces and "
        "Kriging to XGBoost and neural networks - and set the validation protocol.",
    )
with c3:
    card(
        "STEP 3",
        "Forge",
        "Every model is tuned with Optuna (TPE + median pruning) on identical K-fold splits, "
        "then refit and scored on an untouched hold-out set.",
    )
with c4:
    card(
        "STEP 4",
        "Export & predict",
        "Download the best model (.joblib), the importance + actual-vs-predicted report and all "
        "tables, or predict new designs with extrapolation warnings.",
    )

st.write("")
left, right = st.columns([1.3, 1])
with left:
    st.subheader("Typical use cases")
    st.markdown(
        "- **BGA / CSP solder-joint fatigue** - plastic work density or Darveaux characteristic life\n"
        "- **Package and strip warpage** - CTE mismatch, mold compound, substrate stack-up\n"
        "- **Interface delamination** - energy release rate, peeling / shear stress at EMC-substrate\n"
        "- **Die cracking and dicing margin**, underfill voids, thermal resistance, and any other "
        "scalar response coming out of an ANSYS Mechanical / Workbench / PyMAPDL DOE"
    )
with right:
    st.subheader("Get started")
    st.page_link("views/data.py", label="Load a dataset", icon=":material/database:")
    st.page_link("views/train.py", label="Train surrogates", icon=":material/model_training:")
    st.page_link(
        "views/predict.py", label="Predict with a saved model", icon=":material/online_prediction:"
    )
    st.caption(
        "New here? The Data page includes a synthetic BGA fatigue DOE to try the full workflow."
    )
