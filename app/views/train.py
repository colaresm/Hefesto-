import pandas as pd
import streamlit as st
from ui_common import (
    OUTPUT_DIR,
    STATE_ARTIFACTS,
    STATE_DATASET_NAME,
    STATE_OUTLIERS,
    STATE_RESULT,
    STATE_SUGGESTED_FEATURES,
    STATE_SUGGESTED_TARGET,
    require_dataset,
)

from hefesto.artifacts import save_run
from hefesto.config import TrainingConfig
from hefesto.data import (
    MAX_CLASSES,
    class_labels,
    is_class_label_column,
    numeric_columns,
    suggest_target,
    suggest_task,
)
from hefesto.metrics import DEFAULT_METRIC, METRICS, metrics_for
from hefesto.models import all_models, available_models, default_models
from hefesto.outliers import METHODS, detect_outliers
from hefesto.training import ProgressEvent, SurrogateTrainer

st.title("Train")
st.caption(
    "Configure the benchmark, then forge: every model is tuned with Optuna on identical "
    "cross-validation folds and scored on an untouched hold-out set. Works for regression "
    "(continuous responses) and binary / multiclass classification (labels)."
)

ds = require_dataset()
df = ds.df
num_cols = numeric_columns(df)
label_cols = [c for c in df.columns if is_class_label_column(df[c])]

DEFAULTS = {
    "train_cv_folds": 5,
    "train_cv_repeats": 1,
    "train_test_size": 0.2,
    "train_n_trials": 30,
    "train_selection": "cv",
    "train_log_target": False,
    "train_stratify": True,
    "train_importance": "permutation",
    "train_top_n": 20,
    "train_timeout": 0,
    "train_seed": 42,
    "train_save_all": False,
    "train_select_all": False,
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

st.subheader("1 · Task, target and inputs")
if "train_task" not in st.session_state:
    first = st.session_state.get(STATE_SUGGESTED_TARGET) or suggest_target(df)
    st.session_state["train_task"] = (
        suggest_task(df[first]) if first in df.columns else "regression"
    )
task = st.radio(
    "Task",
    ["regression", "classification"],
    horizontal=True,
    key="train_task",
    format_func={
        "regression": "Regression (continuous response)",
        "classification": "Classification (binary / multiclass labels)",
    }.get,
)
is_clf = task == "classification"
target_options = label_cols if is_clf else num_cols
if not target_options:
    st.error(
        f"No column can be a classification target (text, or integers with 2-{MAX_CLASSES} "
        "distinct values)."
        if is_clf
        else "The dataset has no numeric column to use as a regression target."
    )
    st.stop()
default_target = st.session_state.get(STATE_SUGGESTED_TARGET) or suggest_target(df)
if st.session_state.get("train_target") not in target_options:
    st.session_state["train_target"] = (
        default_target if default_target in target_options else target_options[-1]
    )
target = st.selectbox(
    "Target (class label)" if is_clf else "Target (response)",
    target_options,
    key="train_target",
    help="The label the classifier will predict."
    if is_clf
    else "The simulation output the surrogate will predict.",
)
candidates = [c for c in df.columns if c != target]
suggested = st.session_state.get(STATE_SUGGESTED_FEATURES)
default_features = [c for c in (suggested or candidates) if c in candidates]
if (
    st.session_state.get("_features_for_target") != target
    or "train_features" not in st.session_state
):
    st.session_state["train_features"] = default_features
    st.session_state["_features_for_target"] = target
features = st.multiselect(
    "Input features (design variables)",
    candidates,
    key="train_features",
    help="Keep only design inputs. Other simulation outputs would leak information.",
)

y = df[target]
n_classes = int(y.nunique()) if is_clf else None
positive = not is_clf and bool((y.dropna() > 0).all())
if is_clf:
    counts = class_labels(y.dropna()).value_counts()
    st.caption(
        f"{'Binary' if n_classes == 2 else 'Multiclass'} target with {n_classes} classes: "
        + " · ".join(f"**{k}** ({v})" for k, v in counts.items())
    )
    if counts.min() < 2:
        st.error(
            "Every class needs at least 2 rows: "
            + ", ".join(f"'{k}' ({v})" for k, v in counts[counts < 2].items()),
            icon=":material/error:",
        )
    elif counts.max() / counts.min() > 4:
        st.info(
            "Imbalanced classes: prefer F1 (macro), balanced accuracy or ROC AUC over accuracy.",
            icon=":material/balance:",
        )
if features and not is_clf:
    num_feats = [f for f in features if f in num_cols]
    if num_feats:
        corr = df[num_feats].corrwith(y).abs().sort_values(ascending=False)
        leaks = corr[corr > 0.97]
        if not leaks.empty:
            st.warning(
                "Possible target leakage - these inputs are almost perfectly correlated with the "
                f"target: {', '.join(f'**{k}** (|r| = {v:.3f})' for k, v in leaks.items())}. "
                "Are they simulation outputs rather than design inputs?",
                icon=":material/warning:",
            )

removal = st.session_state.get(STATE_OUTLIERS)
if removal:
    st.info(
        f"Training on {len(df)} design points after removing {removal['rows_flagged']} outlier "
        f"row(s) ({METHODS[removal['method']]['label']}, threshold {removal['threshold']:g}).",
        icon=":material/filter_alt:",
    )
else:
    o_method = st.session_state.get("train_outlier_method", "iqr")
    o_threshold = st.session_state.get("train_outlier_threshold", METHODS[o_method]["threshold"])
    screened = features if is_clf else [target, *features]
    check = detect_outliers(df, screened, o_method, o_threshold)
    if check.has_outliers:
        hits = check.table().query("Outliers > 0")
        where = "selected inputs" if is_clf else "target or the selected inputs"
        st.warning(
            f"**{check.n_outlier_rows}** design point(s) have outliers in the {where} ({METHODS[o_method]['label']}, threshold {o_threshold:g}): "
            + ", ".join(f"{r.Column} ({r.Outliers})" for r in hits.head(6).itertuples())
            + (" ..." if len(hits) > 6 else "")
            + ". Review and optionally remove them on the Data page.",
            icon=":material/warning:",
        )
        st.page_link("views/data.py", label="Review outliers", icon=":material/filter_alt:")

st.subheader("2 · Models")
available = available_models(task, n_classes)
unavailable = [
    s.name + (" (binary only)" if s.binary_only else "")
    for k, s in all_models(task).items()
    if k not in available
]


def _label(key: str) -> str:
    spec = available[key]
    return f"{spec.name}  ·  {spec.family}"


def _toggle_all() -> None:
    st.session_state["train_models"] = (
        list(available) if st.session_state["train_select_all"] else default_models(task, n_classes)
    )


model_scope = (task, bool(n_classes and n_classes > 2))
if st.session_state.get("train_models_scope") != model_scope:
    st.session_state["train_models"] = default_models(task, n_classes)
    st.session_state["train_models_scope"] = model_scope
st.session_state["train_models"] = [m for m in st.session_state["train_models"] if m in available]
st.toggle("Select every available model", key="train_select_all", on_change=_toggle_all)
models = st.multiselect(
    "Classification models" if is_clf else "Regression models",
    list(available),
    format_func=_label,
    key="train_models",
)
if is_clf and n_classes and n_classes > 2:
    st.caption("Logistic Regression is offered for binary targets only.")
with st.expander("Model catalogue", icon=":material/menu_book:"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": s.name,
                    "Family": s.family,
                    "Tunable": "yes" if s.tunable else "no",
                    "Description": s.description,
                }
                for s in available.values()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if unavailable:
        st.caption(
            f"Not installed (optional): {', '.join(unavailable)}. "
            "Install the package and restart to enable."
        )

st.subheader("3 · Validation and optimization")
p1, p2, p3 = st.columns(3)
with p1:
    cv_folds = st.slider("Cross-validation folds (K)", 2, 10, key="train_cv_folds")
    cv_repeats = st.number_input(
        "K-fold repetitions",
        1,
        10,
        key="train_cv_repeats",
        help="> 1 uses repeated K-fold for a lower-variance estimate.",
    )
    test_size = st.slider(
        "Hold-out test fraction",
        0.0,
        0.4,
        step=0.05,
        key="train_test_size",
        help="Untouched until the very end. Set 0 to rely on CV only.",
    )
with p2:
    n_trials = st.number_input(
        "Optuna trials per model",
        1,
        1000,
        step=5,
        key="train_n_trials",
        help="Hyperparameter-free models (Linear Regression) run once.",
    )
    task_metrics = metrics_for(task)
    if st.session_state.get("train_metric") not in task_metrics:
        st.session_state["train_metric"] = DEFAULT_METRIC[task]
    metric = st.selectbox(
        "Optimization metric",
        list(task_metrics),
        format_func=lambda k: METRICS[k].label,
        key="train_metric",
        help="\n".join(f"- **{m.label}**: {m.description}" for m in task_metrics.values()),
    )
    selection_basis = st.radio(
        "Select the best model by",
        ["cv", "test"],
        horizontal=True,
        key="train_selection",
        format_func={"cv": "CV score (recommended)", "test": "Hold-out score"}.get,
        disabled=test_size == 0,
    )
with p3:
    log_target = st.toggle(
        "Log-transform target",
        key="train_log_target",
        disabled=not positive,
        help="Recommended for responses spanning orders of magnitude (fatigue life, cycles to "
        "failure). Requires a strictly positive target."
        if positive
        else "Regression only."
        if is_clf
        else "Disabled: the target has values <= 0.",
    )
    stratify = st.toggle(
        "Stratify hold-out on classes" if is_clf else "Stratify hold-out on target quantiles",
        key="train_stratify",
        help="Keeps the class proportions in the test set (CV folds are always stratified)."
        if is_clf
        else "Ensures the test set covers the full response range.",
    )
    importance_method = st.radio(
        "Feature importance",
        ["permutation", "native"],
        horizontal=True,
        key="train_importance",
        format_func={"permutation": "Permutation", "native": "Native (trees / linear)"}.get,
    )

with st.expander("Advanced", icon=":material/tune:"):
    a1, a2, a3, a4 = st.columns(4)
    top_n = a1.number_input("Features in importance chart", 1, 100, key="train_top_n")
    timeout = a2.number_input(
        "Time budget per model (s, 0 = none)", 0, 86_400, step=60, key="train_timeout"
    )
    seed = a3.number_input("Random seed", 0, 2**31 - 1, key="train_seed")
    save_all = a4.toggle("Save every tuned model", key="train_save_all")

st.divider()
n_valid = int(y.notna().sum())
n_train = int(round(n_valid * (1 - test_size)))
st.caption(
    f"{n_valid} design points with a valid target → ~{n_train} for training/CV and "
    f"~{n_valid - n_train} for hold-out · {len(models)} model(s) × {n_trials} trial(s) × "
    f"{cv_folds * cv_repeats} folds."
)

if st.button(
    "Forge surrogates",
    type="primary",
    icon=":material/local_fire_department:",
    disabled=not (features and models),
):
    try:
        config = TrainingConfig(
            target=target,
            features=features,
            models=models,
            task=task,
            n_trials=int(n_trials),
            cv_folds=int(cv_folds),
            cv_repeats=int(cv_repeats),
            test_size=float(test_size),
            metric=metric,
            selection_basis=selection_basis if test_size > 0 else "cv",
            log_target=bool(log_target and positive),
            stratify_split=stratify,
            importance_method=importance_method,
            top_n_features=int(top_n),
            timeout_per_model=float(timeout) or None,
            random_state=int(seed),
            save_all_models=save_all,
            output_dir=OUTPUT_DIR,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    metric_spec = METRICS[metric]
    status = st.status("Forging surrogates...", expanded=True)
    with status:
        progress = st.progress(0.0, text="Preparing data")
        live = st.empty()
        board_slot = st.empty()
    finished: list[dict] = []

    def on_progress(ev: ProgressEvent) -> None:
        if ev.stage == "model_start":
            live.markdown(f"**[{ev.model_index + 1}/{ev.n_models}] {ev.model_name}** - tuning")
        elif ev.stage == "trial":
            best = "-" if ev.best_value is None else f"{ev.best_value:.5g}"
            progress.progress(
                ev.fraction,
                text=f"{ev.model_name}: trial {ev.trial}/{ev.n_trials} · "
                f"best CV {metric_spec.label} = {best}",
            )
        elif ev.stage == "model_end":
            finished.append(
                {
                    "Model": ev.model_name,
                    f"CV {metric_spec.label}": ev.best_value,
                    "Status": "failed: " + ev.message if ev.message else "ok",
                }
            )
            board = pd.DataFrame(finished).sort_values(
                f"CV {metric_spec.label}", ascending=not metric_spec.greater_is_better
            )
            board_slot.dataframe(board, hide_index=True, width="stretch")
        elif ev.stage == "importance":
            progress.progress(0.99, text=f"Computing feature importance for {ev.model_name}")
            live.markdown(f"Best model: **{ev.model_name}** - computing feature importance")

    try:
        trainer = SurrogateTrainer(config, progress_callback=on_progress)
        result = trainer.fit(df, dataset_name=st.session_state.get(STATE_DATASET_NAME, "dataset"))
        artifacts = save_run(
            result,
            units=ds.report.units,
            data_preprocessing={"outliers_removed": removal} if removal else None,
        )
    except Exception as exc:
        status.update(label="Training failed", state="error")
        st.error(str(exc), icon=":material/error:")
        st.stop()

    progress.progress(1.0, text="Done")
    status.update(
        label=f"Done in {result.duration_s:.1f}s · best model: {result.best.name}",
        state="complete",
        expanded=False,
    )
    st.session_state[STATE_RESULT] = result
    st.session_state[STATE_ARTIFACTS] = artifacts
    st.success(
        f"**{result.best.name}** won. Artifacts saved to `{artifacts.run_dir}`.",
        icon=":material/emoji_events:",
    )
    st.page_link("views/results.py", label="Open results", icon=":material/insights:")
