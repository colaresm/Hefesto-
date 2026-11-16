import json

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from ui_common import (
    BLUE,
    EMBER,
    GRAY,
    ORANGE,
    STATE_ARTIFACTS,
    STATE_DATASET,
    chart_config,
    require_result,
)

from hefesto.artifacts import make_report_figure
from hefesto.metrics import get_metric

result = require_result()
artifacts = st.session_state.get(STATE_ARTIFACTS)
cfg = result.config
is_clf = result.is_classification
metric = get_metric(cfg.metric)
best = result.best
ds = st.session_state.get(STATE_DATASET)
units = ds.report.units if ds is not None else {}

st.title("Results")
task_label = (
    f"{'Binary' if len(result.classes) == 2 else 'Multiclass'} classification "
    f"({len(result.classes)} classes)"
    if is_clf
    else "Regression"
)
st.caption(
    f"{task_label} · Target **{cfg.target}** · {len(result.y_train)} training / "
    f"{len(result.y_test) if result.has_test else 0} hold-out design points · "
    f"{cfg.cv_folds}-fold CV{f' × {cfg.cv_repeats}' if cfg.cv_repeats > 1 else ''} · "
    f"{cfg.n_trials} trials per model · run time {result.duration_s:.1f}s"
)

st.markdown(f"### :material/emoji_events: Best model: {best.name}")
k1, k2, k3, k4 = st.columns(4)
src = best.test_metrics if result.has_test else best.cv_metrics
where = "Hold-out" if result.has_test else "CV"
if is_clf:
    k1.metric(f"{where} accuracy", f"{src['accuracy']:.4f}")
    k2.metric(f"{where} F1 (macro)", f"{src['f1_macro']:.4f}")
    auc = src.get("roc_auc", float("nan"))
    k3.metric(f"{where} ROC AUC", "-" if np.isnan(auc) else f"{auc:.4f}")
else:
    k1.metric(f"{where} R²", f"{src['r2']:.4f}")
    k2.metric(f"{where} RMSE", f"{src['rmse']:.4g}")
    k3.metric(f"{where} MAE", f"{src['mae']:.4g}")
k4.metric(
    f"CV {metric.label} (± {best.cv_std:.3g} std)",
    f"{best.cv_mean:.4g}",
    help="Mean and standard deviation of the optimization metric across CV folds.",
)

tab_report, tab_board, tab_opt, tab_resid, tab_params, tab_files = st.tabs(
    [
        "Report",
        "Leaderboard",
        "Optimization",
        "Classes" if is_clf else "Residuals",
        "Hyperparameters",
        "Downloads",
    ]
)

with tab_report:
    if artifacts is not None and artifacts.figure_png.exists():
        st.image(str(artifacts.figure_png), width="stretch")
    else:
        fig = make_report_figure(result, units=units)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    if result.importance is not None and result.importance.note:
        st.caption(result.importance.note)
    ok_models = [r for r in result.results if r.ok and r.key != best.key]
    if ok_models:
        with st.expander(
            f"Inspect another model ({'confusion matrix' if is_clf else 'actual vs. predicted'})",
            icon=":material/compare:",
        ):
            other = st.selectbox(
                "Model", [r.key for r in ok_models], format_func=lambda k: result.get(k).name
            )
            fig = make_report_figure(result, key=other, units=units)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption("Feature importance is computed for the best model only.")

with tab_board:
    board = result.leaderboard()
    mean_col, std_col = f"CV {metric.label} (mean)", f"CV {metric.label} (std)"
    chart_df = board[board["Status"] == "ok"][["Model", "key", mean_col, std_col]].rename(
        columns={mean_col: "mean", std_col: "std"}
    )
    chart_df["low"], chart_df["high"] = (
        chart_df["mean"] - chart_df["std"],
        chart_df["mean"] + chart_df["std"],
    )
    chart_df["Selected"] = np.where(chart_df["key"] == best.key, "Best model", "Other models")
    order = chart_df["Model"].tolist()
    x_title = f"Mean CV {metric.label} (error bar = ±1 std across folds)"
    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("mean:Q", title=x_title),
            color=alt.Color(
                "Selected:N",
                scale=alt.Scale(domain=["Best model", "Other models"], range=[EMBER, BLUE]),
                title=None,
            ),
            tooltip=[
                "Model",
                alt.Tooltip("mean:Q", format=".5g", title="Mean"),
                alt.Tooltip("std:Q", format=".3g", title="Std"),
            ],
        )
    )
    errs = (
        alt.Chart(chart_df)
        .mark_errorbar(color=GRAY, thickness=1.2)
        .encode(y=alt.Y("Model:N", sort=order), x=alt.X("low:Q", title=x_title), x2="high:Q")
    )
    st.altair_chart(chart_config((bars + errs).properties(height=alt.Step(38))), width="stretch")
    display = board.drop(columns=["key"])
    float_cols = display.select_dtypes("number").columns.drop(["Rank"], errors="ignore")
    st.dataframe(
        display.style.format({c: "{:.5g}" for c in float_cols}, na_rep="-"),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Ranking by {'mean CV' if cfg.selection_basis == 'cv' else 'hold-out'} "
        f"{metric.label} ({'higher' if metric.greater_is_better else 'lower'} is better). "
        "CV metrics in the table are computed on pooled out-of-fold predictions."
    )

with tab_opt:
    tuned = [r for r in result.results if not r.trials.empty]
    if not tuned:
        st.info("No optimization history available.")
    else:
        key = st.selectbox(
            "Model",
            [r.key for r in tuned],
            format_func=lambda k: result.get(k).name,
            index=[r.key for r in tuned].index(best.key)
            if best.key in [r.key for r in tuned]
            else 0,
        )
        trials = result.get(key).trials.copy()
        trials["state"] = trials["state"].astype(str)
        for c in trials.select_dtypes("object").columns:
            trials[c] = trials[c].astype(str)
        done = trials[trials["state"] == "COMPLETE"].copy()
        if done.empty:
            st.info("No completed trial for this model.")
        else:
            agg = np.maximum.accumulate if metric.greater_is_better else np.minimum.accumulate
            done["best_so_far"] = agg(done["value"].to_numpy())
            done["trial"] = done["number"] + 1
            base = alt.Chart(done).encode(
                x=alt.X("trial:Q", title="Trial", axis=alt.Axis(format="d", tickMinStep=1))
            )
            pts = base.mark_circle(size=70, color=BLUE, opacity=0.7).encode(
                y=alt.Y("value:Q", title=f"Mean CV {metric.label}", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("trial:Q", title="Trial"),
                    alt.Tooltip("value:Q", format=".5g"),
                ],
            )
            line = base.mark_line(color=ORANGE, strokeWidth=2, interpolate="step-after").encode(
                y="best_so_far:Q"
            )
            st.altair_chart(chart_config((pts + line).properties(height=340)), width="stretch")
            st.caption(
                "Dots: completed trials · orange step line: best value so far. "
                f"{(trials['state'] == 'PRUNED').sum()} trial(s) pruned, "
                f"{(trials['state'] == 'FAIL').sum()} failed."
            )
            st.dataframe(trials, hide_index=True, width="stretch", height=300)

with tab_resid:
    if is_clf:
        split_options = (["Hold-out"] if result.has_test else []) + ["CV out-of-fold"]
        split = st.radio("Split", split_options, horizontal=True)
        if split == "Hold-out":
            y_true, y_pred = result.y_test.to_numpy(), best.test_pred
        else:
            y_true, y_pred = result.y_train.to_numpy(), best.oof_pred
        classes = [str(c) for c in result.classes]
        cm = pd.crosstab(
            pd.Categorical(y_true.astype(str), categories=classes),
            pd.Categorical(np.asarray(y_pred).astype(str), categories=classes),
            dropna=False,
        )
        long = cm.rename_axis(index="Actual", columns="Predicted").stack().rename("Count")
        long = long.reset_index()
        long["Share of actual class"] = long["Count"] / long.groupby("Actual", observed=False)[
            "Count"
        ].transform("sum").clip(lower=1)
        base = alt.Chart(long).encode(
            x=alt.X("Predicted:N", sort=classes, title="Predicted class"),
            y=alt.Y("Actual:N", sort=classes, title="Actual class"),
        )
        heat = base.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "Share of actual class:Q",
                scale=alt.Scale(domain=[0, 1], range=["#f0efec", BLUE]),
                legend=alt.Legend(format="%"),
            ),
            tooltip=[
                "Actual:N",
                "Predicted:N",
                "Count:Q",
                alt.Tooltip("Share of actual class:Q", format=".1%"),
            ],
        )
        text = base.mark_text(fontSize=13).encode(
            text="Count:Q",
            color=alt.condition(
                "datum['Share of actual class'] > 0.55", alt.value("white"), alt.value("#1f1e1c")
            ),
        )
        size = max(260, 70 * len(classes))
        st.altair_chart(
            chart_config((heat + text).properties(height=size, width=size)), width="content"
        )
        st.caption(
            "Rows: actual class · columns: predicted class · color: share of the actual class. "
            "The diagonal holds the correct predictions."
        )
        from sklearn.metrics import classification_report

        report = pd.DataFrame(
            classification_report(
                y_true.astype(str),
                np.asarray(y_pred).astype(str),
                labels=classes,
                output_dict=True,
                zero_division=0,
            )
        ).T
        report["support"] = report["support"].astype(int)
        st.markdown("##### Per-class metrics")
        st.dataframe(
            report.style.format({c: "{:.4f}" for c in ("precision", "recall", "f1-score")}),
            width="stretch",
        )
    else:
        parts = [
            pd.DataFrame(
                {
                    "Split": "Cross-validation (out-of-fold)",
                    "Predicted": best.oof_pred,
                    "Actual": result.y_train.to_numpy(),
                }
            )
        ]
        if result.has_test:
            parts.append(
                pd.DataFrame(
                    {
                        "Split": "Hold-out test",
                        "Predicted": best.test_pred,
                        "Actual": result.y_test.to_numpy(),
                    }
                )
            )
        res = pd.concat(parts, ignore_index=True).dropna()
        res["Residual"] = res["Predicted"] - res["Actual"]
        scatter = (
            alt.Chart(res)
            .mark_point(filled=True, size=55, opacity=0.8)
            .encode(
                x=alt.X(
                    "Predicted:Q", title=f"Predicted {cfg.target}", scale=alt.Scale(zero=False)
                ),
                y=alt.Y("Residual:Q", title="Residual (predicted − actual)"),
                color=alt.Color("Split:N", scale=alt.Scale(range=[BLUE, ORANGE]), title=None),
                shape=alt.Shape(
                    "Split:N", scale=alt.Scale(range=["circle", "diamond"]), title=None
                ),
                tooltip=[alt.Tooltip(c, format=".5g") for c in ("Actual", "Predicted", "Residual")]
                + ["Split"],
            )
        )
        zero = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color=GRAY, strokeDash=[5, 4])
            .encode(y="y:Q")
        )
        st.altair_chart(chart_config((scatter + zero).properties(height=380)), width="stretch")
        st.caption(
            "Look for structure: funnels suggest a log target, curves suggest missing interactions."
        )

with tab_params:
    for r in result.results:
        label = f"{'★ ' if r.key == best.key else ''}{r.name}"
        with st.expander(label, expanded=r.key == best.key):
            if r.ok:
                st.json(r.best_params or {"note": "no tunable hyperparameters"})
            else:
                st.error(r.error)

with tab_files:
    if artifacts is None:
        st.info("Artifacts were not saved for this run.")
    else:
        st.markdown(f"Run folder: `{artifacts.run_dir}`")
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Best model (.joblib)",
            artifacts.model_path.read_bytes(),
            file_name=artifacts.model_path.name,
            icon=":material/deployed_code:",
            type="primary",
            width="stretch",
        )
        d2.download_button(
            "Report figure (.png)",
            artifacts.figure_png.read_bytes(),
            file_name=artifacts.figure_png.name,
            mime="image/png",
            icon=":material/image:",
            width="stretch",
        )
        d3.download_button(
            "Report with metrics table (.pdf)",
            artifacts.figure_pdf.read_bytes(),
            file_name=artifacts.figure_pdf.name,
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
            width="stretch",
        )
        d4, d5, d6 = st.columns(3)
        d4.download_button(
            "Leaderboard (.csv)",
            artifacts.leaderboard_csv.read_bytes(),
            file_name="leaderboard.csv",
            mime="text/csv",
            icon=":material/table:",
            width="stretch",
        )
        d5.download_button(
            "Predictions (.csv)",
            artifacts.predictions_csv.read_bytes(),
            file_name="predictions.csv",
            mime="text/csv",
            icon=":material/table:",
            width="stretch",
        )
        d6.download_button(
            "Complete run (.zip)",
            artifacts.zip_bytes(),
            file_name=f"{artifacts.run_dir.name}.zip",
            mime="application/zip",
            icon=":material/folder_zip:",
            width="stretch",
        )
        panels = {
            name: path
            for name, path in getattr(artifacts, "panel_pngs", {}).items()
            if path.exists()
        }
        if panels:
            st.markdown("##### Individual figures (.png)")
            panel_cols = st.columns(len(panels))
            for panel_col, (name, path) in zip(panel_cols, panels.items(), strict=True):
                panel_col.image(str(path), width="stretch")
                panel_col.download_button(
                    f"{name.replace('_', ' ').capitalize().replace('vs ', 'vs. ')} (.png)",
                    path.read_bytes(),
                    file_name=path.name,
                    mime="image/png",
                    icon=":material/image:",
                    width="stretch",
                    key=f"dl_{name}",
                )
        st.markdown("##### Using the model in Python")
        st.code(
            "import joblib\nimport pandas as pd\n\n"
            f"model = joblib.load('{artifacts.model_path.name}')\n"
            "features = model.hefesto_metadata_['features']\n"
            "X_new = pd.read_csv('new_designs.csv')[features]\n"
            + (
                "labels = model.predict(X_new)        # original class labels\n"
                "proba = model.predict_proba(X_new)   # columns follow model.classes_"
                if is_clf
                else "y_pred = model.predict(X_new)  # already in original target units"
            ),
            language="python",
        )
        with st.expander("Model card (model_card.json)", icon=":material/badge:"):
            st.json(json.loads(artifacts.card_path.read_text(encoding="utf-8")), expanded=1)
