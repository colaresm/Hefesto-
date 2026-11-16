from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from hefesto import __version__
from hefesto.config import DEFAULT_OUTPUT_DIR, TrainingConfig
from hefesto.logging_utils import configure_logging, get_logger
from hefesto.metrics import METRICS
from hefesto.models import all_models, default_models

logger = get_logger("hefesto.cli")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hefesto",
        description="Hefesto - surrogate-model forge for ANSYS semiconductor packaging simulations.",
    )
    parser.add_argument("--version", action="version", version=f"hefesto {__version__}")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING (default: INFO)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="List registered regression and classification models.")

    t = sub.add_parser(
        "train", help="Tune, cross-validate and select regression or classification surrogates."
    )
    t.add_argument(
        "--data", required=True, type=Path, help="CSV/TXT/XLSX/Parquet simulation table."
    )
    t.add_argument("--target", required=True, help="Response column to model.")
    t.add_argument(
        "--task",
        choices=["auto", "regression", "classification"],
        default="auto",
        help="'auto' picks classification for text or few-valued integer targets.",
    )
    t.add_argument(
        "--features", nargs="+", help="Input columns (default: every other numeric column)."
    )
    t.add_argument("--exclude", nargs="+", default=[], help="Columns to exclude from the inputs.")
    t.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys, or 'all' (default: the task's default benchmark).",
    )
    t.add_argument("--trials", type=int, default=30, help="Optuna trials per model.")
    t.add_argument("--folds", type=int, default=5, help="K in K-fold cross-validation.")
    t.add_argument("--repeats", type=int, default=1, help="Repetitions of K-fold.")
    t.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction (0 disables).")
    t.add_argument(
        "--metric",
        choices=sorted(METRICS),
        default=None,
        help="Default: rmse (regression) or f1_macro (classification).",
    )
    t.add_argument("--select-on", choices=["cv", "test"], default="cv")
    t.add_argument("--log-target", action="store_true", help="Model log(y) (e.g. fatigue life).")
    t.add_argument("--no-stratify", action="store_true", help="Plain random hold-out split.")
    t.add_argument(
        "--remove-outliers",
        choices=["iqr", "zscore"],
        default=None,
        help="Drop rows with outliers in the numeric inputs (and a regression target) first.",
    )
    t.add_argument(
        "--outlier-threshold",
        type=float,
        default=None,
        help="IQR multiplier (default 1.5) or robust |z| limit (default 3.5).",
    )
    t.add_argument("--importance", choices=["permutation", "native"], default="permutation")
    t.add_argument("--top-n", type=int, default=20, help="Features shown in the importance chart.")
    t.add_argument("--timeout", type=float, default=None, help="Seconds per model (optional).")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--n-jobs", type=int, default=-1)
    t.add_argument("--save-all", action="store_true", help="Also save every tuned model.")
    t.add_argument(
        "--keep-ansys-ids",
        action="store_true",
        help="Keep 'P# - ' prefixes in ANSYS parameter names.",
    )
    t.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root directory.")

    p = sub.add_parser("predict", help="Predict with a saved surrogate.")
    p.add_argument("--model", required=True, type=Path, help="Path to a .joblib saved by Hefesto.")
    p.add_argument("--data", required=True, type=Path, help="Table with the model's input columns.")
    p.add_argument("--output", type=Path, default=Path("predictions.csv"))
    p.add_argument("--keep-ansys-ids", action="store_true")
    return parser


def _cmd_list_models() -> int:
    for task in ("regression", "classification"):
        print(f"\n{task.upper()}\n{'key':<24}{'name':<32}{'family':<16}available")
        for key, spec in all_models(task).items():
            note = " (binary only)" if spec.binary_only else ""
            print(
                f"{key:<24}{spec.name:<32}{spec.family:<16}"
                f"{'yes' if spec.available else 'no'}{note}"
            )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from hefesto.artifacts import save_run
    from hefesto.data import load_dataset, numeric_columns, suggest_task
    from hefesto.models import available_models
    from hefesto.training import SurrogateTrainer

    loaded = load_dataset(args.data, simplify_names=not args.keep_ansys_ids)
    for line in loaded.report.summary():
        logger.info(line)
    df = loaded.df
    if args.target not in df.columns:
        logger.error("Target '%s' not found. Columns: %s", args.target, list(df.columns))
        return 2
    features = args.features or [
        c for c in numeric_columns(df) if c != args.target and c not in args.exclude
    ]
    task = suggest_task(df[args.target]) if args.task == "auto" else args.task
    n_classes = int(df[args.target].nunique()) if task == "classification" else None
    if args.models in (None, ["all"]):
        models = (
            list(available_models(task, n_classes))
            if args.models
            else default_models(task, n_classes)
        )
    else:
        models = args.models
    logger.info("Task: %s%s", task, f" ({n_classes} classes)" if n_classes else "")

    preprocessing = None
    if args.remove_outliers:
        from hefesto.outliers import detect_outliers, remove_outliers

        screened = [f for f in features if f in numeric_columns(df)]
        if task == "regression":
            screened.insert(0, args.target)
        report = detect_outliers(df, screened, args.remove_outliers, args.outlier_threshold)
        df = remove_outliers(df, report)
        preprocessing = {"outliers_removed": report.to_dict()}
        logger.info(
            "Removed %d outlier row(s) of %d (%s).",
            report.n_outlier_rows,
            report.n_rows,
            args.remove_outliers,
        )

    config = TrainingConfig(
        target=args.target,
        features=features,
        models=models,
        task=task,
        n_trials=args.trials,
        cv_folds=args.folds,
        cv_repeats=args.repeats,
        test_size=args.test_size,
        metric=args.metric,
        selection_basis=args.select_on,
        log_target=args.log_target,
        stratify_split=not args.no_stratify,
        importance_method=args.importance,
        top_n_features=args.top_n,
        timeout_per_model=args.timeout,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        save_all_models=args.save_all,
        output_dir=args.output,
    )

    def progress(ev):
        if ev.stage == "model_start":
            logger.info("[%d/%d] Tuning %s ...", ev.model_index + 1, ev.n_models, ev.model_name)

    result = SurrogateTrainer(config, progress_callback=progress).fit(
        df, dataset_name=args.data.name
    )
    artifacts = save_run(result, units=loaded.report.units, data_preprocessing=preprocessing)

    board = result.leaderboard()
    metric_label = METRICS[config.metric].label
    headline = "Accuracy" if result.is_classification else "R²"
    cols = ["Rank", "Model", f"CV {metric_label} (mean)", f"CV {metric_label} (std)"]
    cols.append(f"CV {headline}")
    if result.has_test:
        cols += [f"Test {headline}", f"Test {metric_label}"]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n" + board[list(dict.fromkeys(cols))].to_string(index=False))
    print(f"\nBest model : {result.best.name}")
    print(f"Run folder : {artifacts.run_dir}")
    print(f"Model file : {artifacts.model_path}")
    print(f"Report     : {artifacts.figure_png}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from hefesto.data import load_dataset
    from hefesto.inference import load_model, predict

    loaded_model = load_model(args.model)
    data = load_dataset(args.data, simplify_names=not args.keep_ansys_ids, drop_constant=False).df
    out = predict(loaded_model, data)
    out.to_csv(args.output, index=False)
    n_ood = int(out.get("out_of_domain", pd.Series(dtype=bool)).sum())
    print(f"Wrote {len(out)} predictions to {args.output}")
    if n_ood:
        print(f"WARNING: {n_ood} row(s) lie outside the training domain (extrapolation).")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.command == "list-models":
            return _cmd_list_models()
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "predict":
            return _cmd_predict(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
