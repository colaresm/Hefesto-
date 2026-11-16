from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class BuildContext:
    random_state: int = 42
    n_jobs: int = -1
    n_samples: int = 100
    n_features: int = 1


SearchSpace = Callable[[optuna.Trial, BuildContext], dict[str, Any]]
Builder = Callable[[dict[str, Any], BuildContext], RegressorMixin]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    family: str
    description: str
    builder: Builder
    search_space: SearchSpace | None = None
    requires: tuple[str, ...] = field(default_factory=tuple)
    max_recommended_samples: int | None = None
    task: str = "regression"
    binary_only: bool = False

    @property
    def available(self) -> bool:
        return all(importlib.util.find_spec(pkg) is not None for pkg in self.requires)

    @property
    def tunable(self) -> bool:
        return self.search_space is not None

    def sample_params(self, trial: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
        return self.search_space(trial, ctx) if self.search_space else {}

    def build(self, params: dict[str, Any], ctx: BuildContext) -> RegressorMixin:
        return self.builder(dict(params), ctx)


def _hidden_layers(params: dict[str, Any]) -> tuple[int, ...]:
    n_layers = int(params.pop("n_layers", 1))
    layers = tuple(int(params.pop(f"units_l{i}", 64)) for i in range(n_layers))
    for i in range(n_layers, 4):
        params.pop(f"units_l{i}", None)
    return layers


def _build_mlp(p: dict[str, Any], ctx: BuildContext, cls=MLPRegressor):
    layers = _hidden_layers(p)
    solver = p.pop("solver", "adam")
    common = dict(
        hidden_layer_sizes=layers,
        solver=solver,
        max_iter=3000,
        random_state=ctx.random_state,
        **p,
    )
    if solver == "adam":
        common.update(early_stopping=ctx.n_samples >= 100, n_iter_no_change=30, tol=1e-6)
    else:
        common.pop("learning_rate_init", None)
        common.pop("batch_size", None)
    return cls(**common)


def _build_mlp_classifier(p: dict[str, Any], ctx: BuildContext) -> MLPClassifier:
    return _build_mlp(p, ctx, cls=MLPClassifier)


_LOGISTIC_PENALTY_DEPRECATED = (
    inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"
)


def _build_logistic(p: dict[str, Any], ctx: BuildContext) -> LogisticRegression:
    l1_ratio = float(p.pop("l1_ratio", 0.0))
    p.pop("regularization", None)
    kwargs: dict[str, Any] = dict(
        C=p.pop("C", 1.0),
        class_weight=p.pop("class_weight", None),
        solver="saga" if l1_ratio > 0 else "lbfgs",
        max_iter=10_000,
        random_state=ctx.random_state,
    )
    if _LOGISTIC_PENALTY_DEPRECATED:
        kwargs["l1_ratio"] = l1_ratio
    elif l1_ratio > 0:
        kwargs.update(penalty="elasticnet", l1_ratio=l1_ratio)
    return LogisticRegression(**kwargs, **p)


def _build_xgboost_classifier(p: dict[str, Any], ctx: BuildContext):
    from xgboost import XGBClassifier

    return XGBClassifier(
        tree_method="hist",
        random_state=ctx.random_state,
        n_jobs=ctx.n_jobs,
        verbosity=0,
        **p,
    )


def _build_gpr(p: dict[str, Any], ctx: BuildContext) -> GaussianProcessRegressor:
    kernel_name = p.pop("kernel", "matern_2.5")
    base = {
        "rbf": RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
        "matern_1.5": Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-3, 1e3)),
        "matern_2.5": Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        "rational_quadratic": RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    }[kernel_name]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-3, (1e-8, 1e0))
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=p.pop("alpha", 1e-8),
        n_restarts_optimizer=p.pop("n_restarts_optimizer", 1),
        normalize_y=False,
        random_state=ctx.random_state,
    )


def _build_polynomial(p: dict[str, Any], ctx: BuildContext) -> Pipeline:
    return Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(
                    degree=p.get("degree", 2),
                    interaction_only=p.get("interaction_only", False),
                    include_bias=False,
                ),
            ),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=p.get("alpha", 1e-3))),
        ]
    )


def _build_adaboost(p: dict[str, Any], ctx: BuildContext) -> AdaBoostRegressor:
    depth = p.pop("base_max_depth", 3)
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=depth, random_state=ctx.random_state),
        random_state=ctx.random_state,
        **p,
    )


def _build_xgboost(p: dict[str, Any], ctx: BuildContext):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=ctx.random_state,
        n_jobs=ctx.n_jobs,
        verbosity=0,
        **p,
    )


def _build_lightgbm(p: dict[str, Any], ctx: BuildContext):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        random_state=ctx.random_state,
        n_jobs=ctx.n_jobs,
        subsample_freq=1,
        verbose=-1,
        **p,
    )


def _build_catboost(p: dict[str, Any], ctx: BuildContext):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        random_seed=ctx.random_state,
        thread_count=ctx.n_jobs,
        verbose=0,
        allow_writing_files=False,
        **p,
    )


def _space_ridge(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {"alpha": t.suggest_float("alpha", 1e-6, 1e3, log=True)}


def _space_lasso(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {"alpha": t.suggest_float("alpha", 1e-6, 1e1, log=True)}


def _space_elasticnet(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-6, 1e1, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.05, 0.95),
    }


def _space_polynomial(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    max_degree = 3 if ctx.n_features <= 12 else 2
    return {
        "degree": t.suggest_int("degree", 2, max_degree),
        "interaction_only": t.suggest_categorical("interaction_only", [False, True]),
        "alpha": t.suggest_float("alpha", 1e-6, 1e2, log=True),
    }


def _space_knn(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    upper = max(2, min(30, int(ctx.n_samples * 0.5)))
    return {
        "n_neighbors": t.suggest_int("n_neighbors", 1, upper),
        "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        "p": t.suggest_int("p", 1, 2),
    }


def _space_svr(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    kernel = t.suggest_categorical("kernel", ["rbf", "poly"])
    params: dict[str, Any] = {
        "kernel": kernel,
        "C": t.suggest_float("C", 1e-2, 1e3, log=True),
        "epsilon": t.suggest_float("epsilon", 1e-4, 0.5, log=True),
        "gamma": t.suggest_categorical("gamma", ["scale", "auto"]),
    }
    if kernel == "poly":
        params["degree"] = t.suggest_int("degree", 2, 4)
    return params


def _space_gpr(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "kernel": t.suggest_categorical(
            "kernel", ["rbf", "matern_1.5", "matern_2.5", "rational_quadratic"]
        ),
        "alpha": t.suggest_float("alpha", 1e-10, 1e-2, log=True),
        "n_restarts_optimizer": t.suggest_int("n_restarts_optimizer", 0, 3),
    }


def _space_tree(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "max_depth": t.suggest_int("max_depth", 2, 24),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 15),
        "max_features": t.suggest_categorical("max_features", [1.0, "sqrt", "log2"]),
    }


def _space_forest(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": t.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16, 24]),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 8),
        "max_features": t.suggest_categorical("max_features", [1.0, 0.7, 0.5, "sqrt"]),
        "bootstrap": t.suggest_categorical("bootstrap", [True, False]),
    }


def _space_gbr(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 1200, step=50),
        "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 15),
        "max_features": t.suggest_categorical("max_features", [1.0, 0.7, "sqrt"]),
        "loss": t.suggest_categorical("loss", ["squared_error", "huber"]),
    }


def _space_hgb(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "max_iter": t.suggest_int("max_iter", 100, 1500, step=50),
        "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 7, 63),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 30),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
    }


def _space_adaboost(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 600, step=25),
        "learning_rate": t.suggest_float("learning_rate", 1e-2, 1.0, log=True),
        "loss": t.suggest_categorical("loss", ["linear", "square", "exponential"]),
        "base_max_depth": t.suggest_int("base_max_depth", 2, 8),
    }


def _space_xgboost(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 1500, step=50),
        "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 10),
        "min_child_weight": t.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": t.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _space_lightgbm(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 1500, step=50),
        "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "num_leaves": t.suggest_int("num_leaves", 7, 127),
        "min_child_samples": t.suggest_int("min_child_samples", 2, 40),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _space_catboost(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    return {
        "iterations": t.suggest_int("iterations", 200, 1500, step=50),
        "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "depth": t.suggest_int("depth", 3, 10),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
    }


def _space_logistic(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    reg = t.suggest_categorical("regularization", ["l2", "l1", "elasticnet"])
    l1_ratio = {"l2": 0.0, "l1": 1.0}.get(reg)
    if l1_ratio is None:
        l1_ratio = t.suggest_float("l1_ratio", 0.05, 0.95)
    return {
        "regularization": reg,
        "l1_ratio": l1_ratio,
        "C": t.suggest_float("C", 1e-4, 1e3, log=True),
        "class_weight": t.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _space_forest_classifier(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    params = _space_forest(t, ctx)
    params["criterion"] = t.suggest_categorical("criterion", ["gini", "entropy"])
    params["class_weight"] = t.suggest_categorical(
        "class_weight", [None, "balanced", "balanced_subsample"]
    )
    return params


def _space_mlp(t: optuna.Trial, ctx: BuildContext) -> dict[str, Any]:
    n_layers = t.suggest_int("n_layers", 1, 3)
    params: dict[str, Any] = {"n_layers": n_layers}
    for i in range(n_layers):
        params[f"units_l{i}"] = t.suggest_int(f"units_l{i}", 16, 256, log=True)
    params["activation"] = t.suggest_categorical("activation", ["relu", "tanh"])
    params["alpha"] = t.suggest_float("alpha", 1e-7, 1e-1, log=True)
    params["solver"] = t.suggest_categorical("solver", ["lbfgs", "adam"])
    if params["solver"] == "adam":
        params["learning_rate_init"] = t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    return params


_REGISTRY: dict[str, dict[str, ModelSpec]] = {"regression": {}, "classification": {}}


def register(spec: ModelSpec) -> ModelSpec:
    registry = _REGISTRY[spec.task]
    if spec.key in registry:
        raise ValueError(f"{spec.task.capitalize()} model '{spec.key}' is already registered.")
    registry[spec.key] = spec
    return spec


def _simple(cls, **fixed) -> Builder:
    def build(p: dict[str, Any], ctx: BuildContext):
        kwargs = dict(fixed)
        if "random_state" in cls().get_params():
            kwargs["random_state"] = ctx.random_state
        if "n_jobs" in cls().get_params():
            kwargs["n_jobs"] = ctx.n_jobs
        kwargs.update(p)
        return cls(**kwargs)

    return build


for _spec in (
    ModelSpec(
        "linear",
        "Linear Regression",
        "Linear",
        "Ordinary least squares. Fast, interpretable baseline.",
        _simple(LinearRegression),
    ),
    ModelSpec(
        "ridge",
        "Ridge Regression",
        "Linear",
        "L2-regularized linear model; stable under collinear inputs.",
        _simple(Ridge),
        _space_ridge,
    ),
    ModelSpec(
        "lasso",
        "Lasso Regression",
        "Linear",
        "L1-regularized linear model; performs implicit feature selection.",
        _simple(Lasso, max_iter=50_000),
        _space_lasso,
    ),
    ModelSpec(
        "elasticnet",
        "Elastic Net",
        "Linear",
        "Blend of L1 and L2 penalties.",
        _simple(ElasticNet, max_iter=50_000),
        _space_elasticnet,
    ),
    ModelSpec(
        "polynomial",
        "Polynomial Regression (RSM)",
        "Linear",
        "Second/third-order response surface with ridge penalty - the classic DOE surrogate.",
        _build_polynomial,
        _space_polynomial,
    ),
    ModelSpec(
        "knn",
        "K-Nearest Neighbors",
        "Instance-based",
        "Distance-weighted interpolation between nearby design points.",
        _simple(KNeighborsRegressor),
        _space_knn,
    ),
    ModelSpec(
        "svr",
        "Support Vector Regression",
        "Kernel",
        "Epsilon-insensitive kernel regression; strong on small, smooth datasets.",
        _simple(SVR, cache_size=500),
        _space_svr,
    ),
    ModelSpec(
        "gpr",
        "Gaussian Process (Kriging)",
        "Kernel",
        "Bayesian kernel interpolator; the reference surrogate for deterministic simulations.",
        _build_gpr,
        _space_gpr,
        max_recommended_samples=3000,
    ),
    ModelSpec(
        "decision_tree",
        "Decision Tree",
        "Tree",
        "Single CART tree; interpretable but high variance.",
        _simple(DecisionTreeRegressor),
        _space_tree,
    ),
    ModelSpec(
        "random_forest",
        "Random Forest",
        "Tree ensemble",
        "Bagged decision trees; robust default with little tuning.",
        _simple(RandomForestRegressor),
        _space_forest,
    ),
    ModelSpec(
        "extra_trees",
        "Extra Trees",
        "Tree ensemble",
        "Extremely randomized trees; smoother than Random Forest on continuous inputs.",
        _simple(ExtraTreesRegressor),
        _space_forest,
    ),
    ModelSpec(
        "gradient_boosting",
        "Gradient Boosting",
        "Boosting",
        "scikit-learn gradient-boosted trees.",
        _simple(GradientBoostingRegressor),
        _space_gbr,
    ),
    ModelSpec(
        "hist_gradient_boosting",
        "Histogram Gradient Boosting",
        "Boosting",
        "Fast histogram-based boosting (LightGBM-style) from scikit-learn.",
        _simple(HistGradientBoostingRegressor),
        _space_hgb,
    ),
    ModelSpec(
        "adaboost",
        "AdaBoost",
        "Boosting",
        "Adaptive boosting of shallow regression trees.",
        _build_adaboost,
        _space_adaboost,
    ),
    ModelSpec(
        "xgboost",
        "XGBoost",
        "Boosting",
        "Regularized gradient boosting; state of the art on tabular data.",
        _build_xgboost,
        _space_xgboost,
        requires=("xgboost",),
    ),
    ModelSpec(
        "lightgbm",
        "LightGBM",
        "Boosting",
        "Leaf-wise gradient boosting; very fast on larger datasets.",
        _build_lightgbm,
        _space_lightgbm,
        requires=("lightgbm",),
    ),
    ModelSpec(
        "catboost",
        "CatBoost",
        "Boosting",
        "Ordered boosting with symmetric trees (optional dependency).",
        _build_catboost,
        _space_catboost,
        requires=("catboost",),
    ),
    ModelSpec(
        "mlp",
        "Multilayer Perceptron (MLP)",
        "Neural network",
        "Feed-forward neural network; tuned depth, width, activation and solver.",
        _build_mlp,
        _space_mlp,
    ),
):
    register(_spec)

for _spec in (
    ModelSpec(
        "logistic",
        "Logistic Regression",
        "Linear",
        "Regularized (L1 / L2 / elastic-net) logistic model; interpretable binary baseline.",
        _build_logistic,
        _space_logistic,
        task="classification",
        binary_only=True,
    ),
    ModelSpec(
        "random_forest",
        "Random Forest",
        "Tree ensemble",
        "Bagged decision trees with tuned class weighting; robust default.",
        _simple(RandomForestClassifier),
        _space_forest_classifier,
        task="classification",
    ),
    ModelSpec(
        "xgboost",
        "XGBoost",
        "Boosting",
        "Regularized gradient boosting; state of the art on tabular data.",
        _build_xgboost_classifier,
        _space_xgboost,
        requires=("xgboost",),
        task="classification",
    ),
    ModelSpec(
        "mlp",
        "Multilayer Perceptron (MLP)",
        "Neural network",
        "Feed-forward neural network; tuned depth, width, activation and solver.",
        _build_mlp_classifier,
        _space_mlp,
        task="classification",
    ),
):
    register(_spec)

DEFAULT_MODELS: tuple[str, ...] = ("polynomial", "gpr", "random_forest", "xgboost", "mlp")
DEFAULT_CLASSIFICATION_MODELS: tuple[str, ...] = ("logistic", "random_forest", "xgboost", "mlp")


def all_models(task: str = "regression") -> dict[str, ModelSpec]:
    return dict(_REGISTRY[task])


def available_models(
    task: str = "regression", n_classes: int | None = None
) -> dict[str, ModelSpec]:
    return {
        k: v
        for k, v in _REGISTRY[task].items()
        if v.available and not (v.binary_only and (n_classes or 2) > 2)
    }


def default_models(task: str = "regression", n_classes: int | None = None) -> list[str]:
    keys = DEFAULT_CLASSIFICATION_MODELS if task == "classification" else DEFAULT_MODELS
    available = available_models(task, n_classes)
    return [k for k in keys if k in available]


def get_model_spec(key: str, task: str = "regression") -> ModelSpec:
    registry = _REGISTRY[task]
    try:
        spec = registry[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{key}' for {task}. Available: {sorted(registry)}"
        ) from exc
    if not spec.available:
        raise ValueError(
            f"Model '{spec.name}' needs the optional package(s) {spec.requires}. "
            f"Install with: pip install {' '.join(spec.requires)}"
        )
    return spec


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_target_transformer(log_target: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if log_target:
        steps.append(("log", FunctionTransformer(np.log, inverse_func=np.exp, check_inverse=False)))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        y = np.asarray(y)
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.pipeline_ = clone(self.pipeline).fit(X, self.encoder_.transform(y))
        return self

    def predict(self, X):
        codes = np.asarray(self.pipeline_.predict(X)).astype(int).ravel()
        return self.encoder_.inverse_transform(codes)

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)


def inner_pipeline(model) -> Pipeline:
    if hasattr(model, "regressor_"):
        return model.regressor_
    return model.pipeline_


def build_pipeline(
    spec: ModelSpec,
    params: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
    ctx: BuildContext,
    log_target: bool = False,
):
    regressor = Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", spec.build(params, ctx)),
        ]
    )
    if spec.task == "classification":
        return LabelEncodedClassifier(regressor)
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=build_target_transformer(log_target),
        check_inverse=False,
    )
