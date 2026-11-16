from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from hefesto.metrics import DEFAULT_METRIC, METRICS, TASKS

ImportanceMethod = Literal["permutation", "native"]
SelectionBasis = Literal["cv", "test"]
Task = Literal["regression", "classification"]

DEFAULT_OUTPUT_DIR = Path(os.getenv("HEFESTO_OUTPUT_DIR", "outputs"))


@dataclass
class TrainingConfig:
    target: str
    features: list[str]
    models: list[str]
    task: Task = "regression"
    n_trials: int = 30
    cv_folds: int = 5
    cv_repeats: int = 1
    test_size: float = 0.2
    metric: str | None = None
    selection_basis: SelectionBasis = "cv"
    log_target: bool = False
    stratify_split: bool = True
    importance_method: ImportanceMethod = "permutation"
    permutation_repeats: int = 15
    top_n_features: int = 20
    timeout_per_model: float | None = None
    random_state: int = 42
    n_jobs: int = -1
    save_all_models: bool = False
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    def __post_init__(self) -> None:
        self.features = list(self.features)
        self.models = list(self.models)
        self.output_dir = Path(self.output_dir)
        if self.metric is None:
            self.metric = DEFAULT_METRIC.get(self.task, "rmse")
        self.validate()

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    def validate(self) -> None:
        errors: list[str] = []
        if not self.target:
            errors.append("A target column is required.")
        if not self.features:
            errors.append("Select at least one input feature.")
        if self.target in self.features:
            errors.append(f"The target '{self.target}' cannot also be an input feature.")
        if len(set(self.features)) != len(self.features):
            errors.append("Duplicated feature names are not allowed.")
        if not self.models:
            errors.append("Select at least one model.")
        if self.n_trials < 1:
            errors.append("n_trials must be >= 1.")
        if self.cv_folds < 2:
            errors.append("cv_folds must be >= 2.")
        if self.cv_repeats < 1:
            errors.append("cv_repeats must be >= 1.")
        if not 0.0 <= self.test_size < 0.9:
            errors.append("test_size must be in [0, 0.9).")
        if self.task not in TASKS:
            errors.append(f"task must be one of {TASKS}.")
        if self.metric not in METRICS:
            errors.append(f"Unknown metric '{self.metric}'. Available: {sorted(METRICS)}.")
        elif METRICS[self.metric].task != self.task:
            valid = sorted(k for k, m in METRICS.items() if m.task == self.task)
            errors.append(
                f"Metric '{self.metric}' is not a {self.task} metric. Use one of {valid}."
            )
        if self.log_target and self.task == "classification":
            errors.append("log_target applies to regression only.")
        if self.selection_basis not in ("cv", "test"):
            errors.append("selection_basis must be 'cv' or 'test'.")
        if self.selection_basis == "test" and self.test_size == 0:
            errors.append("selection_basis='test' requires test_size > 0.")
        if self.importance_method not in ("permutation", "native"):
            errors.append("importance_method must be 'permutation' or 'native'.")
        if self.top_n_features < 1:
            errors.append("top_n_features must be >= 1.")
        if self.timeout_per_model is not None and self.timeout_per_model <= 0:
            errors.append("timeout_per_model must be positive when provided.")
        if errors:
            raise ValueError("Invalid training configuration:\n- " + "\n- ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data
