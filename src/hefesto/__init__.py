from hefesto.config import TrainingConfig
from hefesto.models import available_models, get_model_spec
from hefesto.training import ModelResult, SurrogateTrainer, TrainingResult

__version__ = "1.0.0"

__all__ = [
    "ModelResult",
    "SurrogateTrainer",
    "TrainingConfig",
    "TrainingResult",
    "__version__",
    "available_models",
    "get_model_spec",
]
