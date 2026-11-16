from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: str | int | None = None) -> None:
    level = level or os.getenv("HEFESTO_LOG_LEVEL", "INFO")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        root.setLevel(level)
    for noisy in ("optuna", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
