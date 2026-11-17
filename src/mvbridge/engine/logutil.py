"""Run logs for the baseline entry points."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml


def configure_logging(output_dir: str | Path | None, name: str = "mvbridge") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_config(logger: logging.Logger, config: dict) -> None:
    logger.info("config\n%s", yaml.safe_dump(config, sort_keys=False))
