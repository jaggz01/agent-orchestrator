"""Spoolable file logger for detailed execution trace."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def build_spoolable_logger(log_path: str = "logs/agent_workflow.log") -> logging.Logger:
    """Create a rotating logger for step-by-step durable logs."""

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("agentic_workflow_engine")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    return logger
