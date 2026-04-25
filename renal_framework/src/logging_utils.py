"""
Logging setup helpers for consistent console/file logs across pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str = "renal_framework", level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a logger with a stream handler.

    This function is idempotent: it won't duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(level)
        logger.addHandler(ch)

    logger.propagate = False
    return logger


def add_file_handler(logger: logging.Logger, log_file: str | Path, level: int = logging.INFO) -> None:
    """
    Add a file handler to an existing logger.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate file handlers for same file
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path:
            return

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)