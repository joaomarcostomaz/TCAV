"""Simple logging helpers to provide consistent console output."""
from __future__ import annotations

import logging

LOGGER_NAME = "tcav_refactor"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(LOGGER_NAME)


def info(msg: str) -> None:
    logger.info(msg)


def warn(msg: str) -> None:
    logger.warning(msg)
