"""
I/O helpers for filesystem operations and common dataframe persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists and return it as Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_feather(path: str | Path) -> pd.DataFrame:
    """
    Load a Feather file into a DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feather file not found: {p}")
    return pd.read_feather(p)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """
    Save dataframe to CSV with parent directory auto-created.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index)


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    return pd.read_csv(p)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """
    Save Python object as JSON.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """
    Load JSON file into Python object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)