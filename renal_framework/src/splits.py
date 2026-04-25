"""
Split utilities for patient-level and temporal-safe partitions.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
from sklearn.model_selection import train_test_split


def patient_split(
    patient_ids: Iterable[str],
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[set[str], set[str]]:
    """
    Split patient IDs into train/test sets.
    """
    patient_ids = np.asarray(list(patient_ids), dtype=str)
    train_pat, test_pat = train_test_split(patient_ids, test_size=test_size, random_state=random_state)
    return set(map(str, train_pat)), set(map(str, test_pat))


def temporal_holdout_split(
    years: np.ndarray,
    n_val_years: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (train_mask, val_mask) using last n years as validation.
    """
    years = np.asarray(years).astype(int)
    uniq = sorted(np.unique(years))
    if len(uniq) <= n_val_years:
        raise ValueError("Not enough unique years for temporal validation split.")
    val_years = set(uniq[-n_val_years:])
    val_mask = np.isin(years, list(val_years))
    train_mask = ~val_mask
    return train_mask, val_mask