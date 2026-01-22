"""Temporal walk-forward utilities."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def temporal_walk_forward_indices(
    years: np.ndarray,
    unique_years_sorted: Sequence[int],
    *,
    initial_train_years: int | None = None,
    test_window: int = 1,
    step: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Replicates the helper defined in the original notebook."""

    yrs = np.asarray(years, dtype=int)
    uniq = list(unique_years_sorted)
    if not uniq:
        return []
    n_unique = len(uniq)
    if initial_train_years is None:
        initial_train_years = max(1, n_unique // 2)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = initial_train_years - 1
    while train_end < n_unique - 1:
        train_years = set(uniq[: train_end + 1])
        test_start = train_end + 1
        test_end = min(n_unique - 1, train_end + test_window)
        test_years = set(uniq[test_start : test_end + 1])
        if not test_years:
            break
        train_idx = np.where(np.isin(yrs, list(train_years)))[0]
        test_idx = np.where(np.isin(yrs, list(test_years)))[0]
        if train_idx.size and test_idx.size:
            folds.append((train_idx, test_idx))
        train_end += step
    return folds
