"""Training helpers for TabPFN models with additional_x drift signals."""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.config import DEFAULT_BATCH_SIZE_PREDICT
from src.utils import torch_helpers


class TabPFNRunArtifacts(dict):
    """Dictionary subclass to carry timing and device metadata."""


def fit_with_drift(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dist_shift_domain: np.ndarray,
    *,
    run_id: str = "tabpfn_temporal_fit_free_light_chain",
) -> Tuple[Any, TabPFNRunArtifacts]:
    """Mirror the notebook's fit cell with timing + device introspection."""

    t0 = time.perf_counter()
    fitted = model.fit(
        X_train.astype(np.float32),
        y_train.astype(int),
        additional_x={"dist_shift_domain": dist_shift_domain.astype(np.int64)},
    )
    fit_time = time.perf_counter() - t0
    add_device = torch.device("cpu")
    example_shape = None
    if hasattr(fitted, "additional_x_"):
        add_map = getattr(fitted, "additional_x_", None) or {}
        tensor = add_map.get("dist_shift_domain") if isinstance(add_map, dict) else None
        if isinstance(tensor, torch.Tensor):
            add_device = tensor.device
            example_shape = tuple(tensor.shape)
    artifacts = TabPFNRunArtifacts(
        run_id=run_id,
        fit_time=fit_time,
        additional_x_device=str(add_device),
        additional_x_shape=example_shape,
    )
    return fitted, artifacts
