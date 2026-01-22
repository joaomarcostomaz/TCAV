"""Utility helpers for building tensors used by TabPFN."""
from __future__ import annotations

import numpy as np
import torch


def make_dist_tensor(
    dist_np: np.ndarray,
    *,
    device: torch.device,
    target_shape_example: tuple[int, ...] | None = None,
) -> torch.Tensor:
    tensor = torch.tensor(dist_np, dtype=torch.long, device=device)
    if tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1, 1)
    if target_shape_example is not None and len(target_shape_example) == 3:
        tensor = tensor.reshape(-1, target_shape_example[1], target_shape_example[2])
    return tensor
