"""Embedding extraction utilities mirrored from the notebook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from src.utils.torch_helpers import make_dist_tensor


@dataclass
class EmbeddingResult:
    raw: np.ndarray
    flat: np.ndarray


def batch_get_embeddings_via_get_embeddings(
    model,
    X: np.ndarray,
    dist_shift_domain: np.ndarray,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
    target_shape_example: tuple[int, ...] | None = None,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        xb = X[start:end].astype(np.float32)
        dist_np = dist_shift_domain[start:end]
        dist_t = make_dist_tensor(
            dist_np,
            device=device or torch.device("cpu"),
            target_shape_example=target_shape_example,
        )
        emb_batch = model.get_embeddings(xb, additional_x={"dist_shift_domain": dist_t})
        if isinstance(emb_batch, torch.Tensor):
            emb_batch = emb_batch.detach().cpu().numpy()
        outputs.append(np.asarray(emb_batch))
    return np.vstack(outputs)


def flatten_embeddings(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3 and arr.shape[1] == 1:
        squeezed = np.squeeze(arr, axis=1)
        return squeezed.reshape(squeezed.shape[0], -1)
    return arr.reshape(arr.shape[0], -1)


def extract_embeddings_robust(
    model,
    X: np.ndarray,
    years: np.ndarray,
    year_to_domain: dict[int, int],
    *,
    max_samples: int | None = None,
    batch_size: int = 512,
    device: torch.device | None = None,
    target_shape_example: tuple[int, ...] | None = None,
) -> EmbeddingResult:
    X_all = X.astype(np.float32)
    years = years.astype(int)
    if max_samples is not None:
        X_all = X_all[:max_samples]
        years = years[:max_samples]
    dist_vec = np.array([year_to_domain[int(y)] for y in years], dtype=np.int64)
    if hasattr(model, "get_embeddings"):
        raw = batch_get_embeddings_via_get_embeddings(
            model,
            X_all,
            dist_vec,
            batch_size=batch_size,
            device=device,
            target_shape_example=target_shape_example,
        )
    else:
        raw = _extract_via_forward(model, X_all, dist_vec, batch_size)
    flat = flatten_embeddings(raw)
    return EmbeddingResult(raw=raw, flat=flat)


def _extract_via_forward(model, X: np.ndarray, dist_vec: np.ndarray, batch_size: int) -> np.ndarray:
    """Fallback path replicating the notebook's internal hook logic."""

    model_proc = getattr(model, "model_processed_", None) or model
    embeddings = []
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        chunk = X[start:end]
        B = chunk.shape[0]
        xb_np = np.stack([chunk, chunk], axis=0)  # (2, B, feat)
        xb_t = torch.tensor(xb_np, dtype=torch.float32, device=torch.device("cpu"))
        yb_t = torch.zeros((1, B, 1), dtype=torch.float32, device=xb_t.device)
        dist_chunk = dist_vec[start:end]
        dist_dup = np.stack([dist_chunk, dist_chunk], axis=0)
        dist_t = torch.tensor(dist_dup.reshape(2, B, 1), dtype=torch.long, device=xb_t.device)
        x_arg = {"main": xb_t, "dist_shift_domain": dist_t}
        with torch.no_grad():
            out = model_proc._forward(x_arg, yb_t, single_eval_pos=1, only_return_standard_out=False)
        tensor = _pick_embedding_tensor(out)
        emb_np = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        if emb_np.ndim == 3:
            emb_np = emb_np[0]
        embeddings.append(emb_np.astype(np.float32))
    return np.vstack(embeddings)


def _pick_embedding_tensor(output) -> np.ndarray | torch.Tensor:
    if isinstance(output, dict):
        for key in ("test_embeddings", "embeddings"):
            if key in output:
                return output[key]
        standard = output.get("standard_out") if "standard_out" in output else None
        if isinstance(standard, dict) and "test_embeddings" in standard:
            return standard["test_embeddings"]
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    return output
