"""TCAV with true gradients from differentiable TabPFN model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from src.concepts.cav import CAV


@dataclass
class TCAVGradientResult:
    cluster_id: int
    tcav_positive_fraction: float
    mean_derivative: float
    derivatives: np.ndarray


def extract_embeddings_with_gradients(
    model,
    X: np.ndarray,
    dist_shift_domain: np.ndarray,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extract embeddings from TabPFN model WITH gradient tracking.

    This is similar to src/embeddings/extractor.py but KEEPS gradients.

    Parameters:
        model: TabPFN model with _forward method
        X: Input features (n_samples, n_features)
        dist_shift_domain: Domain indices (n_samples,)
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        torch.Tensor: Embeddings with gradients enabled, shape (n_samples, emb_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_proc = getattr(model, "model_processed_", None) or model
    model_proc = model_proc.to(device)
    model_proc.eval()  # Set to eval mode but DON'T use no_grad

    model_proc.

    embeddings_list = []

    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        chunk = X[start:end]
        B = chunk.shape[0]

        # Prepare inputs in TabPFN format
        xb_np = np.stack([chunk, chunk], axis=0)  # (2, B, feat)
        xb_t = torch.tensor(xb_np, dtype=torch.float32, device=device, requires_grad=True)
        yb_t = torch.zeros((1, B, 1), dtype=torch.float32, device=device)

        dist_chunk = dist_shift_domain[start:end]
        dist_dup = np.stack([dist_chunk, dist_chunk], axis=0)
        dist_t = torch.tensor(dist_dup.reshape(2, B, 1), dtype=torch.long, device=device)

        x_arg = {"main": xb_t, "dist_shift_domain": dist_t}

        # Forward pass WITHOUT torch.no_grad() to keep gradients
        out = model_proc._forward(
            x_arg,
            yb_t,
            single_eval_pos=1,
            only_return_standard_out=False
        )

        # Extract embeddings (DON'T detach - keep gradients!)
        if "test_embeddings" in out:
            emb_batch = out["test_embeddings"]
        elif "embeddings" in out:
            emb_batch = out["embeddings"]
        else:
            raise ValueError(f"No embeddings found in output keys: {out.keys()}")

        # Shape should be (seq_len, batch, emb_dim) -> take first seq position
        if emb_batch.ndim == 3:
            emb_batch = emb_batch[0]  # (batch, emb_dim)

        embeddings_list.append(emb_batch)

    # Concatenate all batches - still has gradients!
    embeddings_full = torch.cat(embeddings_list, dim=0)

    return embeddings_full


def compute_tcav_with_gradients(
    cavs: Dict[int, CAV],
    model,
    X: np.ndarray,
    dist_shift_domain: np.ndarray,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> Dict[int, TCAVGradientResult]:
    """
    Compute TCAV scores using TRUE GRADIENTS from the model.

    This computes the directional derivative ∇_e p(y|e) · v_concept
    using PyTorch autograd, NOT finite differences.

    Parameters:
        cavs: Dictionary of CAV objects (cluster_id -> CAV)
        model: TabPFN model
        X: Input features (n_samples, n_features)
        dist_shift_domain: Domain indices (n_samples,)
        batch_size: Batch size
        device: Device to use

    Returns:
        Dictionary mapping cluster_id to TCAVGradientResult
    """
    if not cavs:
        return {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_proc = getattr(model, "model_processed_", None) or model
    model_proc = model_proc.to(device)
    model_proc.eval()

    results: Dict[int, TCAVGradientResult] = {}

    # Process in batches to avoid OOM
    n_samples = X.shape[0]
    all_derivatives = {cid: [] for cid in cavs.keys()}

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]
        dist_batch = dist_shift_domain[start:end]
        B = X_batch.shape[0]

        # Prepare inputs
        xb_np = np.stack([X_batch, X_batch], axis=0)
        xb_t = torch.tensor(xb_np, dtype=torch.float32, device=device, requires_grad=True)
        yb_t = torch.zeros((1, B, 1), dtype=torch.float32, device=device)

        dist_dup = np.stack([dist_batch, dist_batch], axis=0)
        dist_t = torch.tensor(dist_dup.reshape(2, B, 1), dtype=torch.long, device=device)

        x_arg = {"main": xb_t, "dist_shift_domain": dist_t}

        # Forward to get embeddings (with gradients)
        out = model_proc._forward(
            x_arg,
            yb_t,
            single_eval_pos=1,
            only_return_standard_out=False
        )

        embeddings = out["test_embeddings"]
        if embeddings.ndim == 3:
            embeddings = embeddings[0]  # (batch, emb_dim)

        # Get final logits/probabilities
        # The "standard" decoder output is the classification logits
        if "standard" in out:
            logits = out["standard"]
            if logits.ndim == 3:
                logits = logits[0]  # (batch, n_classes)
        else:
            # Fallback: pass embeddings through the standard decoder
            decoder = model_proc.decoder_dict["standard"]
            if decoder is None:
                raise ValueError("No standard decoder found in model")
            logits = decoder(embeddings.unsqueeze(0)).squeeze(0)

        # Convert to probabilities (positive class for binary, or class 1)
        probs = torch.softmax(logits, dim=-1)
        if probs.shape[-1] == 2:
            prob_positive = probs[:, 1]  # Binary classification
        else:
            # For multiclass, you might want to specify which class to track
            # Here we default to class 1
            prob_positive = probs[:, 1]

        # For each CAV, compute directional derivative
        for cluster_id, cav in cavs.items():
            v_concept = torch.tensor(
                cav.vector,
                dtype=torch.float32,
                device=device
            )

            # Compute gradient of each output w.r.t. embeddings
            derivatives_batch = []
            for i in range(B):
                # Compute ∇_e p_i
                if embeddings.grad is not None:
                    embeddings.grad.zero_()

                prob_positive[i].backward(retain_graph=True)

                # Gradient of p_i w.r.t. embedding_i
                grad_e = embeddings.grad[i]  # shape (emb_dim,)

                # Directional derivative: ∇_e p · v
                directional_deriv = torch.dot(grad_e, v_concept).item()
                derivatives_batch.append(directional_deriv)

                # Zero out for next iteration
                embeddings.grad.zero_()

            all_derivatives[cluster_id].extend(derivatives_batch)

    # Aggregate results per CAV
    for cluster_id, derivs in all_derivatives.items():
        derivs_array = np.array(derivs, dtype=np.float32)
        tcav_prop = float(np.mean(derivs_array > 0))
        mean_deriv = float(np.nanmean(derivs_array))

        results[cluster_id] = TCAVGradientResult(
            cluster_id=cluster_id,
            tcav_positive_fraction=tcav_prop,
            mean_derivative=mean_deriv,
            derivatives=derivs_array,
        )

    return results


def compute_tcav_with_gradients_efficient(
    cavs: Dict[int, CAV],
    model,
    X: np.ndarray,
    dist_shift_domain: np.ndarray,
    *,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> Dict[int, TCAVGradientResult]:
    """
    Compute TCAV using TRUE GRADIENTS through TabPFN model (not finite differences).

    This follows the approach documented in docs/FIXED_AND_TESTED.md:
    1. Start with requires_grad=True on INPUT tensor
    2. Forward pass propagates gradients naturally to embeddings
    3. Recompute logits from embeddings to maintain gradient connection
    4. Use torch.autograd.grad to compute true Jacobian

    The key fix: Disable save_peak_mem_factor to avoid in-place operation errors.
    """
    if not cavs:
        return {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_proc = getattr(model, "model_processed_", None) or model
    model_proc = model_proc.to(device)
    model_proc.eval()

    # CRITICAL FIX: Disable save_peak_mem_factor to allow gradients
    # This prevents "AssertionError: save_peak_mem_factor only supported during inference"
    if hasattr(model_proc, 'reset_save_peak_mem_factor'):
        model_proc.reset_save_peak_mem_factor(None)

    results: Dict[int, TCAVGradientResult] = {}

    for cluster_id, cav in cavs.items():
        v_concept = torch.tensor(
            cav.vector,
            dtype=torch.float32,
            device=device
        )

        all_derivatives = []

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            X_batch = X[start:end]
            dist_batch = dist_shift_domain[start:end]
            B = X_batch.shape[0]

            # Prepare inputs - KEY: requires_grad=True on INPUT
            xb_np = np.stack([X_batch, X_batch], axis=0)
            xb_t = torch.tensor(xb_np, dtype=torch.float32, device=device, requires_grad=True)
            yb_t = torch.zeros((1, B, 1), dtype=torch.float32, device=device)

            dist_dup = np.stack([dist_batch, dist_batch], axis=0)
            dist_t = torch.tensor(dist_dup.reshape(2, B, 1), dtype=torch.long, device=device)

            x_arg = {"main": xb_t, "dist_shift_domain": dist_t}

            # Forward pass with gradients enabled
            with torch.enable_grad():
                out = model_proc._forward(
                    x_arg,
                    yb_t,
                    single_eval_pos=1,
                    only_return_standard_out=False
                )

                embeddings = out["test_embeddings"]
                if embeddings.ndim == 3:
                    embeddings = embeddings[0]  # (batch, emb_dim)

                # CRITICAL: Recompute logits from embeddings to maintain gradient connection
                # Don't use out["standard"] - it may be disconnected from embeddings
                decoder = model_proc.decoder_dict["standard"]
                if decoder is None:
                    raise ValueError("No standard decoder found")

                logits = decoder(embeddings.unsqueeze(0)).squeeze(0)

                # Probabilities
                probs = torch.softmax(logits, dim=-1)
                prob_positive = probs[:, 1] if probs.shape[-1] >= 2 else probs[:, 0]

                # Compute jacobian: dp/de for all samples in batch
                jacobian = torch.autograd.grad(
                    outputs=prob_positive.sum(),
                    inputs=embeddings,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Directional derivative for each sample: J · v
                directional_derivs = torch.matmul(jacobian, v_concept)  # (batch_size,)

            all_derivatives.append(directional_derivs.detach().cpu().numpy())

        derivs_array = np.concatenate(all_derivatives).astype(np.float32)
        tcav_prop = float(np.mean(derivs_array > 0))
        mean_deriv = float(np.nanmean(derivs_array))

        results[cluster_id] = TCAVGradientResult(
            cluster_id=cluster_id,
            tcav_positive_fraction=tcav_prop,
            mean_derivative=mean_deriv,
            derivatives=derivs_array,
        )

    return results
