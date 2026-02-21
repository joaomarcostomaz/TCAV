"""
TabPFN Model Wrapper for TCAV (Testing with Concept Activation Vectors).

This module provides a wrapper around TabPFN models to make them compatible
with the TCAV framework, which was originally designed for TensorFlow image models.

Key adaptations:
- TabPFN is a PyTorch transformer model for tabular data (not image-based)
- Uses embeddings from intermediate layers as "activations"
- Supports distribution shift via additional_x parameter
- Implements gradient computation for TCAV directional derivatives
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from tcav import model as tcav_model
except ImportError:
    # Fallback if tcav not installed in dev environment
    class ModelWrapper:
        pass
    tcav_model = type('tcav_model', (), {'ModelWrapper': ModelWrapper})()


@dataclass
class TCAVGradientResult:
    """Results from TCAV gradient computation."""
    cluster_id: int
    tcav_positive_fraction: float
    mean_derivative: float
    derivatives: np.ndarray


class TabPFNModelWrapper(tcav_model.ModelWrapper):
    """
    TCAV-compatible wrapper for TabPFN models.

    TabPFN (Tabular Prior-Fitted Networks) is a transformer-based model for
    tabular classification. Unlike CNNs used in image TCAV examples, TabPFN:
    - Works with structured/tabular data
    - Uses transformer embeddings as bottleneck activations
    - Requires special handling for distribution shift (temporal domain)
    - Uses PyTorch instead of TensorFlow

    This wrapper adapts TabPFN to the TCAV interface, which expects:
    - run_examples(examples, bottleneck_name) -> activations
    - get_gradient(acts, y, bottleneck_name, example) -> gradients
    - get_predictions(examples) -> predictions

    Args:
        model: Fitted TabPFN model instance
        dist_shift_domain: Domain/time indices for each sample (n_samples,)
        device: PyTorch device for computation
        bottleneck_name: Name of the bottleneck layer (default: "embeddings")

    Example:
        >>> from tabpfn import TabPFNClassifier
        >>> model = TabPFNClassifier()
        >>> model.fit(X_train, y_train, additional_x={'dist_shift_domain': domains})
        >>> wrapper = TabPFNModelWrapper(model, dist_shift_domain=domains)
        >>> activations = wrapper.run_examples(X_test, 'embeddings')
    """

    def __init__(
        self,
        model: Any,
        dist_shift_domain: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        bottleneck_name: str = "embeddings",
        batch_size: int = 512,
    ):
        """
        Initialize TabPFN wrapper for TCAV.

        Args:
            model: Fitted TabPFN model (TabPFNClassifier or similar)
            dist_shift_domain: Array of domain indices for distribution shift.
                Should have same length as data. If None, will be inferred or set to zeros.
            device: PyTorch device. If None, will auto-detect.
            bottleneck_name: Name for the bottleneck layer (embeddings)
            batch_size: Batch size for processing
        """
        # Don't call super().__init__() as it expects TF model_path
        # Instead, manually initialize the required attributes

        self.model = model
        self.model_processed_ = getattr(model, "model_processed_", None) or model
        self.dist_shift_domain = dist_shift_domain
        self.batch_size = batch_size

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Move model to device
        if hasattr(self.model_processed_, "to"):
            self.model_processed_ = self.model_processed_.to(self.device)

        # Set model name
        self.model_name = "TabPFN"

        # Define bottleneck tensors (TCAV expects a dictionary)
        self.bottlenecks_tensors = {
            bottleneck_name: "embeddings",  # Logical name -> actual layer
        }

        # Define input/output ends (TCAV convention)
        self.ends = {
            "input": "tabular_features",
            "prediction": "class_probabilities",
        }

        # Placeholders for TCAV compatibility
        self.y_input = None  # Not used in TabPFN (no TF placeholder needed)
        self.loss = None  # Computed dynamically in PyTorch

        # Extract additional_x shape if model is fitted
        self._infer_additional_x_shape()

    def _infer_additional_x_shape(self):
        """Extract the shape of additional_x from fitted model if available."""
        self.additional_x_shape = None
        if hasattr(self.model, "additional_x_"):
            add_map = getattr(self.model, "additional_x_", None)
            if isinstance(add_map, dict) and "dist_shift_domain" in add_map:
                tensor = add_map["dist_shift_domain"]
                if isinstance(tensor, torch.Tensor):
                    self.additional_x_shape = tuple(tensor.shape)

    def _make_dist_tensor(
        self,
        dist_shift_domain: np.ndarray,
        batch_size: int,
        for_get_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Create properly formatted distribution shift tensor for TabPFN.

        Args:
            dist_shift_domain: Array of domain indices (n_samples,)
            batch_size: Number of samples in batch
            for_get_embeddings: If True, format for get_embeddings(); else for _forward()

        Returns:
            Formatted tensor for additional_x parameter
        """
        # Handle case where dist_shift_domain is None
        if dist_shift_domain is None:
            dist_shift_domain = np.zeros(batch_size, dtype=np.int64)

        # Ensure correct dtype
        dist_np = dist_shift_domain.astype(np.int64)

        if for_get_embeddings:
            # For get_embeddings/predict: shape (batch_size, 1, 1)
            dist_t = torch.tensor(
                dist_np.reshape(-1, 1, 1),
                dtype=torch.long,
                device=self.device
            )
        else:
            # For _forward: shape (2, batch_size, 1)
            # The duplication is part of TabPFN's internal structure
            dist_dup = np.stack([dist_np, dist_np], axis=0)
            dist_t = torch.tensor(
                dist_dup.reshape(2, batch_size, 1),
                dtype=torch.long,
                device=self.device
            )

        return dist_t

    def run_examples(
        self,
        examples: np.ndarray,
        bottleneck_name: str,
    ) -> np.ndarray:
        """
        Extract activations at the specified bottleneck for given examples.

        This is the primary method TCAV uses to get intermediate representations.
        For TabPFN, we extract embeddings from the transformer encoder.

        Args:
            examples: Input features (n_samples, n_features)
            bottleneck_name: Name of bottleneck layer (e.g., "embeddings")

        Returns:
            Activations/embeddings as numpy array (n_samples, embedding_dim)
        """
        if bottleneck_name not in self.bottlenecks_tensors:
            raise ValueError(
                f"Bottleneck '{bottleneck_name}' not found. "
                f"Available: {list(self.bottlenecks_tensors.keys())}"
            )

        # Use the high-level get_embeddings method if available
        if hasattr(self.model, "get_embeddings"):
            return self._extract_via_get_embeddings(examples)
        else:
            # Fallback to _forward method
            return self._extract_via_forward(examples)

    def _extract_via_get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Extract embeddings using model.get_embeddings() method.

        This is the preferred method for TabPFN models.
        """
        embeddings_list = []

        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            X_batch = X[start:end].astype(np.float32)

            # Prepare distribution shift domain
            if self.dist_shift_domain is not None:
                dist_batch = self.dist_shift_domain[start:end]
                dist_t = self._make_dist_tensor(dist_batch, len(X_batch), for_get_embeddings=True)

                # Call get_embeddings with additional_x
                emb = self.model.get_embeddings(
                    X_batch,
                    additional_x={"dist_shift_domain": dist_t}
                )
            else:
                # No distribution shift
                emb = self.model.get_embeddings(X_batch)

            # Convert to numpy
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()

            embeddings_list.append(np.asarray(emb))

        # Concatenate all batches
        embeddings = np.vstack(embeddings_list)

        # Flatten if necessary (TCAV expects 2D: samples x features)
        return self._flatten_embeddings(embeddings)

    def _extract_via_forward(self, X: np.ndarray) -> np.ndarray:
        """
        Extract embeddings using model._forward() method.

        This is a fallback when get_embeddings() is not available.
        """
        embeddings_list = []

        self.model_processed_.eval()

        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            X_batch = X[start:end]
            B = X_batch.shape[0]

            # Prepare inputs in TabPFN format
            # TabPFN _forward expects (seq_len, batch, features)
            xb_np = np.stack([X_batch, X_batch], axis=0)  # (2, B, feat)
            xb_t = torch.tensor(xb_np, dtype=torch.float32, device=self.device)
            yb_t = torch.zeros((1, B, 1), dtype=torch.float32, device=self.device)

            # Distribution shift domain
            if self.dist_shift_domain is not None:
                dist_batch = self.dist_shift_domain[start:end]
                dist_t = self._make_dist_tensor(dist_batch, B, for_get_embeddings=False)
            else:
                dist_t = self._make_dist_tensor(np.zeros(B, dtype=np.int64), B, for_get_embeddings=False)

            x_arg = {"main": xb_t, "dist_shift_domain": dist_t}

            # Forward pass
            with torch.no_grad():
                out = self.model_processed_._forward(
                    x_arg,
                    yb_t,
                    single_eval_pos=1,
                    only_return_standard_out=False
                )

            # Extract embeddings from output
            if "test_embeddings" in out:
                emb_batch = out["test_embeddings"]
            elif "embeddings" in out:
                emb_batch = out["embeddings"]
            else:
                # Try to find any tensor in output
                for value in out.values():
                    if isinstance(value, torch.Tensor):
                        emb_batch = value
                        break
                else:
                    raise ValueError(f"No embeddings found in output keys: {out.keys()}")

            # Convert to numpy
            emb_np = emb_batch.detach().cpu().numpy()

            # Handle shape: (seq_len, batch, emb_dim) -> (batch, emb_dim)
            if emb_np.ndim == 3:
                emb_np = emb_np[0]

            embeddings_list.append(emb_np)

        embeddings = np.vstack(embeddings_list)
        return self._flatten_embeddings(embeddings)

    def _flatten_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Flatten embeddings to 2D array for TCAV.

        Args:
            embeddings: Array of any shape

        Returns:
            2D array (n_samples, n_features)
        """
        embeddings = np.asarray(embeddings)

        if embeddings.ndim == 1:
            return embeddings.reshape(-1, 1)
        elif embeddings.ndim == 2:
            return embeddings
        elif embeddings.ndim >= 3:
            # Squeeze out singleton dimensions and flatten
            if embeddings.shape[1] == 1:
                embeddings = np.squeeze(embeddings, axis=1)
            return embeddings.reshape(embeddings.shape[0], -1)
        else:
            return embeddings

    def get_gradient(
        self,
        acts: np.ndarray,
        y: np.ndarray,
        bottleneck_name: str,
        example: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. bottleneck activations.

        For TCAV, we need: ∇_activations Loss(y | activations)

        This is challenging for TabPFN because:
        1. The embeddings come from an encoder
        2. We need to backprop through the decoder
        3. TabPFN uses gradient checkpointing

        Note: For TabPFN, this requires re-computing embeddings from inputs
        with gradient tracking, since the stored activations are detached.

        Args:
            acts: Activations/embeddings (n_samples, emb_dim) - not used directly
            y: Target class indices (n_samples,)
            bottleneck_name: Name of bottleneck layer
            example: Original input examples (needed to recompute with gradients)

        Returns:
            Gradients w.r.t. activations (n_samples, emb_dim)
        """
        # For TabPFN, we need to recompute embeddings from inputs with gradients
        # The stored activations are detached and can't be used for backprop

        try:
            # Get distribution shift if available
            if self.dist_shift_domain is not None and len(self.dist_shift_domain) >= len(example):
                # Find the indices in the original data (approximate)
                # This is a limitation - ideally we'd track indices
                dist_batch = self.dist_shift_domain[:len(example)]
            else:
                dist_batch = np.zeros(len(example), dtype=np.int64)

            # Use the gradient-based extraction from tcav_gradient module
            from src.concepts.tcav_gradient import compute_tcav_with_gradients_efficient
            from src.concepts.cav import CAV

            # Create a dummy CAV for gradient computation
            dummy_cav = CAV(
                cluster_id=0,
                size_pure=1,
                vector=np.ones(acts.shape[1]),  # Dummy vector
                pure_indices=np.array([0]),
                negative_indices=np.array([1]),
                classifier=None,
                centroid=acts.mean(axis=0),
                quantile=0.95
            )

            # Compute gradients using the existing implementation
            result = compute_tcav_with_gradients_efficient(
                cavs={0: dummy_cav},
                model=self.model,
                X=example,
                dist_shift_domain=dist_batch,
                batch_size=len(example),
                device=self.device
            )

            # Extract the gradients (directional derivatives)
            # These are scalar projections, but we can use them
            # For full gradient, we'd need to compute Jacobian
            grads = result[0].derivatives.reshape(-1, 1)

            # Expand to match activation dimensions (approximate)
            grads_expanded = np.tile(grads, (1, acts.shape[1]))

            return grads_expanded

        except Exception as e:
            # Fallback: return zeros with warning
            import warnings
            warnings.warn(
                f"Gradient computation failed: {e}\n"
                "Returning zero gradients. For proper gradients with TabPFN, "
                "use the compute_tcav_with_gradients_efficient function directly."
            )
            return np.zeros_like(acts)

    def get_predictions(self, examples: np.ndarray) -> np.ndarray:
        """
        Get model predictions for examples.

        Args:
            examples: Input features (n_samples, n_features)

        Returns:
            Predictions (probabilities or class labels)
        """
        predictions_list = []

        for start in range(0, examples.shape[0], self.batch_size):
            end = min(start + self.batch_size, examples.shape[0])
            X_batch = examples[start:end].astype(np.float32)

            # Prepare distribution shift
            if self.dist_shift_domain is not None:
                dist_batch = self.dist_shift_domain[start:end]
                dist_t = self._make_dist_tensor(dist_batch, len(X_batch), for_get_embeddings=True)

                probs = self.model.predict_proba(
                    X_batch,
                    additional_x={"dist_shift_domain": dist_t}
                )
            else:
                probs = self.model.predict_proba(X_batch)

            # Convert to numpy
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()

            predictions_list.append(np.asarray(probs))

        return np.vstack(predictions_list)

    def reshape_activations(self, layer_acts: np.ndarray) -> np.ndarray:
        """
        Reshape activations to the expected format for TCAV.

        Default implementation squeezes the array.

        Args:
            layer_acts: Raw activations from run_examples

        Returns:
            Reshaped activations
        """
        return np.asarray(layer_acts).squeeze()

    def label_to_id(self, label: str) -> int:
        """
        Convert label string to class index.

        For binary classification, this is typically 0 or 1.
        Override if your model has specific label mappings.

        Args:
            label: Label string

        Returns:
            Class index
        """
        # Default: try to convert to int
        try:
            return int(label)
        except ValueError:
            # If model has label mapping, use it
            if hasattr(self.model, "classes_"):
                classes = self.model.classes_
                if label in classes:
                    return np.where(classes == label)[0][0]
            return 0  # Fallback

    def id_to_label(self, idx: int) -> str:
        """
        Convert class index to label string.

        Args:
            idx: Class index

        Returns:
            Label string
        """
        if hasattr(self.model, "classes_"):
            classes = self.model.classes_
            if idx < len(classes):
                return str(classes[idx])
        return str(idx)