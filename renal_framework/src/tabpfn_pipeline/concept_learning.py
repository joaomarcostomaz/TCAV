"""
tabpfn_pipeline.concept_learning

Concept decomposition modules:
- Dictionary Learning wrapper
- Sparse Autoencoder (tied decoder)
- utility functions for activations/reconstruction stats
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import DictionaryLearning


# -----------------------------
# Configs
# -----------------------------
@dataclass
class DictionaryLearningConfig:
    n_components: int = 8
    transform_algorithm: str = "lasso_lars"
    max_iter: int = 1000
    random_state: int = 42


@dataclass
class SAEConfig:
    emb_dim: int
    n_factors: int
    alpha_sparse: float = 1e-1
    lr: float = 1e-3
    epochs: int = 1000
    use_decoder_bias: bool = False
    device: str = "cpu"


# -----------------------------
# Dictionary learning
# -----------------------------
def fit_dictionary_learning(
    embeddings_discovery: np.ndarray,
    embeddings_cav_train: Optional[np.ndarray],
    cfg: DictionaryLearningConfig,
) -> Dict[str, Any]:
    """
    Fit dictionary learning on discovery embeddings and transform CAV train embeddings.
    """
    dl = DictionaryLearning(
        n_components=cfg.n_components,
        transform_algorithm=cfg.transform_algorithm,
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
    )

    A_discovery = dl.fit_transform(embeddings_discovery)
    W_latent = dl.components_.T  # [emb_dim, n_factors]

    out = {
        "dict_learner": dl,
        "W_latent": W_latent,
        "activations_discovery_train": A_discovery,
        "embeddings_discovery": embeddings_discovery,
    }

    if embeddings_cav_train is not None:
        A_cav = dl.transform(embeddings_cav_train)
        out["activations_cav_train"] = A_cav
        out["embeddings_cav_train"] = embeddings_cav_train

    return out


def transform_dictionary_learning(dict_learner: DictionaryLearning, embeddings: np.ndarray) -> np.ndarray:
    """
    Transform embeddings to concept activations using fitted dictionary learner.
    """
    return dict_learner.transform(embeddings)


# -----------------------------
# Sparse Autoencoder
# -----------------------------
class SparseAutoencoderTied(nn.Module):
    """
    Tied-weight SAE:
      code = ReLU(encoder(x))
      recon = code @ encoder.weight
    """

    def __init__(
        self,
        emb_dim: int,
        n_factors: int,
        alpha_sparse: float = 1e-3,
        use_decoder_bias: bool = False,
    ):
        super().__init__()
        self.encoder = nn.Linear(emb_dim, n_factors, bias=True)
        self.activation = nn.ReLU()
        self.alpha_sparse = alpha_sparse

        if use_decoder_bias:
            self.decoder_bias = nn.Parameter(torch.zeros(emb_dim))
        else:
            self.decoder_bias = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.activation(self.encoder(x))
        recon = torch.nn.functional.linear(code, self.encoder.weight.t(), bias=self.decoder_bias)
        return recon, code

    def loss(self, x: torch.Tensor, recon: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        mse = torch.nn.functional.mse_loss(recon, x)
        l1 = self.alpha_sparse * code.abs().mean()
        return mse + l1


def fit_sae(
    embeddings_discovery: np.ndarray,
    cfg: SAEConfig,
    verbose_every: int = 50,
) -> Dict[str, Any]:
    """
    Train SAE on discovery embeddings.
    """
    device = torch.device(cfg.device)
    x_np = embeddings_discovery.astype(np.float32)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)

    model = SparseAutoencoderTied(
        emb_dim=cfg.emb_dim,
        n_factors=cfg.n_factors,
        alpha_sparse=cfg.alpha_sparse,
        use_decoder_bias=cfg.use_decoder_bias,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        recon, code = model(x_t)
        loss = model.loss(x_t, recon, code)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu().item())
        history.append(loss_val)

        if verbose_every > 0 and (ep + 1) % verbose_every == 0:
            with torch.no_grad():
                sparsity = float((code <= 1e-5).float().mean().detach().cpu().item())
            print(f"[SAE] epoch={ep+1:04d} loss={loss_val:.6f} sparsity={sparsity:.2%}")

    model.eval()
    with torch.no_grad():
        recon, code = model(x_t)
        codes = code.detach().cpu().numpy()
        recon_np = recon.detach().cpu().numpy()
        decoder_atoms = model.encoder.weight.t().detach().cpu().numpy()  # [n_factors, emb_dim]
        mse = float(np.mean((x_np - recon_np) ** 2))
        sparsity = float((codes <= 1e-5).mean())

    return {
        "model_sae": model,
        "codes_train_sae": codes,
        "decoder_atoms_sae": decoder_atoms,
        "reconstruction_mse": mse,
        "sparsity_level": sparsity,
        "history": history,
    }


def transform_sae(model_sae: SparseAutoencoderTied, embeddings: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Encode embeddings through trained SAE encoder.
    """
    x = torch.tensor(embeddings.astype(np.float32), dtype=torch.float32, device=device)
    model_sae = model_sae.to(device)
    model_sae.eval()
    with torch.no_grad():
        codes = model_sae.activation(model_sae.encoder(x)).detach().cpu().numpy()
    return codes


# -----------------------------
# Shared concept accessor
# -----------------------------
def get_concept_activations(
    embeddings: np.ndarray,
    source: str,
    dict_learning_info: Optional[Dict[str, Any]] = None,
    model_sae: Optional[SparseAutoencoderTied] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Unified accessor:
      source='dl'  -> DictionaryLearning transform
      source='sae' -> SAE encoder activations
    """
    source = source.lower()
    if source == "dl":
        if dict_learning_info is None or "dict_learner" not in dict_learning_info:
            raise ValueError("dict_learning_info with 'dict_learner' is required for source='dl'.")
        return transform_dictionary_learning(dict_learning_info["dict_learner"], embeddings)

    if source == "sae":
        if model_sae is None:
            raise ValueError("model_sae is required for source='sae'.")
        return transform_sae(model_sae, embeddings, device=device)

    raise ValueError(f"Unknown source: {source}")