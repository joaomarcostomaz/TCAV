"""Embedding clustering utilities (KPCA + KMeans, RFF pipelines)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE

sns.set(style="whitegrid")


@dataclass
class ClusterArtifacts:
    labels: np.ndarray
    kpca_model: KernelPCA | None
    kmeans_model: KMeans | MiniBatchKMeans
    centroids: np.ndarray


def kernel_kmeans_on_embeddings(
    embeddings: np.ndarray,
    *,
    n_clusters: int = 8,
    kpca_components: int = 64,
    use_minibatch: bool = True,
    tsne_subsample: int | None = 2000,
    random_state: int = 42,
) -> ClusterArtifacts:
    kpca = KernelPCA(n_components=kpca_components, kernel="rbf", random_state=random_state)
    emb_kpca = kpca.fit_transform(embeddings)
    if use_minibatch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1024)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(emb_kpca)
    centroids = _centroids_from_labels(embeddings, labels, n_clusters)
    _plot_tsne(embeddings, labels, tsne_subsample)
    return ClusterArtifacts(labels=labels, kpca_model=kpca, kmeans_model=kmeans, centroids=centroids)


def _centroids_from_labels(embeddings: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centroids = []
    for idx in range(k):
        mask = labels == idx
        if not mask.any():
            centroids.append(np.zeros((embeddings.shape[1],), dtype=float))
        else:
            centroids.append(embeddings[mask].mean(axis=0))
    return np.vstack(centroids)


def _plot_tsne(embeddings: np.ndarray, labels: np.ndarray, subsample: int | None) -> None:
    n = embeddings.shape[0]
    if subsample is not None and n > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, subsample, replace=False)
        emb = embeddings[idx]
        lab = labels[idx]
    else:
        emb = embeddings
        lab = labels
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=800, init="pca")
    emb_tsne = tsne.fit_transform(emb)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=emb_tsne[:, 0], y=emb_tsne[:, 1], hue=lab, palette="tab10", s=20, alpha=0.8)
    plt.title("Clusters (t-SNE)")
    plt.legend(title="cluster", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
