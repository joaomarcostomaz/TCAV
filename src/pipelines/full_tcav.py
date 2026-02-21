"""End-to-end pipeline orchestrating the TCAV workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from src import config
from src.concepts import (
    CAV,
    SurrogateArtifacts,
    TCAVResult,
    compute_model_probabilities,
    compute_tcav,
    train_cavs,
    train_surrogate,
)
from src.data.ingestion import DatasetSummary, load_free_light_chain_dataset, summarize_dataset
from src.data.preprocessing import (
    NormalizedData,
    PreprocessResult,
    TemporalSplit,
    build_feature_matrix,
    normalize_features,
    temporal_split,
)
from src.embeddings.clustering import ClusterArtifacts, kernel_kmeans_on_embeddings
from src.embeddings.extractor import EmbeddingResult, extract_embeddings_robust
from src.modeling.inference import InferenceResult, batched_predict_proba, evaluate_predictions, per_year_metrics
from src.modeling.tabpfn_loader import load_tabpfn_model
from src.modeling.tabpfn_trainer import TabPFNRunArtifacts, fit_with_drift
from src.reporting.tcav_summary import plot_tcav_bar, plot_tcav_histograms, tcav_results_to_frame
from src.utils import logging as log


@dataclass
class TCAVPipelineConfig:
    data_path: str = "free_light_chain_mortality.csv"
    include_sex: bool = True
    train_year_frac: float = config.TRAIN_YEAR_FRAC
    embedding_batch_size: int = 512
    embedding_max_samples: Optional[int] = None
    n_clusters: int = 8
    kpca_components: int = 64
    use_minibatch_kmeans: bool = True
    tsne_subsample: Optional[int] = 2000
    cav_quantile: float = 0.25
    cav_min_pure: int = 10
    cav_negative_multiplier: int = 5
    cav_min_negative: int = 200
    surrogate_alpha: float = 1.0
    surrogate_subsample: Optional[int] = None
    surrogate_standardize: bool = True
    tcav_finite_rel: float = 0.05
    tcav_batch_size: int = 1024
    show_plots: bool = False


@dataclass
class TCAVPipelineOutputs:
    dataset: Any
    dataset_summary: DatasetSummary
    preprocess: PreprocessResult
    split: TemporalSplit
    normalized: NormalizedData
    model: Any
    fit_artifacts: TabPFNRunArtifacts
    inference: InferenceResult
    per_year_metrics: Optional[Any]
    train_embeddings: EmbeddingResult
    test_embeddings: EmbeddingResult
    cluster_artifacts: ClusterArtifacts
    cavs: Dict[int, CAV]
    surrogate: SurrogateArtifacts
    tcav_results: Dict[int, TCAVResult]
    tcav_frame: Any


def run_full_tcav_pipeline(cfg: TCAVPipelineConfig = TCAVPipelineConfig()) -> TCAVPipelineOutputs:
    log.info("Loading dataset ...")
    df = load_free_light_chain_dataset(cfg.data_path)
    dataset_summary = summarize_dataset(df, show_plots=cfg.show_plots)

    log.info("Building feature matrix ...")
    preprocess = build_feature_matrix(df, include_sex=cfg.include_sex)

    log.info("Temporal split ...")
    split = temporal_split(
        preprocess.feature_frame,
        preprocess.target,
        preprocess.years,
        train_frac=cfg.train_year_frac,
    )

    log.info("Normalizing features ...")
    normalized = normalize_features(split)
    X_train_np = normalized.X_train.to_numpy(dtype=np.float32)
    X_test_np = normalized.X_test.to_numpy(dtype=np.float32)
    dist_train = np.array([split.year_to_domain[int(y)] for y in split.years_train], dtype=np.int64)
    dist_test = np.array([split.year_to_domain[int(y)] for y in split.years_test], dtype=np.int64)

    log.info("Loading TabPFN model ...")
    model = load_tabpfn_model()
    model, fit_artifacts = fit_with_drift(
        model,
        X_train_np,
        split.y_train,
        dist_train,
    )

    device_str = fit_artifacts.get("additional_x_device", "cpu")
    try:
        add_device = torch.device(device_str)
    except Exception:
        add_device = torch.device("cpu")
    example_shape = fit_artifacts.get("additional_x_shape")

    log.info("Evaluating on test set ...")
    probs = batched_predict_proba(
        model,
        X_test_np,
        dist_test,
        device=add_device,
        example_shape=example_shape,
    )
    inference = evaluate_predictions(split.y_test, probs)
    year_metrics = per_year_metrics(split.years_test, split.y_test, inference.y_pred, inference.probs_pos)

    log.info("Extracting embeddings ...")
    year_to_domain = split.year_to_domain
    train_embeddings = extract_embeddings_robust(
        model,
        X_train_np,
        split.years_train,
        year_to_domain,
        max_samples=cfg.embedding_max_samples,
        batch_size=cfg.embedding_batch_size,
        device=add_device,
        target_shape_example=example_shape,
    )
    test_embeddings = extract_embeddings_robust(
        model,
        X_test_np,
        split.years_test,
        year_to_domain,
        max_samples=cfg.embedding_max_samples,
        batch_size=cfg.embedding_batch_size,
        device=add_device,
        target_shape_example=example_shape,
    )

    log.info("Clustering embeddings ...")
    cluster_artifacts = kernel_kmeans_on_embeddings(
        train_embeddings.flat,
        n_clusters=cfg.n_clusters,
        kpca_components=cfg.kpca_components,
        use_minibatch=cfg.use_minibatch_kmeans,
        tsne_subsample=cfg.tsne_subsample,
    )

    log.info("Training CAVs ...")
    cavs = train_cavs(
        train_embeddings.flat,
        cluster_artifacts.labels,
        cluster_artifacts.centroids,
        quantile=cfg.cav_quantile,
        min_pure=cfg.cav_min_pure,
        negative_multiplier=cfg.cav_negative_multiplier,
        min_negative=cfg.cav_min_negative,
    )

    log.info("Computing surrogate targets ...")
    p0_train = compute_model_probabilities(
        model,
        X_train_np,
        split.years_train,
        year_to_domain,
        device=add_device,
        example_shape=example_shape,
    )

    log.info("Training surrogate Ridge model ...")
    surrogate = train_surrogate(
        train_embeddings.flat,
        p0_train,
        alpha=cfg.surrogate_alpha,
        subsample=cfg.surrogate_subsample,
        standardize=cfg.surrogate_standardize,
    )

    log.info("Running TCAV ...")
    tcav_results = compute_tcav(
        cavs,
        test_embeddings.flat,
        surrogate,
        finite_rel=cfg.tcav_finite_rel,
        batch_size=cfg.tcav_batch_size,
    )
    tcav_frame = tcav_results_to_frame(tcav_results, cavs)

    if cfg.show_plots:
        plot_tcav_bar(tcav_results)
        plot_tcav_histograms(tcav_results)

    return TCAVPipelineOutputs(
        dataset=df,
        dataset_summary=dataset_summary,
        preprocess=preprocess,
        split=split,
        normalized=normalized,
        model=model,
        fit_artifacts=fit_artifacts,
        inference=inference,
        per_year_metrics=year_metrics,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        cluster_artifacts=cluster_artifacts,
        cavs=cavs,
        surrogate=surrogate,
        tcav_results=tcav_results,
        tcav_frame=tcav_frame,
    )
