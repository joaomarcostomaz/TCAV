#!/usr/bin/env python
"""
run.py

Unified CLI entrypoint for:
- TS pipeline (deep temporal baselines)
- TabPFN concept pipeline (embeddings, concepts, TCAV, phenotype, ACE)

Orchestration-first:
wires src/* modules while keeping implementation details modular.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.io_utils import ensure_dir, load_feather, save_csv, save_json
from src.logging_utils import get_logger, add_file_handler
from src.splits import patient_split

# TS pipeline modules
from src.ts_pipeline.preprocessing import (
    SelectionConfig,
    LGBSelectionConfig,
    infer_train_test_years as ts_infer_train_test_years,
    build_patient_availability_table,
    select_full_balanced_patients,
    select_top_events_via_lgbm,
    build_train_test_rows as ts_build_train_test_rows,
    build_temporal_sequences,
)
from src.ts_pipeline.datasets import (
    TemporalSequenceDataset,
    LoaderConfig,
    build_model_aware_loader,
)
from src.ts_pipeline.models import build_model
from src.ts_pipeline.training import (
    TrainConfig,
    train_one_model,
    predict_logits,
    evaluate_per_year,
)
from src.ts_pipeline.evaluation import (
    RunArtifacts,
    save_run_outputs,
    aggregate_all_runs,
)

# TabPFN pipeline modules
from src.tabpfn_pipeline.preprocessing import (
    TabPFNPrepConfig,
    canonicalize_event_df as tab_canonicalize_event_df,
    prepare_tabpfn_rows,
)
from src.tabpfn_pipeline.evaluation import (
    TabPFNEvalConfig,
    fit_drift_resilient_tabpfn,
    walkforward_evaluate_tabpfn,
    save_tabpfn_temporal_artifacts,
)
from src.tabpfn_pipeline.embedding import (
    EmbeddingExtractConfig,
    fit_or_load_feature_scaler,
    load_or_extract_embeddings,
    fit_embedding_scaler,
    transform_embeddings,
    temporal_test_subsplits,
)
from src.tabpfn_pipeline.concept_learning import (
    DictionaryLearningConfig,
    SAEConfig,
    fit_dictionary_learning,
    fit_sae,
    get_concept_activations,
)
from src.tabpfn_pipeline.rules import (
    RuleExtractionConfig,
    extract_rules_per_percentile,
    select_best_percentile,
)
from src.tabpfn_pipeline.tcav import (
    train_cavs_from_activations,
    train_sae_cavs_from_rules,
    compute_decoder_gradients_vmap,
    compute_tcav_from_gradients,
    robust_tcav_significance_test,
    filter_significant_factors_by_tcav,
    build_matched_rules_tcav_table,
)
from src.tabpfn_pipeline.phenotype import (
    run_feature_association_dual_split,
    run_sparse_readout_dual_split,
    run_outcome_association_dual_split,
    build_phenotype_report,
)
from src.tabpfn_pipeline.ace import (
    run_ace_for_concepts,
    ace_validation_summary,
    build_ace_summary_df,
)

from src.tabpfn_pipeline.reporting import (
    robust_tcav_results_to_df,
    tcav_results_to_df,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(use_cuda_if_available: bool = True) -> str:
    if use_cuda_if_available and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def default_lgb_params(seed: int) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": seed,
        "n_jobs": 4,
        "num_leaves": 64,
        "learning_rate": 0.1,
        "max_depth": -1,
    }


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / torch-ish objects to JSON-safe python types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# -----------------------------------------------------------------------------
# TS pipeline orchestration
# -----------------------------------------------------------------------------
def run_ts_pipeline(cfg_path: str, logger_name: str = "renal_framework.ts") -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg.ts.rng_seed)

    logger = get_logger(logger_name)
    results_dir = ensure_dir(cfg.paths.results_dir)
    add_file_handler(logger, Path(results_dir) / "ts_pipeline.log")

    logger.info("Starting TS pipeline | config=%s", cfg_path)

    # Load + canonical preprocessing
    df = load_feather(cfg.paths.data_feather).copy()
    df["date"] = np.array(df["date"], dtype="datetime64[ns]")
    df["year"] = np.asarray(df["date"], dtype="datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
    df["patient_id"] = df["patient_id"].astype(str)

    years_all = sorted(df["year"].unique())
    train_years, test_years = ts_infer_train_test_years(
        years_all,
        forced_start=cfg.ts.train_year_start if cfg.ts.train_year_start is not None else 1997,
        forced_end=cfg.ts.train_year_end if cfg.ts.train_year_end is not None else 2006,
    )

    all_patients = np.asarray(df["patient_id"].unique(), dtype=str)
    train_pat_set, test_pat_set = patient_split(all_patients, test_size=0.1, random_state=cfg.ts.rng_seed)
    train_patients = sorted(list(train_pat_set))
    test_patients = sorted(list(test_pat_set))

    # Balanced patient selection
    df_train_long = df[
        df["patient_id"].isin(train_patients) & df["year"].isin(train_years)
    ][["patient_id", "year", "event", "date"]].drop_duplicates()

    patients_df = build_patient_availability_table(df_train_long, train_patients)
    pos_sel, neg_sel, n_each = select_full_balanced_patients(patients_df, rng_seed=cfg.ts.rng_seed)
    selected_train_patients = pos_sel + neg_sel
    logger.info("Selected balanced patients per class: %d", n_each)

    # Feature selection
    sel_cfg = SelectionConfig(
        rng_seed=cfg.ts.rng_seed,
        final_top_k=cfg.ts.final_top_k,
        m_candidates=cfg.ts.m_candidates,
        train_years_forced_start=cfg.ts.train_year_start or 1997,
        train_years_forced_end=cfg.ts.train_year_end or 2006,
    )
    lgb_cfg = LGBSelectionConfig(params=default_lgb_params(cfg.ts.rng_seed), num_boost_round=200)

    top_k_events = select_top_events_via_lgbm(
        df=df,
        selected_train_patients=selected_train_patients,
        train_years=train_years,
        cfg=sel_cfg,
        lgb_cfg=lgb_cfg,
    )
    logger.info("Top-k selected events: %d", len(top_k_events))

    # Build final rows
    train_rows, test_rows = ts_build_train_test_rows(
        df=df,
        selected_train_patients=selected_train_patients,
        candidate_test_patients=test_patients,
        train_years=train_years,
        test_years=test_years,
        top_k_events=top_k_events,
    )

    prep_dir = ensure_dir(Path(results_dir) / "prep_artifacts_ts")
    save_csv(train_rows, prep_dir / "train_rows_temporal.csv", index=False)
    save_csv(test_rows, prep_dir / "test_rows_temporal.csv", index=False)
    save_csv(pd.DataFrame({"event": top_k_events}), prep_dir / "top_k_events.csv", index=False)

    # Sequence construction
    feature_cols = list(top_k_events)
    X_train_seq, y_train_seq, year_train_seq, pid_train_seq = build_temporal_sequences(
        train_rows, feature_cols=feature_cols, lookback=cfg.ts.lookback
    )
    X_test_seq, y_test_seq, year_test_seq, pid_test_seq = build_temporal_sequences(
        test_rows, feature_cols=feature_cols, lookback=cfg.ts.lookback
    )

    train_ds = TemporalSequenceDataset(X_train_seq, y_train_seq, year_train_seq, pid_train_seq)
    test_ds = TemporalSequenceDataset(X_test_seq, y_test_seq, year_test_seq, pid_test_seq)

    device = resolve_device(cfg.ts.use_cuda_if_available)
    base_loader_cfg = LoaderConfig(
        batch_size=cfg.ts.batch_size,
        num_workers=cfg.ts.num_workers,
        pin_memory=device == "cuda",
    )
    train_cfg = TrainConfig(
        device=device,
        epochs=cfg.ts.epochs,
        lr=cfg.ts.lr,
        weight_decay=cfg.ts.weight_decay,
        grad_clip=cfg.ts.grad_clip,
        early_stop_patience=cfg.ts.early_stop_patience,
    )

    logger.info("Device: %s | Models: %s", device, cfg.ts.model_names)

    for model_name in cfg.ts.model_names:
        run_id = f"{model_name}_default"
        run_dir = ensure_dir(Path(results_dir) / run_id)

        train_loader = build_model_aware_loader(train_ds, model_name=model_name, base_cfg=base_loader_cfg, train=True)
        test_loader = build_model_aware_loader(test_ds, model_name=model_name, base_cfg=base_loader_cfg, train=False)

        model = build_model(model_name=model_name, seq_len=X_train_seq.shape[1], n_features=X_train_seq.shape[2])

        logger.info("Training %s", run_id)

        if device == "cuda":
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        else:
            import time
            cpu_t0 = time.perf_counter()

        model = train_one_model(model=model, train_loader=train_loader, y_train_all=y_train_seq, cfg=train_cfg, loss_name="bce")

        if device == "cuda":
            t1.record()
            torch.cuda.synchronize()
            train_time_sec = float(t0.elapsed_time(t1) / 1000.0)
        else:
            import time
            train_time_sec = float(time.perf_counter() - cpu_t0)

        if device == "cuda":
            i0 = torch.cuda.Event(enable_timing=True)
            i1 = torch.cuda.Event(enable_timing=True)
            i0.record()
        else:
            import time
            cpu_i0 = time.perf_counter()

        logits, y_true, years, pids = predict_logits(model, test_loader, device=device)

        if device == "cuda":
            i1.record()
            torch.cuda.synchronize()
            infer_time_sec = float(i0.elapsed_time(i1) / 1000.0)
        else:
            import time
            infer_time_sec = float(time.perf_counter() - cpu_i0)

        eval_df = evaluate_per_year(y_true, logits, years)

        artifacts = RunArtifacts(
            run_id=run_id,
            model_name=model_name,
            model_class=model.__class__.__name__,
            train_time_sec=train_time_sec,
            infer_time_sec=infer_time_sec,
            lookback=cfg.ts.lookback,
            n_features=X_train_seq.shape[2],
            n_train_samples=len(train_ds),
            n_test_samples=len(test_ds),
        )
        save_run_outputs(run_dir=run_dir, eval_df=eval_df, pids=pids, years=years, y_true=y_true, logits=logits, artifacts=artifacts)
        logger.info("Finished %s | train=%.2fs infer=%.2fs", run_id, train_time_sec, infer_time_sec)

    yearly_df, summary_df = aggregate_all_runs(results_dir)
    if yearly_df is not None and summary_df is not None:
        logger.info("Aggregation done: yearly=%s summary=%s", len(yearly_df), len(summary_df))

    logger.info("TS pipeline finished.")


# -----------------------------------------------------------------------------
# TabPFN pipeline orchestration
# -----------------------------------------------------------------------------
def run_tabpfn_pipeline(cfg_path: str, logger_name: str = "renal_framework.tabpfn") -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg.tabpfn.rng_seed)

    logger = get_logger(logger_name)
    results_dir = ensure_dir(cfg.paths.results_dir)
    add_file_handler(logger, Path(results_dir) / "tabpfn_pipeline.log")

    logger.info("Starting TabPFN pipeline | config=%s", cfg_path)

    # 1) Load + preprocess rows
    df_raw = load_feather(cfg.paths.data_feather)
    df = tab_canonicalize_event_df(df_raw)

    prep_cfg = TabPFNPrepConfig(
        rng_seed=cfg.tabpfn.rng_seed,
        target_pos_lines=5000,
        target_neg_lines=5000,
        max_total_rows=10000,
        final_top_k=cfg.ts.final_top_k,
        m_candidates=cfg.ts.m_candidates,
        forced_train_year_start=cfg.ts.train_year_start or 1997,
        forced_train_year_end=cfg.ts.train_year_end or 2006,
    )

    prep_out = prepare_tabpfn_rows(df=df, cfg=prep_cfg, lgb_params=default_lgb_params(cfg.tabpfn.rng_seed))
    train_rows = prep_out["train_rows"]
    test_rows = prep_out["test_rows"]
    top_k_events = prep_out["top_k_events"]
    train_years = prep_out["train_years"]

    # 2) Fit drift-resilient TabPFN + yearly eval
    X_train_np = train_rows[top_k_events].to_numpy(dtype=np.float32, copy=True)
    y_train_np = (train_rows["DEATH"] > 0).astype(int).to_numpy(copy=True)
    years_train_np = train_rows["year"].astype(int).to_numpy(copy=True)

    eval_cfg = TabPFNEvalConfig(
        rng_seed=cfg.tabpfn.rng_seed,
        tabpfn_model_name=cfg.tabpfn.model_name,
        batch_size_predict=cfg.tabpfn.default_batch_predict,
        run_id="tabpfn_pipeline",
    )

    fit_out = fit_drift_resilient_tabpfn(
        X_train_np=X_train_np,
        y_train_np=y_train_np,
        train_years=years_train_np,
        eval_cfg=eval_cfg,
    )
    drift_model = fit_out["model"]
    model_add_x_device = fit_out["model_add_x_device"]
    example_add_shape = fit_out["example_add_shape"]

    wf = walkforward_evaluate_tabpfn(
        drift_model=drift_model,
        test_rows=test_rows,
        top_k_events=top_k_events,
        train_years=train_years,
        model_add_x_device=model_add_x_device,
        batch_size_predict=cfg.tabpfn.default_batch_predict,
        example_add_shape=example_add_shape,
    )
    results_per_year = wf["results_per_year"]
    year_to_domain_combined = wf["year_to_domain_combined"]
    test_rows_checked = wf["test_rows_checked"]

    out_dir = ensure_dir(Path(results_dir) / "tabpfn_pipeline_outputs")
    save_tabpfn_temporal_artifacts(
        out_dir=out_dir,
        results_per_year=results_per_year,
        train_rows=train_rows,
        test_rows=test_rows_checked,
        meta={"fit_time_sec": fit_out["fit_time_sec"]},
    )

    # 3) Embeddings + scaling
    feature_cols = list(top_k_events)
    X_test_np = test_rows_checked[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_test_np = (test_rows_checked["DEATH"] > 0).astype(int).to_numpy(copy=True)
    years_test_np = test_rows_checked["year"].astype(int).to_numpy(copy=True)

    # feature scaler parity
    _, _, _ = fit_or_load_feature_scaler(
        train_rows=train_rows,
        test_rows=test_rows_checked,
        feature_cols=feature_cols,
        scaler_path=Path(results_dir) / "scaler_standard.pkl",
    )

    emb_cfg = EmbeddingExtractConfig(
        batch_size=cfg.tabpfn.default_batch_predict,
        max_extract=None,
        use_cache=True,
    )
    emb_out = load_or_extract_embeddings(
        model=drift_model,
        X_train_np=X_train_np,
        X_test_np=X_test_np,
        years_train=years_train_np,
        years_test=years_test_np,
        year_to_domain_map=year_to_domain_combined,
        embeddings_dir=Path(results_dir) / "embeddings",
        cfg=emb_cfg,
        device=model_add_x_device,
        example_add_shape=example_add_shape,
    )
    train_emb_flat = emb_out["train_emb_flat"]
    test_emb_flat = emb_out["test_emb_flat"]

    scaler_emb = fit_embedding_scaler(train_emb_flat)
    train_emb_flat_norm = transform_embeddings(scaler_emb, train_emb_flat)
    test_emb_flat_norm = transform_embeddings(scaler_emb, test_emb_flat)

    # 4) Test split protocol
    split_idx = temporal_test_subsplits(y_test_np, random_state=cfg.tabpfn.rng_seed)
    idx_test_discover = split_idx["idx_test_discover"]
    idx_test_cav_train = split_idx["idx_test_cav_train"]
    idx_test_tcav_eval = split_idx["idx_test_tcav_eval"]
    idx_test_held_out = split_idx["idx_test_held_out"]

    y_test_cav_train = y_test_np[idx_test_cav_train]
    y_test_tcav_eval = y_test_np[idx_test_tcav_eval]
    y_test_held_out = y_test_np[idx_test_held_out]
    years_test_tcav_eval = years_test_np[idx_test_tcav_eval]
    years_test_held_out = years_test_np[idx_test_held_out]

    # 5) Concept decomposition (DL + SAE)
    dl_info = fit_dictionary_learning(
        embeddings_discovery=test_emb_flat_norm[idx_test_discover],
        embeddings_cav_train=test_emb_flat_norm[idx_test_cav_train],
        cfg=DictionaryLearningConfig(
            n_components=cfg.tabpfn.n_factors,
            transform_algorithm="lasso_lars",
            max_iter=1000,
            random_state=cfg.tabpfn.rng_seed,
        ),
    )

    sae_info = fit_sae(
        embeddings_discovery=test_emb_flat_norm[idx_test_discover],
        cfg=SAEConfig(
            emb_dim=train_emb_flat_norm.shape[1],
            n_factors=int(192 * 1.5),
            alpha_sparse=1e-1,
            lr=1e-3,
            epochs=1000,
            use_decoder_bias=False,
            device="cpu",
        ),
        verbose_every=50,
    )

    analysis_source = "sae"  # default notebook parity

    # 6) Rule extraction (for SAE rule-conditioned CAVs)
    rng = np.random.default_rng(42)
    n_cav = len(idx_test_cav_train)
    idx_local = np.arange(n_cav)
    idx_dt_train = rng.choice(idx_local, size=int(n_cav * 0.5), replace=False)
    idx_cav_final_train = np.setdiff1d(idx_local, idx_dt_train)

    X_cav_train_df = test_rows_checked.iloc[idx_test_cav_train][feature_cols].reset_index(drop=True)
    X_feat_dt_train = X_cav_train_df.iloc[idx_dt_train].to_numpy()

    codes_cav_all = get_concept_activations(
        embeddings=test_emb_flat_norm[idx_test_cav_train],
        source="sae",
        model_sae=sae_info["model_sae"],
        device="cpu",
    )
    codes_dt_train = codes_cav_all[idx_dt_train]
    codes_cav_final_train = codes_cav_all[idx_cav_final_train]

    rules_cfg = RuleExtractionConfig(
        percentiles=[90, 80, 70, 60, 50],
        max_depth=15,
        min_samples_leaf=0.01,
        min_positive_samples=50,
        min_rule_precision=0.90,
        min_rule_recall=0.25,
        random_state=42,
    )
    rules_per_percentile, _ = extract_rules_per_percentile(
        X_feat_dt_train=X_feat_dt_train,
        factor_scores_dt_train=codes_dt_train,
        feature_cols=feature_cols,
        cfg=rules_cfg,
    )
    best_p = select_best_percentile(rules_per_percentile)

    # 7) Train CAVs (DL + SAE-rule-conditioned)
    cav_dict_activ = train_cavs_from_activations(
        embeddings_cav_train=dl_info["embeddings_cav_train"].astype(np.float32),
        activations_cav_train=dl_info["activations_cav_train"],
        y_cav_train=y_test_cav_train,
        scaler_emb=scaler_emb,
        pure_quantile=cfg.tabpfn.pure_quantile,
        min_pure=10,
        random_state=cfg.tabpfn.rng_seed,
    )

    emb_cav_all = test_emb_flat_norm[idx_test_cav_train]
    emb_cav_final_train = emb_cav_all[idx_cav_final_train]
    y_cav_final_train = y_test_cav_train[idx_cav_final_train]
    X_cav_final_df = X_cav_train_df.iloc[idx_cav_final_train].reset_index(drop=True)

    cav_dict_sae = train_sae_cavs_from_rules(
        rules_for_percentile=rules_per_percentile[best_p],
        X_cav_train_df=X_cav_final_df,
        cav_train_embeddings_np=emb_cav_final_train.astype(np.float32),
        codes_cav_train_sae=codes_cav_final_train,
        y_cav_train=y_cav_final_train,
        feature_cols=feature_cols,
        scaler_emb=scaler_emb,
        pure_quantile_sae=1.0 - (best_p / 100.0),
        min_pure_sae=50,
        random_state=42,
    )

    cav_dict = cav_dict_sae if analysis_source == "sae" else cav_dict_activ

    # 8) True gradients + TCAV
    model_proc = drift_model.model_processed_ if hasattr(drift_model, "model_processed_") else drift_model.model_proc
    decoder = model_proc.decoder_dict["standard"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = decoder.to(device).eval()

    X_tcav_raw = X_test_np[idx_test_tcav_eval]
    dist_tcav = np.array([year_to_domain_combined[int(y)] for y in years_test_tcav_eval], dtype=np.int64)

    all_grads_chunks: List[np.ndarray] = []
    batch_tcav = 128

    for s in range(0, len(X_tcav_raw), batch_tcav):
        e = min(s + batch_tcav, len(X_tcav_raw))
        xb = X_tcav_raw[s:e].astype(np.float32)
        db = dist_tcav[s:e]
        dist_t = torch.tensor(db, dtype=torch.long, device="cpu").reshape(-1, 1, 1)

        with torch.enable_grad():
            out = drift_model.get_embeddings(xb, additional_x={"dist_shift_domain": dist_t})
            if isinstance(out, torch.Tensor):
                t = out
            else:
                t = torch.tensor(out, dtype=torch.float32)

            if t.ndim == 3 and t.shape[0] == 1:
                t = t[0]
            elif t.ndim == 3 and t.shape[1] == 1:
                t = t.squeeze(1)

            inp = t.clone().detach().to(device, dtype=torch.float32).requires_grad_(True)
            grads = compute_decoder_gradients_vmap(decoder, inp, target_class=1)
            all_grads_chunks.append(grads.detach().cpu().numpy())

    all_grads = np.vstack(all_grads_chunks)

    results_dl = compute_tcav_from_gradients(all_grads, cav_dict_activ)
    results_sae = compute_tcav_from_gradients(all_grads, cav_dict_sae)

    # robust TCAV test on selected analysis source cav_dict
    robust_tcav_results = {}
    for cid, info in cav_dict.items():
        if not isinstance(cid, int):
            continue
        idx_concept = np.asarray(info["pos_idx"])
        idx_non_concept = np.asarray(info["neg_idx"])

        # embeddings used for robust CAV resampling should align with indices
        if analysis_source == "sae":
            robust_embeddings = emb_cav_final_train.astype(np.float32)
        else:
            robust_embeddings = dl_info["embeddings_cav_train"].astype(np.float32)

        res = robust_tcav_significance_test(
            concept_id=cid,
            embeddings=robust_embeddings,
            idx_concept=idx_concept,
            idx_non_concept=idx_non_concept,
            gradients=all_grads,
            scaler_emb=scaler_emb,
            n_runs=cfg.tabpfn.n_random_runs,
            sample_fraction=cfg.tabpfn.concept_sample_fraction,
            seed_base=cfg.tabpfn.rng_seed,
        )
        robust_tcav_results[cid] = res

    filtered_factors, _, _ = filter_significant_factors_by_tcav(
        robust_tcav_results=robust_tcav_results,
        min_distance_from_half=0.1,
    )

    matched_df = build_matched_rules_tcav_table(
        rules_per_percentile=rules_per_percentile,
        best_p=best_p,
        filtered_factors=filtered_factors,
        robust_tcav_results=robust_tcav_results,
    )
    matched_factors = sorted(matched_df["Factor"].unique().tolist()) if len(matched_df) else []

    # 9) Phenotype
    if analysis_source == "dl":
        tcav_eval_acts = get_concept_activations(
            embeddings=test_emb_flat_norm[idx_test_tcav_eval], source="dl", dict_learning_info=dl_info
        )
        held_out_acts = get_concept_activations(
            embeddings=test_emb_flat_norm[idx_test_held_out], source="dl", dict_learning_info=dl_info
        )
        cav_train_acts = get_concept_activations(
            embeddings=test_emb_flat_norm[idx_test_cav_train], source="dl", dict_learning_info=dl_info
        )
    else:
        tcav_eval_acts = get_concept_activations(
            embeddings=test_emb_flat_norm[idx_test_tcav_eval], source="sae", model_sae=sae_info["model_sae"], device="cpu"
        )
        held_out_acts = get_concept_activations(
            embeddings=test_emb_flat_norm[idx_test_held_out], source="sae", model_sae=sae_info["model_sae"], device="cpu"
        )
        cav_train_acts = get_concept_activations(
            embeddings=test_emb_flat_norm[idx_test_cav_train], source="sae", model_sae=sae_info["model_sae"], device="cpu"
        )

    X_test_df = test_rows_checked[feature_cols].copy()
    X_cav_train_raw = X_test_df.to_numpy()[idx_test_cav_train]
    X_tcav_eval_raw = X_test_df.to_numpy()[idx_test_tcav_eval]
    X_held_out_raw = X_test_df.to_numpy()[idx_test_held_out]

    feature_association_results, feature_association_results_test, fa_consistency_df = run_feature_association_dual_split(
        matched_factors=matched_factors,
        tcav_eval_concept_activations=tcav_eval_acts,
        held_out_concept_activations=held_out_acts,
        X_tcav_eval_raw=X_tcav_eval_raw,
        X_held_out_raw=X_held_out_raw,
        feature_cols=feature_cols,
        quantile=0.1,
    )

    sparse_readout_results, sparse_readout_validation, sparse_summary_df = run_sparse_readout_dual_split(
        matched_factors=matched_factors,
        cav_train_concept_activations=cav_train_acts,
        tcav_eval_concept_activations=tcav_eval_acts,
        held_out_concept_activations=held_out_acts,
        X_cav_train_raw=X_cav_train_raw,
        X_tcav_eval_raw=X_tcav_eval_raw,
        X_held_out_raw=X_held_out_raw,
        feature_cols=feature_cols,
        cv=5,
        overfit_drop_warn_threshold=0.2,
    )

    outcome_df_primary, outcome_df_validation, robust_concepts, outcome_consistency_df = run_outcome_association_dual_split(
        matched_factors=matched_factors,
        tcav_eval_concept_activations=tcav_eval_acts,
        held_out_concept_activations=held_out_acts,
        y_tcav_eval=y_test_tcav_eval,
        y_test_labels=y_test_held_out,
        quantile=0.1,
    )

    phenotype_report_df = build_phenotype_report(
        matched_factors=matched_factors,
        sparse_readout_results=sparse_readout_results,
        sparse_validation_results={k: v["test"] for k, v in sparse_readout_validation.items()},
        feature_assoc_results=feature_association_results,
        outcome_df_primary=outcome_df_primary,
        outcome_df_validation=outcome_df_validation,
    )

    # 10) ACE
    ace_embeddings_eval = test_emb_flat_norm[idx_test_held_out]
    ace_domain_ids_eval = np.array([year_to_domain_combined[int(y)] for y in years_test_held_out], dtype=np.int64)

    ace_concept_acts = held_out_acts
    ace_results = run_ace_for_concepts(
        concept_ids=matched_factors,
        cav_dict=cav_dict,
        embeddings_eval=ace_embeddings_eval,
        concept_activations_eval=ace_concept_acts,
        domain_ids_eval=ace_domain_ids_eval,
        decoder=decoder,
        device=device,
    )

    validated, partial, failed = ace_validation_summary(ace_results)
    ace_df = build_ace_summary_df(ace_results, phenotype_df=phenotype_report_df)

    # Save outputs
    save_json(
        _json_safe(
            {
                "analysis_source": analysis_source,
                "best_percentile": int(best_p),
                "matched_factors": matched_factors,
                "n_validated_ace": len(validated),
                "n_partial_ace": len(partial),
                "n_failed_ace": len(failed),
                "robust_concepts": robust_concepts,
            }
        ),
        out_dir / "summary.json",
    )

    save_csv(pd.DataFrame(results_per_year), out_dir / "evaluation_by_year.csv", index=False)
    save_csv(robust_tcav_results_to_df(robust_tcav_results), out_dir / "robust_tcav_results.csv", index=False)
    save_csv(tcav_results_to_df(results_sae, cav_dict_sae), out_dir / "tcav_sae.csv", index=False)
    save_csv(tcav_results_to_df(results_dl, cav_dict_activ), out_dir / "tcav_dl.csv", index=False)
    save_csv(matched_df, out_dir / "matched_rules_tcav.csv", index=False)

    save_csv(fa_consistency_df, out_dir / "feature_assoc_consistency.csv", index=False)
    save_csv(sparse_summary_df, out_dir / "sparse_readout_summary.csv", index=False)
    save_csv(outcome_df_primary, out_dir / "outcome_association_primary.csv", index=False)
    save_csv(outcome_df_validation, out_dir / "outcome_association_validation.csv", index=False)
    save_csv(outcome_consistency_df, out_dir / "outcome_consistency.csv", index=False)
    save_csv(phenotype_report_df, out_dir / "phenotype_report.csv", index=False)
    save_csv(ace_df, out_dir / "ace_results.csv", index=False)

    # Optional full json dumps
    save_json(_json_safe(robust_tcav_results), out_dir / "robust_tcav_results_full.json")
    save_json(
        _json_safe(
            {
                "validated": [x["concept_id"] for x in validated],
                "partial": [x["concept_id"] for x in partial],
                "failed": [x["concept_id"] for x in failed],
            }
        ),
        out_dir / "ace_validation_summary.json",
    )

    logger.info("TabPFN pipeline finished. Outputs saved to %s", out_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Renal Framework Runner")
    parser.add_argument("--pipeline", type=str, choices=["ts", "tabpfn"], required=True, help="Which pipeline to run.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pipeline == "ts":
        run_ts_pipeline(args.config)
    elif args.pipeline == "tabpfn":
        run_tabpfn_pipeline(args.config)
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}")


if __name__ == "__main__":
    main()