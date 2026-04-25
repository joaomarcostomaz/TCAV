"""
tabpfn_pipeline package exports
"""

from .preprocessing import (
    TabPFNPrepConfig,
    canonicalize_event_df,
    infer_train_test_years,
    split_patients,
    build_pivot_preserve_presence,
    trim_post_death_rows,
    select_equal_patients_with_line_cap,
    select_top_events_lgbm,
    build_train_test_rows,
    prepare_tabpfn_rows,
)

from .embedding import (
    EmbeddingExtractConfig,
    flatten_embeddings,
    make_dist_tensor,
    extract_embeddings_robust,
    fit_embedding_scaler,
    transform_embeddings,
    temporal_test_subsplits,
)

from .concept_learning import (
    DictionaryLearningConfig,
    SAEConfig,
    SparseAutoencoderTied,
    fit_dictionary_learning,
    transform_dictionary_learning,
    fit_sae,
    transform_sae,
    get_concept_activations,
)

from .tcav import (
    CAV,
    TCAVGradientResult,
    parse_decision_tree_rule,
    apply_rule_conditions,
    train_cavs_from_activations,
    compute_decoder_gradients_vmap,
    compute_tcav_from_gradients,
    robust_tcav_significance_test,
)

from .phenotype import (
    compute_feature_associations,
    sparse_readout,
    evaluate_sparse_readout,
    compute_outcome_association,
    build_phenotype_report,
)

from .ace import (
    ACESanityResult,
    project_out_direction,
    project_onto_direction,
    run_embedding_destruction_test,
    run_embedding_sufficiency_test,
    compute_drift_metrics,
    run_tests_by_domain,
    run_ace_for_concepts,
    ace_validation_summary,
)