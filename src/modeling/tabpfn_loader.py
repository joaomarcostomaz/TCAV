"""Utilities to install and instantiate TabPFN models."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import os

try:
    import tabpfn
    from tabpfn.scripts.estimator.base import TabPFNClassifier
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
    # TABPFN_AVAILABLE = True
except Exception:
    from tabpfn.scripts.estimator.base import TabPFNClassifier  # caso a import acima falhe, tentar apenas a classe
    # try:
    #     # TABPFN_AVAILABLE = True
    # except Exception:
        # TABPFN_AVAILABLE = False

from src import config


def load_tabpfn_model() -> Any:
    """Instantiate a TabPFN model mirroring the resilient logic from the notebook."""

    if tabpfn is None:
        raise RuntimeError(
            "TabPFN package is not available in this environment.")
    # try:
    print("Loading TabPFN model using best_dist configuration.")
    libpath = Path(tabpfn.__file__).parents[0]
    model_path_config = TabPFNModelPathsConfig(
        paths=[f"{libpath}/model_cache/tabpfn_dist_model_1.cpkt"],
        task_type="dist_shift_multiclass",
    )
    model = get_best_tabpfn(
        task_type="dist_shift_multiclass",
        model_type="best_dist",
        paths_config=model_path_config,
        debug=False,
        device="auto",
    )
    model.show_progress = False
    model.seed = config.RNG_SEED
    return model

    # except Exception:
    #     print("Falling back to default TabPFNClassifier instantiation.")
    #     if TabPFNClassifier is None:
    #         raise
    #     model = TabPFNClassifier(device="auto")
    #     model.show_progress = False
    #     model.seed = config.RNG_SEED
    #     return model
