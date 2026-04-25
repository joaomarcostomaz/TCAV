"""
Central configuration objects and YAML loading utilities.

All experiment parameters should be defined here (or in YAML),
never hardcoded inside pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml


# -----------------------------
# Path-level configuration
# -----------------------------
@dataclass
class PathsConfig:
    root_dir: Path = Path.cwd()
    data_feather: Path = Path("tidy_event_data.feather")
    results_dir: Path = Path("results")
    external_dir: Path = Path("external")


# -----------------------------
# Time-series pipeline config
# -----------------------------
@dataclass
class TSConfig:
    rng_seed: int = 42

    # Data and feature selection
    final_top_k: int = 500
    m_candidates: int = 3000
    train_year_start: Optional[int] = 1997
    train_year_end: Optional[int] = 2006  # inclusive if available in data

    # Training
    lookback: int = 5
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 4
    grad_clip: float = 1.0

    # Execution
    num_workers: int = 0
    use_cuda_if_available: bool = True
    model_names: List[str] = field(default_factory=lambda: ["tsmixer", "dlinear", "itransformer"])


# -----------------------------
# TabPFN / concept pipeline config
# -----------------------------
@dataclass
class TabPFNConfig:
    rng_seed: int = 42
    model_name: str = "tabpfn_dist_model_1"
    default_batch_predict: int = 512
    n_factors: int = 8
    pure_quantile: float = 0.05
    n_random_runs: int = 15
    concept_sample_fraction: float = 1.0


# -----------------------------
# Global app config
# -----------------------------
@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    ts: TSConfig = field(default_factory=TSConfig)
    tabpfn: TabPFNConfig = field(default_factory=TabPFNConfig)


def _merge_dataclass(dc_obj: Any, raw: Dict[str, Any]) -> Any:
    """Update dataclass fields from a raw dictionary (only matching keys)."""
    for k, v in raw.items():
        if hasattr(dc_obj, k):
            setattr(dc_obj, k, v)
    return dc_obj


def load_config(config_path: str | Path) -> AppConfig:
    """
    Load YAML config into AppConfig dataclasses.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML file.

    Returns
    -------
    AppConfig
        Fully merged config object.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppConfig()

    if "paths" in raw:
        p = raw["paths"]
        cfg.paths = PathsConfig(
            root_dir=Path(p.get("root_dir", cfg.paths.root_dir)),
            data_feather=Path(p.get("data_feather", cfg.paths.data_feather)),
            results_dir=Path(p.get("results_dir", cfg.paths.results_dir)),
            external_dir=Path(p.get("external_dir", cfg.paths.external_dir)),
        )

    if "ts" in raw:
        cfg.ts = _merge_dataclass(cfg.ts, raw["ts"])

    if "tabpfn" in raw:
        cfg.tabpfn = _merge_dataclass(cfg.tabpfn, raw["tabpfn"])

    return cfg