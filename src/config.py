"""Global configuration shared across the TCAV refactor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT
ARTIFACTS_DIR = PROJECT_ROOT / "activations_stream"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Seeds / randomness defaults
RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# Modeling defaults
DEFAULT_BATCH_SIZE_PREDICT = 512
TRAIN_YEAR_FRAC = 0.7
SAFE_NUM_WORKERS = 0

# Plotting defaults
PLOT_STYLE = "whitegrid"

def get_data_path(filename: str) -> Path:
    """Return an absolute path to a default data file."""
    return DATA_DIR / filename
