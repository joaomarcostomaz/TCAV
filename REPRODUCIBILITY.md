# Reproducibility Guide

This document describes how to reproduce the analysis in this repository and
documents the constraints that govern the notebook's code and outputs.

---

## Environment

| Item                    | Value / Constraint                                      |
| ----------------------- | ------------------------------------------------------- |
| Python version          | 3.10.19 (pinned in `.python-version`)                   |
| Package manager         | uv (venv created by uv 0.9.7)                           |
| Direct dependencies     | Listed in `pyproject.toml`                              |
| Full pinned environment | `requirements-lock.txt` (198 packages, `uv pip freeze`) |
| Critical custom package | `drift-resilient-tabpfn` at commit `a6e75af` (GitHub)   |

### Why Python 3.10?

`drift-resilient-tabpfn` and its transitive dependency tree (torch 2.1.2,
numpy < 2.0, etc.) are tested against Python 3.10. Other minor versions in the
3.10 series should work.

### Why numpy < 2.0?

`drift-resilient-tabpfn` includes compiled extensions built against the NumPy
1.x C ABI. Importing under NumPy 2.x causes `RuntimeError` at import time.

---

## Reproducing the environment

### Option A -- exact reproduction (recommended)

```bash
uv venv                                # creates .venv with Python 3.10.19
uv pip install -r requirements-lock.txt
```

This installs the identical package versions used to produce the paper results.

### Option B -- dependency resolution from pyproject.toml

```bash
uv sync
```

This lets uv resolve from the declared direct dependencies. The result may
differ from the lockfile if upstream packages have released new builds.

### Post-install patch

After either option, apply the missing-function patch to
`drift-resilient-tabpfn`:

```bash
echo -e '\ndef print_on_master_only(msg):\n    print(msg)' \
  >> .venv/lib/python3.10/site-packages/tabpfn/utils.py
```

See `README.md` for details.

---

## Verification

After installation, confirm the environment is correct:

```bash
# Python version
python --version                       # expected: Python 3.10.19

# Core imports
python -c "
from tabpfn.utils import print_on_master_only
from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
import numpy, torch, lightgbm, sklearn
print(f'numpy     {numpy.__version__}')
print(f'torch     {torch.__version__}')
print(f'lightgbm  {lightgbm.__version__}')
print(f'sklearn   {sklearn.__version__}')
print('All imports OK')
"
```

Expected output:

```
numpy     1.26.4
torch     2.1.2
lightgbm  3.3.5
sklearn   1.5.2
All imports OK
```

---

## Notebook execution

The canonical notebook is:

```
notebooks/renal_mechanistic_dynamic_interpretability_final.ipynb
```

### With the original dataset

Set `dataset_path` in cell 4 to the location of the Feather file, then run all
cells top-to-bottom. The notebook is deterministic given the same data and
environment (`np.random.seed(42)` and `random_state=42` throughout).

### Without the original dataset

The dataset is confidential. Pre-computed embeddings are provided in
`embeddings_saved/` so that cells 20 onward (embedding loading, concept
decomposition, TCAV, ablations, reporting) can execute without the raw data.
Cells 6-19 (data ingestion, feature engineering, model training) will fail
without the dataset.

---

## Files for reproducibility

| File                    | Purpose                                     |
| ----------------------- | ------------------------------------------- |
| `pyproject.toml`        | Direct dependencies with pinned versions    |
| `requirements-lock.txt` | Full environment snapshot (`uv pip freeze`) |
| `.python-version`       | Python interpreter pin for uv               |

---
