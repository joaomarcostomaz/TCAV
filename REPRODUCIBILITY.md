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

### Option A -- dependency resolution from pyproject.toml

```bash
uv sync
```

This lets uv resolve from the declared direct dependencies. The result may
differ from the lockfile if upstream packages have released new builds.

### Option B -- installation from requirements-lock.txt

```bash
uv venv                                # creates .venv with Python 3.10.19
uv pip install -r requirements-lock.txt
```

This installs the identical package versions used to produce the paper results.


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

The canonical notebooks are:

```
notebooks/renal_mechanistic_dynamic_interpretability_final.ipynb
notebooks/myeloma_mechanistic_dynamic_interpretability_final.ipynb
```

`myeloma_mechanistic_dynamic_interpretability_final` is the one you should run to reproduce results.