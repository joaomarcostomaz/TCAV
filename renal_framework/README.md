# Renal Framework

A modular Python framework to replace notebook-centric research code for:

- Temporal deep models (TSMixer, DLinear, iTransformer, PatchTST, NBEATSx)
- TabPFN-based embedding/concept interpretability
- TCAV significance testing
- Phenotype characterization
- ACE-style necessity/sufficiency and drift checks

The goal is to preserve research logic while making experimentation, testing, and iteration much easier.

---

## 1) Project Structure

```text
renal_framework/
├── run.py
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── ts_experiments.yaml
│   └── tabpfn_experiments.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   ├── logging_utils.py
│   ├── metrics.py
│   ├── plotting.py
│   ├── splits.py
│   ├── ts_pipeline/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── interpretability.py
│   │   ├── ccta.py
│   │   └── cctsi.py
│   └── tabpfn_pipeline/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── embedding.py
│       ├── concept_learning.py
│       ├── tcav.py
│       ├── phenotype.py
│       └── ace.py
└── tests/
    ├── __init__.py
    ├── test_splits.py
    └── test_metrics.py
```

---

## 2) Installation

### 2.1 Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

### 2.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 2.3 Install package in editable mode

```bash
pip install -e .
```

---

## 3) Configuration

All runtime parameters are centralized in YAML and dataclasses:

- `configs/base.yaml` – general defaults
- `configs/ts_experiments.yaml` – TS-focused overrides
- `configs/tabpfn_experiments.yaml` – TabPFN-focused overrides

No experiment constants should be hardcoded inside pipeline modules.

---

## 4) Running Pipelines

## 4.1 Time-Series Pipeline

```bash
python run.py --pipeline ts --config configs/ts_experiments.yaml
```

This runs:

1. Data loading + canonical preprocessing
2. Temporal train/test year split
3. Patient split and balancing
4. LGB-based top-k event selection
5. Sequence building
6. Model training/evaluation for configured `model_names`
7. Artifact saving and run aggregation

Output goes to `paths.results_dir` from YAML.

### 4.2 TabPFN Pipeline

```bash
python run.py --pipeline tabpfn --config configs/tabpfn_experiments.yaml
```

**Important:** In `run.py`, `run_tabpfn_pipeline()` intentionally has a model-loading hook marked `NotImplementedError` because TabPFN runtime APIs differ by version/environment.

You should paste your exact notebook model-loading/fitting block there (e.g., `get_best_tabpfn` path-based loading or `TabPFNClassifier`) and then continue with the provided modular steps.

---

## 5) Testing

Run unit tests:

```bash
pytest
```

Current test coverage includes:

- split utilities (`tests/test_splits.py`)
- metrics/threshold utilities (`tests/test_metrics.py`)

Add new tests whenever you add or modify pipeline logic.

---

## 6) Artifact Conventions

TS pipeline writes per-run folders:

- `evaluation_by_year.csv`
- `test_predictions.csv`
- `run_config.json`

and aggregate files at results root:

- `all_runs_evaluation_by_year.csv`
- `all_runs_summary.csv`

TabPFN pipeline should follow the same principle:
small, explicit, reproducible artifacts per stage.

---

## 7) Development Principles

1. **Single responsibility per module/function**
2. **Config-driven experiments**
3. **No hidden globals in implementation**
4. **Pure utility functions where possible**
5. **Reproducibility first** (seed control, deterministic setup)
6. **Research parity over premature optimization**

---

## 8) Migration Notes from Notebooks

This framework is derived from large notebook workflows and intentionally keeps feature parity:
- preprocessing logic
- feature selection protocol
- concept decomposition options (DL/SAE)
- robust TCAV strategy
- phenotype and ACE analyses

During migration, preserve behavior first; refactor for elegance second.

---

## 9) Known TODOs

- Finalize TabPFN model-loading block in `run.py` for your environment.
- Add dedicated orchestrator scripts for:
  - `run_ts_improved.py`
  - `run_tabpfn_tcav.py`
  - `run_ace_validation.py`
- Expand tests to concept/TCAV/ACE modules.
- Add CI (lint + tests).

---

## 10) Quick Start (recommended)

```bash
pip install -r requirements.txt
pip install -e .
python run.py --pipeline ts --config configs/ts_experiments.yaml
pytest
```

Then wire your TabPFN runtime block and run:

```bash
python run.py --pipeline tabpfn --config configs/tabpfn_experiments.yaml
```

---

If you want, next step I can provide:
- a **fully wired `run_tabpfn_pipeline()`** using your exact notebook TabPFN loading strategy, and
- a **`Makefile`** with one-command targets (`make ts`, `make tabpfn`, `make test`).