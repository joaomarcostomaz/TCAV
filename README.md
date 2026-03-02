# Mechanistic Dynamic Interpretability for Tabular Foundation Models in Healthcare

TCAV-based mechanistic interpretability applied to renal transplant outcome
prediction using Drift-Resilient TabPFN.

This repository accompanies the paper
_"Mechanistic Dynamic Interpretability for Tabular Foundation Models in Healthcare"_
and contains a single end-to-end Jupyter notebook that reproduces every result
reported in the manuscript.

## Repository layout

```
notebooks/
  renal_mechanistic_dynamic_interpretability_final.ipynb   # Main analysis notebook with renal dataset

pyproject.toml             # Direct dependencies (uv / pip-compatible)
requirements-lock.txt      # Full pinned environment (uv pip freeze)
.python-version            # Python 3.10.19 pin for uv
REPRODUCIBILITY.md         # Reproducibility checklist and constraints
```

## Prerequisites

- Python 3.10 (tested with 3.10.19).
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip.
- A CUDA-capable GPU is recommended but not required (the notebook falls back
  to CPU).

## Installation

### With uv (recommended)

```bash
# Clone the repository
git clone <repo-url> && cd <repo-dir>

# Create the environment and install all dependencies.
# uv reads .python-version and pyproject.toml automatically.
uv sync

# -- OR, for exact byte-for-byte reproduction --
uv venv
uv pip install -r requirements-lock.txt
```

### With pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

### Post-install patch

We noticed that, in some installations, the `drift-resilient-tabpfn` package missed a helper function that it references internally. If that happens to you, add the following to
`.venv/lib/python3.10/site-packages/tabpfn/utils.py` after installation:

```python
def print_on_master_only(msg):
    """Print message only on master process (for distributed training compatibility)."""
    print(msg)
```

This can be applied in one line:

```bash
echo -e '\ndef print_on_master_only(msg):\n    print(msg)' \
  >> .venv/lib/python3.10/site-packages/tabpfn/utils.py
```

## Usage

The project is entirely notebook-driven. The notebook that uses the renal dataset is not meant to be ran, though, as the dataset will not be shared. Open and run the Myeloma notebook
top-to-bottom:

```
notebooks/myeloma_mechanistic_dynamic_interpretability_final.ipynb
```

The notebook covers the full pipeline:

1. Dataset preparation.
2. Drift-Resilient TabPFN training and walk-forward evaluation.
3. Embedding extraction.
4. Concept decomposition via Dictionary Learning and Sparse Autoencoders.
5. Decision-tree rule extraction for concept labelling.
6. TCAV with true gradient computation and statistical significance testing.
7. Necessity / sufficiency ablation experiments.
8. Temporal stability analysis and integrated reporting.

### Dataset

The renal dataset is confidential and not included in this repository. Cell 4 of the
notebook defines `dataset_path`; point it to a Feather file with columns
`patient_id`, `year`, `event`, and `date`. The sampling, splitting, and
feature-engineering logic adapts to any dataset that follows this schema.

The Myeloma dataset, however, is public and was shared with the Drift-Resilient TabPFN project.

### Renal notebook outputs

The renal notebook ships with all outputs stored. It is designed as a
**read-only artifact** for the paper: re-execution is possible on the original
dataset, but the stored outputs are the canonical results. See
`REPRODUCIBILITY.md` for the full set of constraints.

## References

- Drift-Resilient TabPFN: <https://github.com/automl/Drift-Resilient_TabPFN\>
- TabPFN: Hollmann et al., 2023 (<https://arxiv.org/abs/2207.01848\>\)
- TCAV: Kim et al., 2018 (<https://arxiv.org/abs/1711.11279\>\)
