# TCAV

Refactored TabPFN + TCAV workflow extracted from the monolithic `tcav.ipynb` notebook. The codebase now exposes reusable Python packages, a command-line pipeline runner, and two curated notebooks for exploration.

## Project structure

- `src/` – modular Python package (data ingestion/prep, modeling, embeddings, concepts, reporting, pipelines, utils).
- `scripts/run_full_tcav.py` – CLI entry point that executes the entire pipeline headlessly.
- `notebooks/`
	- `tcav_refactored.ipynb`: step-by-step walkthrough that mirrors the legacy workflow using the new modules.
	- `tutorial_tabpfn_tcav.ipynb`: quick-start demo that configures and runs the pipeline in a few cells.
- `docs/refactor_overview.md` – running log of the refactor plan, status, and pending tasks.

## Setup & Installation

### Prerequisites
- Python 3.10 (recommended)
- pip or uv package manager

### Step-by-step Installation

1. **Create and activate a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies in the correct order**:
```bash
# First, install numpy<2 (critical for compatibility)
pip install "numpy<2.0.0"

# Install drift-resilient-tabpfn from GitHub
pip install git+https://github.com/automl/Drift-Resilient_TabPFN.git

# Install opencv with compatible version
pip install "opencv-contrib-python<4.10"

# Install remaining dependencies
pip install -r requirements.txt
```

3. **Apply the critical TabPFN patch**:

The drift-resilient-tabpfn package has a missing function that needs to be manually added. Run:
```bash
cat >> .venv/lib/python3.10/site-packages/tabpfn/utils.py << 'EOF'


def print_on_master_only(msg):
    """Print message only on master process (for distributed training compatibility)."""
    print(msg)
EOF
```

**Note**: This patch is required because the drift-resilient-tabpfn package expects a `print_on_master_only` function in `tabpfn.utils` that doesn't exist in the package. This is a known issue with the package.

### Verifying the Installation

Test that everything is working:
```bash
python -c "
from tabpfn.utils import print_on_master_only, skew, hash_tensor
from tabpfn.scripts.estimator.base import TabPFNClassifier
from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
print('✓ All imports successful!')
"
```

## Usage

1. **Run the pipeline via CLI**:
```bash
python scripts/run_full_tcav.py --plots --summary-json outputs/tcav_summary.json
```

2. **Explore notebooks**:
   - Open `notebooks/tcav_refactored.ipynb` for step-by-step walkthrough
   - Open `notebooks/tutorial_tabpfn_tcav.ipynb` for quick-start demo

   Both notebooks use the modular `src/` package and require the setup steps above.

3. **Inspect docs**: read `docs/refactor_overview.md` for architecture notes and to-dos.

## Important Notes

### Why drift-resilient-tabpfn?

This project requires the **drift-resilient-tabpfn** package (not the official `tabpfn` from PyPI) because:
- It supports the `additional_x` parameter for drift indicators/domain tokens
- The official `tabpfn` package doesn't have this functionality
- This is critical for the temporal drift modeling in this project

### Import Path

Always import `TabPFNClassifier` from:
```python
from tabpfn.scripts.estimator.base import TabPFNClassifier
```

NOT from:
```python
from tabpfn import TabPFNClassifier  # ❌ Wrong path for drift-resilient version
```

### Version Constraints

- **numpy must be < 2.0.0** – The drift-resilient-tabpfn package is not compatible with numpy 2.x
- **opencv-contrib-python < 4.10** – To maintain compatibility with numpy 1.x

The original `tcav.ipynb` remains untouched for historical reference.
