# Reproducibility Checklist

This document ensures anyone can reproduce your TCAV environment and results.

## ✅ What's Been Done

### 1. Environment Setup
- [x] Automated setup script (`setup.sh`)
- [x] Detailed manual instructions (`SETUP.md`)
- [x] Frozen requirements with exact versions (`requirements-frozen.txt`)
- [x] Comprehensive README with setup instructions
- [x] `.gitignore` to exclude unnecessary files

### 2. Code Fixes Applied
- [x] Fixed TabPFN imports in `src/modeling/tabpfn_loader.py`
- [x] Fixed TabPFN imports in `tcav.ipynb` (cell 4)
- [x] Documented the `print_on_master_only` patch requirement
- [x] Updated notebooks to use correct import paths

### 3. Documentation
- [x] Main README with installation steps
- [x] SETUP.md with detailed troubleshooting
- [x] This REPRODUCIBILITY.md checklist
- [x] Inline code comments explaining critical parts

## 📋 Quick Start for New Users

1. Clone the repository
2. Run `./setup.sh` (or follow `SETUP.md` for manual setup)
3. Verify installation works
4. Run notebooks or scripts

## 🔧 Critical Configuration

### Python Version
- **Required**: Python 3.10
- **Tested**: Python 3.10.19
- **Not tested**: Python 3.11+, Python 3.9-

### Key Dependencies

| Package | Version Constraint | Why? |
|---------|-------------------|------|
| numpy | `<2.0.0` | drift-resilient-tabpfn compiled against NumPy 1.x |
| drift-resilient-tabpfn | From GitHub | Supports `additional_x` parameter for drift |
| opencv-contrib-python | `<4.10` | Compatible with NumPy 1.x |
| torch | `<2.2,>=2.1` | Required by drift-resilient-tabpfn |
| scikit-learn | `<1.6,>=1.4.2` | Required by drift-resilient-tabpfn |

### Manual Patch Required

The drift-resilient-tabpfn package has a bug. After installation, add this function to `tabpfn/utils.py`:

```python
def print_on_master_only(msg):
    """Print message only on master process (for distributed training compatibility)."""
    print(msg)
```

**Location**: `.venv/lib/python3.10/site-packages/tabpfn/utils.py`

This is automated in `setup.sh` but needs manual application if installing manually.

## 🎯 Correct Import Paths

### ✅ Correct
```python
from tabpfn.scripts.estimator.base import TabPFNClassifier
from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
from tabpfn.utils import print_on_master_only, skew, hash_tensor
```

### ❌ Incorrect
```python
from tabpfn import TabPFNClassifier  # Wrong for drift-resilient version
```

## 📦 Files for Reproducibility

| File | Purpose | Status |
|------|---------|--------|
| `setup.sh` | Automated setup script | ✅ Created |
| `SETUP.md` | Detailed setup guide | ✅ Created |
| `requirements.txt` | Main dependencies | ✅ Exists |
| `requirements-frozen.txt` | Exact versions snapshot | ✅ Created |
| `README.md` | Project overview + setup | ✅ Updated |
| `.gitignore` | Exclude temp files | ✅ Created |
| `REPRODUCIBILITY.md` | This checklist | ✅ Created |

## 🧪 Verification Steps

After setup, verify everything works:

```bash
# 1. Check Python version
python --version  # Should be 3.10.x

# 2. Verify imports
python -c "
from tabpfn.utils import print_on_master_only, skew, hash_tensor
from tabpfn.scripts.estimator.base import TabPFNClassifier
from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
print('✅ All imports successful!')
"

# 3. Check numpy version
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
# Should be 1.26.x (not 2.x)

# 4. Verify additional_x support
python -c "
from tabpfn.scripts.estimator.base import TabPFNClassifier
import numpy as np
clf = TabPFNClassifier(device='cpu')
X = np.random.randn(50, 5)
y = np.random.randint(0, 2, 50)
clf.fit(X, y, additional_x={'dist_shift_domain': np.zeros(50)})
print('✅ additional_x parameter works!')
"
```

## 🐳 Docker Option (Future)

For absolute reproducibility, consider creating a Docker image:

```bash
docker build -t tcav:latest .
docker run -it -v $(pwd):/app tcav:latest
```

A `Dockerfile` template is provided in `SETUP.md`.

## 📊 Data Files

The project requires:
- `free_light_chain_mortality.csv` (included in repo)

## 🚀 Running the Project

### Option 1: Notebooks
```bash
jupyter notebook notebooks/tcav_refactored.ipynb
# or
code notebooks/tcav_refactored.ipynb  # VS Code
```

### Option 2: CLI Script
```bash
python scripts/run_full_tcav.py --plots --summary-json outputs/tcav_summary.json
```

### Option 3: Python Module
```python
from src.pipelines.full_tcav import run_full_tcav_pipeline
results = run_full_tcav_pipeline(
    data_path="./free_light_chain_mortality.csv",
    plots=True
)
```

## ⚠️ Known Issues

1. **NumPy 2.x incompatibility**: Will cause crashes. Must use NumPy < 2.0.0
2. **Missing print_on_master_only**: Requires manual patch (automated in setup.sh)
3. **Wrong import path**: Must use `tabpfn.scripts.estimator.base`, not `tabpfn` directly

## 🔄 Updating Dependencies

If you need to update packages:

```bash
# Update a specific package
pip install --upgrade package-name

# Recreate frozen requirements
pip freeze > requirements-frozen.txt

# Test that everything still works
python -c "from src import config; print('✓ Import test passed')"
```

## 📝 Version History

| Date | Python | NumPy | drift-resilient-tabpfn | Notes |
|------|--------|-------|------------------------|-------|
| 2026-01-14 | 3.10.19 | 1.26.4 | commit a6e75af | Initial reproducible setup |

## 🆘 Getting Help

If setup fails:
1. Check `SETUP.md` troubleshooting section
2. Verify you followed steps in exact order
3. Check Python version is 3.10.x
4. Ensure virtual environment is activated
5. Try the automated `./setup.sh` script

## ✨ Success Criteria

Your setup is successful when:
- [ ] `./setup.sh` runs without errors
- [ ] All verification steps pass
- [ ] Notebooks open and run cell 4 without import errors
- [ ] `python scripts/run_full_tcav.py` executes
- [ ] Model can use `additional_x` parameter

## 📚 References

- drift-resilient-tabpfn: https://github.com/automl/Drift-Resilient_TabPFN
- TabPFN paper: https://arxiv.org/abs/2207.01848
- TCAV paper: https://arxiv.org/abs/1711.11279
