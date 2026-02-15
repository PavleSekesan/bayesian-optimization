# Bayesian Optimization (Educational Project)

This repository contains a from-scratch Bayesian optimization implementation in Python.

## Scope

- Gaussian-process surrogate model with an RBF kernel
- Acquisition function:
  - Expected Improvement (EI)
- Acquisition maximization with a single SciPy L-BFGS-B call
- BBOB benchmarking scripts using `cocoex` and postprocessing with `cocopp`
- Unit tests for core methods
- LaTeX report in `report/`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install --no-build-isolation -e .[dev,experiments]
```

Note: the COCO experiment module is imported as `cocoex`, but the pip package name is `coco-experiment`.

## Run tests

```bash
pytest
```

## Run BBOB experiments

```bash
python experiments/run_bbob_bo.py --max-budget 200 --output results/bo_ei
python experiments/run_bbob_random.py --max-budget 200 --output results/random
```

## Compare against published methods and generate plots

```bash
python experiments/postprocess.py \
  --input results/bo_ei \
  --archive "CMA-ES" "BFGS" \
  --output ppdata
```

`experiments/postprocess.py` and `experiments/fetch_archives.py` use local runtime directories under `.runtime/` for cache/config data to avoid home-directory permission issues.

`experiments/fetch_archives.py` can also resolve and download matching official archive datasets.

## Repository layout

- `src/bayesopt/`: BO library
- `tests/`: unit tests
- `experiments/`: BBOB and cocopp scripts
- `report/`: LaTeX report
