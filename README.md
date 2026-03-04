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
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,experiments]
```

## Run tests

```bash
pytest
```

## Run BBOB experiments

```bash
python experiments/run_bbob_bo.py
python experiments/run_bbob_optuna.py
python experiments/run_bbob_random.py
```

## Postprocess results with cocopp

```bash
python -m cocopp -o ppdata \
  exdata/bayesian-optimization \
  exdata/optuna-gpei \
  exdata/random \
```

Archive comparison helper scripts are not present in the current `experiments/` folder. Use `cocopp` directly or add your own wrapper script.

## Repository layout

- `src/bayesopt/`: BO library
- `tests/`: unit tests
- `experiments/`: BBOB and cocopp scripts
- `report/`: LaTeX report
