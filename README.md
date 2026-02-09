# Bayesian Optimization (Educational Project)

This repository contains a from-scratch Bayesian optimization implementation in Python.

## Scope

- Gaussian-process surrogate model with an RBF kernel
- Acquisition functions:
  - Expected Improvement (EI)
  - Probability of Improvement (PI)
- From-scratch acquisition maximization (random multi-start + local coordinate refinement)
- BBOB benchmarking scripts using `cocoex` and postprocessing with `cocopp`
- Unit tests for core methods
- LaTeX report in `report/`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,experiments]
```

Note: the COCO experiment module is imported as `cocoex`, but the pip package name is `coco-experiment`.

## Run tests

```bash
pytest
```

## Run BBOB experiments

```bash
python experiments/run_bbob_bo.py --acquisition ei --output results/bo_ei
python experiments/run_bbob_bo.py --acquisition pi --output results/bo_pi
python experiments/run_bbob_random.py --output results/random
```

## Compare against published methods and generate plots

```bash
python experiments/postprocess.py \
  --input results/bo_ei results/bo_pi \
  --archive "CMA-ES" "BFGS" \
  --output ppdata
```

`experiments/fetch_archives.py` can also resolve and download matching official archive datasets.

## Repository layout

- `src/bayesopt/`: BO library
- `tests/`: unit tests
- `experiments/`: BBOB and cocopp scripts
- `report/`: LaTeX report
