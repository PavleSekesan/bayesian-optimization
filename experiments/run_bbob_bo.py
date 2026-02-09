from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from bayesopt.optimizer import AcquisitionConfig, BayesianOptimizer

try:
    from experiments.common import build_suite_filter, ensure_directory, parse_dimensions, seed_for_problem
except ModuleNotFoundError:
    from common import build_suite_filter, ensure_directory, parse_dimensions, seed_for_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian optimization on COCO BBOB problems.")
    parser.add_argument("--dimensions", default="2,3,5,10", help="Comma-separated dimensions.")
    parser.add_argument("--instances", default="1-5", help="BBOB instance range or list.")
    parser.add_argument("--budget-factor", type=float, default=40.0, help="Budget = factor * dimension.")
    parser.add_argument("--acquisition", choices=["ei", "pi"], default="ei")
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--n-candidates", type=int, default=2048)
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--n-initial", type=int, default=0, help="0 means use library default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/bo")
    return parser.parse_args()


def run_problem(problem: Any, args: argparse.Namespace) -> None:
    bounds = np.column_stack((problem.lower_bounds, problem.upper_bounds)).astype(np.float64)
    budget = max(1, int(args.budget_factor * problem.dimension))
    local_seed = seed_for_problem(args.seed, str(problem.id))

    acq_config = AcquisitionConfig(
        kind=args.acquisition,
        xi=args.xi,
        n_candidates=args.n_candidates,
        n_starts=args.n_starts,
    )

    optimizer = BayesianOptimizer(
        bounds=bounds,
        acquisition=args.acquisition,
        acquisition_config=acq_config,
        seed=local_seed,
    )

    def objective(x: np.ndarray) -> float:
        return float(problem(np.asarray(x, dtype=np.float64)))

    n_initial = None if args.n_initial == 0 else args.n_initial
    optimizer.run(objective=objective, budget=budget, n_initial=n_initial)


def main() -> None:
    args = parse_args()
    dimensions = parse_dimensions(args.dimensions)
    ensure_directory(args.output)

    try:
        import cocoex  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "cocoex module is required. Install package with: pip install coco-experiment"
        ) from error

    suite = cocoex.Suite("bbob", "", build_suite_filter(dimensions, args.instances))
    observer = cocoex.Observer("bbob", f"result_folder: {args.output}")

    total = len(suite)
    for index, problem in enumerate(suite, start=1):
        problem.observe_with(observer)
        run_problem(problem, args)
        print(f"[{index}/{total}] solved {problem.id}", flush=True)


if __name__ == "__main__":
    main()
