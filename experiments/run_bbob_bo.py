from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from bayesopt.optimizer import AcquisitionConfig, BayesianOptimizer
from common import build_suite_filter, ensure_directory, parse_dimensions, seed_for_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian optimization on COCO BBOB problems.")
    parser.add_argument("--dimensions", default="2,3,5,10", help="Comma-separated dimensions.")
    parser.add_argument("--instances", default="1-5", help="BBOB instance range or list.")
    parser.add_argument("--budget-factor", type=float, default=40.0, help="Budget = factor * dimension.")
    parser.add_argument(
        "--max-budget",
        type=int,
        default=0,
        help="Optional cap for the per-problem budget. 0 means no cap.",
    )
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--n-initial", type=int, default=0, help="0 means use library default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/bo")
    return parser.parse_args()


def run_problem(problem: Any, args: argparse.Namespace) -> None:
    bounds = np.column_stack((problem.lower_bounds, problem.upper_bounds)).astype(np.float64)
    budget = max(1, int(args.budget_factor * problem.dimension))
    if args.max_budget > 0:
        budget = min(budget, args.max_budget)
    local_seed = seed_for_problem(args.seed, str(problem.id))

    acq_config = AcquisitionConfig(
        xi=args.xi,
    )

    optimizer = BayesianOptimizer(
        bounds=bounds,
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

    import cocoex  # type: ignore[import-not-found]

    suite = cocoex.Suite("bbob", "", build_suite_filter(dimensions, args.instances))
    observer = cocoex.Observer("bbob", f"result_folder: {args.output}")

    total = len(suite)
    for index, problem in enumerate(suite, start=1):
        problem.observe_with(observer)
        run_problem(problem, args)
        print(f"[{index}/{total}] solved {problem.id}", flush=True)


if __name__ == "__main__":
    main()
