from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from bayesopt.space import sample_uniform
from experiments.common import build_suite_filter, ensure_directory, parse_dimensions, seed_for_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random search on COCO BBOB problems.")
    parser.add_argument("--dimensions", default="2,3,5,10", help="Comma-separated dimensions.")
    parser.add_argument("--instances", default="1-5", help="BBOB instance range or list.")
    parser.add_argument("--budget-factor", type=float, default=40.0, help="Budget = factor * dimension.")
    parser.add_argument(
        "--max-budget",
        type=int,
        default=0,
        help="Optional cap for the per-problem budget. 0 means no cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/random")
    return parser.parse_args()


def run_problem(problem: Any, budget_factor: float, max_budget: int, seed: int) -> None:
    bounds = np.column_stack((problem.lower_bounds, problem.upper_bounds)).astype(np.float64)
    budget = max(1, int(budget_factor * problem.dimension))
    if max_budget > 0:
        budget = min(budget, max_budget)
    rng = np.random.default_rng(seed)

    samples = sample_uniform(rng=rng, bounds=bounds, n_samples=budget)
    for x in samples:
        _ = problem(np.asarray(x, dtype=np.float64))


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
        local_seed = seed_for_problem(args.seed, str(problem.id))
        run_problem(
            problem,
            budget_factor=args.budget_factor,
            max_budget=args.max_budget,
            seed=local_seed,
        )
        print(f"[{index}/{total}] solved {problem.id}", flush=True)


if __name__ == "__main__":
    main()
