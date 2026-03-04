from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from common import build_suite_filter, ensure_directory, parse_dimensions, seed_for_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna GP-EI on COCO BBOB problems.")
    parser.add_argument("--dimensions", default="2,3,5,10", help="Comma-separated dimensions.")
    parser.add_argument("--instances", default="1-5", help="BBOB instance range or list.")
    parser.add_argument("--budget-factor", type=float, default=40.0, help="Budget = factor * dimension.")
    parser.add_argument(
        "--max-budget",
        type=int,
        default=0,
        help="Optional cap for the per-problem budget. 0 means no cap.",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=10,
        help="Random startup trials before GP-EI takes over.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="optuna-gpei")
    return parser.parse_args()


def run_problem(
    problem: Any,
    budget_factor: float,
    max_budget: int,
    n_startup_trials: int,
    seed: int,
) -> None:
    import optuna  # type: ignore[import-not-found]

    budget = max(1, int(budget_factor * problem.dimension))
    if max_budget > 0:
        budget = min(budget, max_budget)

    sampler = optuna.samplers.GPSampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        deterministic_objective=True,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        values = [
            trial.suggest_float(
                f"x{index}",
                float(problem.lower_bounds[index]),
                float(problem.upper_bounds[index]),
            )
            for index in range(int(problem.dimension))
        ]
        point = np.asarray(values, dtype=np.float64)
        return float(problem(point))

    study.optimize(objective, n_trials=budget, show_progress_bar=False)


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
            n_startup_trials=args.n_startup_trials,
            seed=local_seed,
        )
        print(f"[{index}/{total}] solved {problem.id}", flush=True)


if __name__ == "__main__":
    main()
