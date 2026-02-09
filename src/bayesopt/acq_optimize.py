from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bayesopt.acquisition import acquisition_values
from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.space import clip_to_bounds, sample_uniform, validate_bounds
from bayesopt.types import AcquisitionKind, FloatArray

AcquisitionFunction = Callable[[FloatArray], FloatArray]


def maximize_acquisition(
    acquisition_fn: AcquisitionFunction,
    bounds: FloatArray,
    rng: np.random.Generator,
    n_candidates: int = 2048,
    n_starts: int = 8,
    initial_step_fraction: float = 0.1,
    min_step_fraction: float = 1e-3,
    max_refine_iters: int = 80,
) -> tuple[FloatArray, float]:
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive.")
    if n_starts <= 0:
        raise ValueError("n_starts must be positive.")
    if initial_step_fraction <= 0.0:
        raise ValueError("initial_step_fraction must be positive.")
    if min_step_fraction <= 0.0:
        raise ValueError("min_step_fraction must be positive.")
    if max_refine_iters <= 0:
        raise ValueError("max_refine_iters must be positive.")

    validated_bounds = validate_bounds(bounds)
    width = validated_bounds[:, 1] - validated_bounds[:, 0]
    min_step = min_step_fraction * width

    candidates = sample_uniform(rng, validated_bounds, n_candidates)
    candidate_values = acquisition_fn(candidates)
    if candidate_values.ndim != 1 or candidate_values.shape[0] != n_candidates:
        raise ValueError("acquisition_fn must return a 1D array with one score per sample.")

    n_selected = min(n_starts, n_candidates)
    top_indices = np.argsort(candidate_values)[-n_selected:]

    best_index = int(top_indices[-1])
    best_x = np.asarray(candidates[best_index], dtype=np.float64).copy()
    best_value = float(candidate_values[best_index])

    for start_index in top_indices[::-1]:
        current_x = np.asarray(candidates[int(start_index)], dtype=np.float64).copy()
        current_value = float(candidate_values[int(start_index)])
        step = initial_step_fraction * width

        for _ in range(max_refine_iters):
            improved = False
            for dim in range(validated_bounds.shape[0]):
                for direction in (-1.0, 1.0):
                    trial_x = current_x.copy()
                    trial_x[dim] += direction * step[dim]
                    trial_x = clip_to_bounds(trial_x, validated_bounds)
                    trial_value = float(acquisition_fn(trial_x.reshape(1, -1))[0])

                    if trial_value > current_value:
                        current_x = trial_x
                        current_value = trial_value
                        improved = True

            if current_value > best_value:
                best_x = current_x.copy()
                best_value = current_value

            if not improved:
                step *= 0.5
                if np.all(step <= min_step):
                    break

    return best_x, best_value


def suggest_next_point(
    gp: GaussianProcessRegressor,
    bounds: FloatArray,
    acquisition_kind: AcquisitionKind,
    best_y: float,
    xi: float,
    rng: np.random.Generator,
    n_candidates: int = 2048,
    n_starts: int = 8,
    initial_step_fraction: float = 0.1,
    min_step_fraction: float = 1e-3,
    max_refine_iters: int = 80,
) -> tuple[FloatArray, float]:
    validated_bounds = validate_bounds(bounds)

    def acquisition_fn(points: FloatArray) -> FloatArray:
        mean, variance = gp.predict(points)
        return acquisition_values(
            kind=acquisition_kind,
            mean=mean,
            variance=variance,
            best_y=best_y,
            xi=xi,
        )

    return maximize_acquisition(
        acquisition_fn=acquisition_fn,
        bounds=validated_bounds,
        rng=rng,
        n_candidates=n_candidates,
        n_starts=n_starts,
        initial_step_fraction=initial_step_fraction,
        min_step_fraction=min_step_fraction,
        max_refine_iters=max_refine_iters,
    )
