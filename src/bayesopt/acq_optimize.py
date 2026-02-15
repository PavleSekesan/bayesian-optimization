from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bayesopt.acquisition import acquisition_values
from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.space import clip_to_bounds, sample_uniform, validate_bounds
from bayesopt.types import FloatArray

AcquisitionFunction = Callable[[FloatArray], FloatArray]


def maximize_acquisition(
    acquisition_fn: AcquisitionFunction,
    bounds: FloatArray,
    rng: np.random.Generator,
    n_candidates: int = 2048,
    n_starts: int = 8,
    max_opt_iters: int = 80,
) -> tuple[FloatArray, float]:
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive.")
    if n_starts <= 0:
        raise ValueError("n_starts must be positive.")
    if max_opt_iters <= 0:
        raise ValueError("max_opt_iters must be positive.")

    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise RuntimeError("SciPy is required for acquisition optimization.") from error

    validated_bounds = validate_bounds(bounds)
    scipy_bounds = [(float(low), float(high)) for low, high in validated_bounds]

    candidates = sample_uniform(rng, validated_bounds, n_candidates)
    candidate_values = acquisition_fn(candidates)
    if candidate_values.ndim != 1 or candidate_values.shape[0] != n_candidates:
        raise ValueError("acquisition_fn must return a 1D array with one score per sample.")

    n_selected = min(n_starts, n_candidates)
    top_indices = np.argsort(candidate_values)[-n_selected:]

    best_index = int(top_indices[-1])
    best_x = np.asarray(candidates[best_index], dtype=np.float64).copy()
    best_value = float(candidate_values[best_index])

    def objective(x: FloatArray) -> float:
        point = clip_to_bounds(np.asarray(x, dtype=np.float64), validated_bounds)
        return -float(acquisition_fn(point.reshape(1, -1))[0])

    for start_index in top_indices[::-1]:
        start = np.asarray(candidates[int(start_index)], dtype=np.float64).copy()
        result = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"maxiter": max_opt_iters},
        )
        candidate_x = clip_to_bounds(np.asarray(result.x, dtype=np.float64), validated_bounds)
        candidate_value = float(acquisition_fn(candidate_x.reshape(1, -1))[0])

        if candidate_value > best_value:
            best_x = candidate_x
            best_value = candidate_value

    return best_x, best_value


def suggest_next_point(
    gp: GaussianProcessRegressor,
    bounds: FloatArray,
    best_y: float,
    xi: float,
    rng: np.random.Generator,
    n_candidates: int = 2048,
    n_starts: int = 8,
    max_opt_iters: int = 80,
) -> tuple[FloatArray, float]:
    validated_bounds = validate_bounds(bounds)

    def acquisition_fn(points: FloatArray) -> FloatArray:
        mean, variance = gp.predict(points)
        return acquisition_values(
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
        max_opt_iters=max_opt_iters,
    )
