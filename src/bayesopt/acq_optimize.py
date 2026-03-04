from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from bayesopt.acquisition import expected_improvement
from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.space import sample_uniform, validate_bounds
from bayesopt.types import FloatArray

AcquisitionFunction = Callable[[FloatArray], float]


def maximize_acquisition(
    acquisition_fn: AcquisitionFunction,
    bounds: FloatArray,
    rng: np.random.Generator,
    n_restarts: int = 8,
) -> tuple[FloatArray, float]:
    validated_bounds = validate_bounds(bounds)
    start_points = sample_uniform(
        rng=rng,
        bounds=validated_bounds,
        n_samples=n_restarts,
    )

    scipy_bounds = [tuple(bound) for bound in validated_bounds]
    best_x = np.asarray(start_points[0], dtype=np.float64)
    best_value = float("-inf")
    for start in start_points:
        result = minimize(
            lambda x: -float(acquisition_fn(np.asarray(x, dtype=np.float64))),
            x0=np.asarray(start, dtype=np.float64),
            bounds=scipy_bounds,
            method="L-BFGS-B",
        )
        candidate_x = np.asarray(result.x, dtype=np.float64)
        candidate_value = float(acquisition_fn(candidate_x))
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
) -> tuple[FloatArray, float]:
    validated_bounds = validate_bounds(bounds)

    def acquisition_fn(point: FloatArray) -> float:
        query = np.asarray(point, dtype=np.float64).reshape(1, -1)
        mean, variance = gp.predict(query)
        return expected_improvement(
            mean=float(mean.item()),
            variance=float(variance.item()),
            best_y=best_y,
            xi=xi,
        )

    return maximize_acquisition(
        acquisition_fn,
        bounds=validated_bounds,
        rng=rng,
    )
