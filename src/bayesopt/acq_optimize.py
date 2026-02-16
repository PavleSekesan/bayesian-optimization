from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from bayesopt.acquisition import expected_improvement
from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.space import validate_bounds
from bayesopt.types import FloatArray

AcquisitionFunction = Callable[[FloatArray], float]


def maximize_acquisition(
    acquisition_fn: AcquisitionFunction,
    x0: FloatArray,
) -> tuple[FloatArray, float]:
    result = minimize(
        lambda x: -float(acquisition_fn(np.asarray(x, dtype=np.float64))),
        x0=np.asarray(x0, dtype=np.float64),
        method="L-BFGS-B",
    )
    best_x = np.asarray(result.x, dtype=np.float64)
    best_value = float(acquisition_fn(best_x))
    return best_x, best_value


def suggest_next_point(
    gp: GaussianProcessRegressor,
    bounds: FloatArray,
    best_y: float,
    xi: float,
) -> tuple[FloatArray, float]:
    validated_bounds = validate_bounds(bounds)
    x0 = np.mean(validated_bounds, axis=1)

    def acquisition_fn(point: FloatArray) -> float:
        query = np.asarray(point, dtype=np.float64).reshape(1, -1)
        mean, variance = gp.predict(query)
        return expected_improvement(
            mean=float(mean[0]),
            variance=float(variance[0]),
            best_y=best_y,
            xi=xi,
        )

    return maximize_acquisition(acquisition_fn, x0=x0)
