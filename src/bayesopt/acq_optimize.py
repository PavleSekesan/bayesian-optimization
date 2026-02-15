from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from bayesopt.acquisition import acquisition_values
from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.space import validate_bounds
from bayesopt.types import FloatArray

AcquisitionFunction = Callable[[FloatArray], float]


def maximize_acquisition(
    acquisition_fn: AcquisitionFunction,
) -> tuple[FloatArray, float]:
    x0 = np.asarray(getattr(acquisition_fn, "x0", np.zeros(1, dtype=np.float64)), dtype=np.float64)
    max_opt_iters = int(getattr(acquisition_fn, "max_opt_iters", 80))
    result = minimize(
        lambda x: -float(acquisition_fn(np.asarray(x, dtype=np.float64))),
        x0=x0,
        method="L-BFGS-B",
        options={"maxiter": max_opt_iters},
    )
    best_x = np.asarray(result.x, dtype=np.float64)
    best_value = float(acquisition_fn(best_x))
    return best_x, best_value


def suggest_next_point(
    gp: GaussianProcessRegressor,
    bounds: FloatArray,
    best_y: float,
    xi: float,
    max_opt_iters: int = 80,
) -> tuple[FloatArray, float]:
    validated_bounds = validate_bounds(bounds)
    x0 = np.mean(validated_bounds, axis=1)

    def acquisition_fn(point: FloatArray) -> float:
        query = np.asarray(point, dtype=np.float64).reshape(1, -1)
        mean, variance = gp.predict(query)
        scores = acquisition_values(
            mean=mean,
            variance=variance,
            best_y=best_y,
            xi=xi,
        )
        return float(scores[0])

    setattr(acquisition_fn, "x0", x0)
    setattr(acquisition_fn, "max_opt_iters", max_opt_iters)
    return maximize_acquisition(acquisition_fn)
