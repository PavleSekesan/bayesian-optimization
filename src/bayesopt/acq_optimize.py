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
    if not hasattr(acquisition_fn, "x0") or not hasattr(acquisition_fn, "bounds"):
        raise ValueError("acquisition_fn must define 'x0' and 'bounds' attributes.")

    x0 = np.asarray(getattr(acquisition_fn, "x0"), dtype=np.float64)
    bounds = validate_bounds(np.asarray(getattr(acquisition_fn, "bounds"), dtype=np.float64))
    if x0.ndim != 1 or x0.shape[0] != bounds.shape[0]:
        raise ValueError("acquisition_fn.x0 must be a 1D vector matching bounds dimensionality.")

    max_opt_iters = int(getattr(acquisition_fn, "max_opt_iters", 80))
    if max_opt_iters <= 0:
        raise ValueError("acquisition_fn.max_opt_iters must be positive.")

    scipy_bounds = [(float(low), float(high)) for low, high in bounds]
    result = minimize(
        lambda x: -float(acquisition_fn(np.asarray(x, dtype=np.float64))),
        x0=x0,
        method="L-BFGS-B",
        bounds=scipy_bounds,
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
    scipy_bounds = [(float(low), float(high)) for low, high in validated_bounds]

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
    setattr(acquisition_fn, "bounds", scipy_bounds)
    setattr(acquisition_fn, "max_opt_iters", max_opt_iters)
    return maximize_acquisition(acquisition_fn)
