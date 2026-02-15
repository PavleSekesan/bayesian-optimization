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
    dimension = int(getattr(acquisition_fn, "dimension", 1))
    x0 = np.zeros(dimension, dtype=np.float64)
    result = minimize(
        lambda x: -float(acquisition_fn(np.asarray(x, dtype=np.float64))),
        x0=x0,
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

    setattr(acquisition_fn, "dimension", validated_bounds.shape[0])
    return maximize_acquisition(acquisition_fn)
