from __future__ import annotations

from math import erf

import numpy as np

from bayesopt.types import AcquisitionKind, FloatArray

_SQRT_TWO = np.sqrt(2.0)
_INV_SQRT_TWO_PI = 1.0 / np.sqrt(2.0 * np.pi)


def expected_improvement(
    mean: FloatArray,
    variance: FloatArray,
    best_y: float,
    xi: float = 0.01,
    epsilon: float = 1e-12,
) -> FloatArray:
    _validate_inputs(mean, variance)

    std = np.sqrt(np.maximum(variance, 0.0))
    improvement = best_y - mean - xi

    scores = np.zeros_like(mean, dtype=np.float64)
    active = std > epsilon
    if not np.any(active):
        return scores

    z = np.zeros_like(mean, dtype=np.float64)
    z[active] = improvement[active] / std[active]
    cdf = _standard_normal_cdf(z[active])
    pdf = _standard_normal_pdf(z[active])
    scores[active] = improvement[active] * cdf + std[active] * pdf
    return scores


def probability_improvement(
    mean: FloatArray,
    variance: FloatArray,
    best_y: float,
    xi: float = 0.01,
    epsilon: float = 1e-12,
) -> FloatArray:
    _validate_inputs(mean, variance)

    std = np.sqrt(np.maximum(variance, 0.0))
    threshold = best_y - xi
    scores = np.zeros_like(mean, dtype=np.float64)

    active = std > epsilon
    if np.any(active):
        z = (threshold - mean[active]) / std[active]
        scores[active] = _standard_normal_cdf(z)

    deterministic = ~active
    if np.any(deterministic):
        scores[deterministic] = (mean[deterministic] <= threshold).astype(np.float64)

    return scores


def acquisition_values(
    kind: AcquisitionKind,
    mean: FloatArray,
    variance: FloatArray,
    best_y: float,
    xi: float = 0.01,
) -> FloatArray:
    if kind == "ei":
        return expected_improvement(mean=mean, variance=variance, best_y=best_y, xi=xi)
    if kind == "pi":
        return probability_improvement(mean=mean, variance=variance, best_y=best_y, xi=xi)
    raise ValueError(f"Unsupported acquisition kind: {kind}")


def _standard_normal_pdf(z: FloatArray) -> FloatArray:
    return np.asarray(_INV_SQRT_TWO_PI * np.exp(-0.5 * z * z), dtype=np.float64)


def _standard_normal_cdf(z: FloatArray) -> FloatArray:
    flat = np.ravel(z)
    erf_values = np.fromiter((erf(float(value) / float(_SQRT_TWO)) for value in flat), dtype=np.float64)
    return np.asarray(0.5 * (1.0 + erf_values.reshape(z.shape)), dtype=np.float64)


def _validate_inputs(mean: FloatArray, variance: FloatArray) -> None:
    if mean.ndim != 1 or variance.ndim != 1:
        raise ValueError("mean and variance must be 1D arrays.")
    if mean.shape != variance.shape:
        raise ValueError("mean and variance must have identical shapes.")
