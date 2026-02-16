from __future__ import annotations

from math import erf, exp, pi, sqrt

_SQRT_TWO = sqrt(2.0)
_INV_SQRT_TWO_PI = 1.0 / sqrt(2.0 * pi)


def expected_improvement(
    mean: float,
    variance: float,
    best_y: float,
    xi: float = 0.01,
    epsilon: float = 1e-12,
) -> float:
    std = sqrt(max(variance, 0.0))
    if std <= epsilon:
        return 0.0

    improvement = best_y - mean - xi
    z = improvement / std
    cdf = _standard_normal_cdf(z)
    pdf = _standard_normal_pdf(z)
    return improvement * cdf + std * pdf


def _standard_normal_pdf(z: float) -> float:
    return _INV_SQRT_TWO_PI * exp(-0.5 * z * z)


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / _SQRT_TWO))
