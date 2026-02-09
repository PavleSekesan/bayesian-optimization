from __future__ import annotations

from typing import Sequence

import numpy as np

from bayesopt.types import FloatArray


def validate_bounds(bounds: Sequence[tuple[float, float]] | FloatArray) -> FloatArray:
    bounds_array = np.asarray(bounds, dtype=np.float64)
    if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
        raise ValueError("Bounds must be a 2D array-like object with shape (dimension, 2).")
    if not np.all(np.isfinite(bounds_array)):
        raise ValueError("Bounds must contain finite values.")

    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]
    if not np.all(lower < upper):
        raise ValueError("Each lower bound must be strictly smaller than the upper bound.")

    return bounds_array


def sample_uniform(
    rng: np.random.Generator,
    bounds: Sequence[tuple[float, float]] | FloatArray,
    n_samples: int,
) -> FloatArray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    validated_bounds = validate_bounds(bounds)
    lower = validated_bounds[:, 0]
    upper = validated_bounds[:, 1]
    samples = rng.uniform(lower, upper, size=(n_samples, validated_bounds.shape[0]))
    return np.asarray(samples, dtype=np.float64)


def clip_to_bounds(x: FloatArray, bounds: Sequence[tuple[float, float]] | FloatArray) -> FloatArray:
    validated_bounds = validate_bounds(bounds)
    if x.ndim != 1 or x.shape[0] != validated_bounds.shape[0]:
        raise ValueError("x must be a 1D vector with the same dimension as bounds.")
    lower = validated_bounds[:, 0]
    upper = validated_bounds[:, 1]
    return np.clip(x, lower, upper)


def to_unit_cube(x: FloatArray, bounds: Sequence[tuple[float, float]] | FloatArray) -> FloatArray:
    validated_bounds = validate_bounds(bounds)
    lower = validated_bounds[:, 0]
    upper = validated_bounds[:, 1]
    span = upper - lower
    if x.ndim == 1:
        if x.shape[0] != validated_bounds.shape[0]:
            raise ValueError("x must match bounds dimensionality.")
        return (x - lower) / span

    if x.ndim == 2:
        if x.shape[1] != validated_bounds.shape[0]:
            raise ValueError("x must match bounds dimensionality.")
        return (x - lower[None, :]) / span[None, :]

    raise ValueError("x must be 1D or 2D.")


def from_unit_cube(u: FloatArray, bounds: Sequence[tuple[float, float]] | FloatArray) -> FloatArray:
    validated_bounds = validate_bounds(bounds)
    lower = validated_bounds[:, 0]
    upper = validated_bounds[:, 1]
    span = upper - lower
    if u.ndim == 1:
        if u.shape[0] != validated_bounds.shape[0]:
            raise ValueError("u must match bounds dimensionality.")
        return lower + u * span

    if u.ndim == 2:
        if u.shape[1] != validated_bounds.shape[0]:
            raise ValueError("u must match bounds dimensionality.")
        return lower[None, :] + u * span[None, :]

    raise ValueError("u must be 1D or 2D.")
