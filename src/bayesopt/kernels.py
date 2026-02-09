from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesopt.types import FloatArray


@dataclass(frozen=True)
class RBFKernel:
    length_scale: float
    amplitude: float

    def __post_init__(self) -> None:
        if self.length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")
        if self.amplitude <= 0.0:
            raise ValueError("amplitude must be positive.")

    def __call__(self, x: FloatArray, y: FloatArray) -> float:
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D vectors.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same dimensionality.")

        diff = (x - y) / self.length_scale
        sq_norm = float(np.dot(diff, diff))
        return float(self.amplitude * np.exp(-0.5 * sq_norm))

    def matrix(self, x: FloatArray, y: FloatArray) -> FloatArray:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D arrays.")
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same feature dimension.")

        x_scaled = x / self.length_scale
        y_scaled = y / self.length_scale

        x_sq = np.sum(x_scaled * x_scaled, axis=1)[:, None]
        y_sq = np.sum(y_scaled * y_scaled, axis=1)[None, :]
        sq_dist = x_sq + y_sq - 2.0 * (x_scaled @ y_scaled.T)
        sq_dist = np.maximum(sq_dist, 0.0)

        return np.asarray(self.amplitude * np.exp(-0.5 * sq_dist), dtype=np.float64)

    def diagonal(self, x: FloatArray) -> FloatArray:
        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")
        return np.full(shape=(x.shape[0],), fill_value=self.amplitude, dtype=np.float64)
