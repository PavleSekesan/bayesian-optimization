from __future__ import annotations

import numpy as np

from bayesopt.types import FloatArray, Kernel


class GaussianProcessRegressor:
    def __init__(
        self,
        kernel: Kernel,
        noise_variance: float,
        mean_value: float = 0.0,
        jitter: float = 1e-8,
    ) -> None:
        if noise_variance < 0.0:
            raise ValueError("noise_variance must be non-negative.")
        if jitter <= 0.0:
            raise ValueError("jitter must be positive.")

        self.kernel = kernel
        self.noise_variance = noise_variance
        self.mean_value = mean_value
        self.jitter = jitter

        self._x_train: FloatArray | None = None
        self._y_train: FloatArray | None = None
        self._cholesky: FloatArray | None = None
        self._alpha: FloatArray | None = None

    def fit(self, x_train: FloatArray, y_train: FloatArray) -> None:
        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array.")
        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array.")
        if x_train.shape[0] == 0:
            raise ValueError("x_train must not be empty.")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must contain the same number of samples.")

        x_data = np.asarray(x_train, dtype=np.float64)
        y_data = np.asarray(y_train, dtype=np.float64)

        gram = self.kernel.matrix(x_data, x_data)
        cholesky = self._stable_cholesky(gram)

        centered_targets = y_data - self.mean_value
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, centered_targets))

        self._x_train = x_data
        self._y_train = y_data
        self._cholesky = cholesky
        self._alpha = np.asarray(alpha, dtype=np.float64)

    def predict(self, x_query: FloatArray) -> tuple[FloatArray, FloatArray]:
        self._require_fitted()
        x_train = self._x_train
        cholesky = self._cholesky
        alpha = self._alpha
        assert x_train is not None
        assert cholesky is not None
        assert alpha is not None

        if x_query.ndim == 1:
            x_data = np.asarray(x_query[None, :], dtype=np.float64)
        elif x_query.ndim == 2:
            x_data = np.asarray(x_query, dtype=np.float64)
        else:
            raise ValueError("x_query must be a 1D or 2D array.")

        if x_data.shape[1] != x_train.shape[1]:
            raise ValueError("x_query has a different feature dimension than the training data.")

        cross_cov = self.kernel.matrix(x_data, x_train)
        mean = self.mean_value + cross_cov @ alpha

        projection = np.linalg.solve(cholesky, cross_cov.T)
        variance = self.kernel.diagonal(x_data) - np.sum(projection * projection, axis=0)
        variance = np.maximum(variance, 0.0)

        return np.asarray(mean, dtype=np.float64), np.asarray(variance, dtype=np.float64)

    def _stable_cholesky(self, gram: FloatArray) -> FloatArray:
        diagonal_indices = np.diag_indices_from(gram)
        jitter = self.jitter

        for _ in range(8):
            regularized = np.asarray(gram, dtype=np.float64).copy()
            regularized[diagonal_indices] += self.noise_variance + jitter
            try:
                return np.linalg.cholesky(regularized)
            except np.linalg.LinAlgError:
                jitter *= 10.0

        raise np.linalg.LinAlgError("Unable to compute a stable Cholesky factor for the kernel matrix.")

    def _require_fitted(self) -> None:
        if self._x_train is None or self._cholesky is None or self._alpha is None:
            raise RuntimeError("GaussianProcessRegressor must be fitted before prediction.")
