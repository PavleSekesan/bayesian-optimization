from __future__ import annotations

import numpy as np
import pytest

from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.kernels import RBFKernel


def test_predict_requires_fit() -> None:
    gp = GaussianProcessRegressor(kernel=RBFKernel(length_scale=0.5, amplitude=1.0), noise_variance=1e-6)

    with pytest.raises(RuntimeError):
        gp.predict(np.array([[0.0]], dtype=np.float64))


def test_gp_interpolates_training_points_with_tiny_noise() -> None:
    x_train = np.array([[0.0], [0.3], [0.8]], dtype=np.float64)
    y_train = np.array([0.0, 0.8, -0.1], dtype=np.float64)

    gp = GaussianProcessRegressor(
        kernel=RBFKernel(length_scale=0.3, amplitude=1.0),
        noise_variance=1e-12,
        jitter=1e-10,
    )
    gp.fit(x_train, y_train)

    mean, variance = gp.predict(x_train)

    assert np.allclose(mean, y_train, atol=1e-4)
    assert np.all(variance >= 0.0)
    assert np.max(variance) < 1e-4


def test_gp_predict_shape() -> None:
    x_train = np.array([[0.0], [1.0]], dtype=np.float64)
    y_train = np.array([0.0, 1.0], dtype=np.float64)

    gp = GaussianProcessRegressor(kernel=RBFKernel(length_scale=0.4, amplitude=1.0), noise_variance=1e-6)
    gp.fit(x_train, y_train)

    mean, variance = gp.predict(np.array([[0.1], [0.5], [0.9]], dtype=np.float64))

    assert mean.shape == (3,)
    assert variance.shape == (3,)
