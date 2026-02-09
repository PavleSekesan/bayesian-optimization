from __future__ import annotations

import numpy as np
import pytest

from bayesopt.optimizer import BayesianOptimizer


def sphere(x: np.ndarray) -> float:
    center = np.array([0.25, 0.25], dtype=np.float64)
    diff = x - center
    return float(np.sum(diff * diff))


def test_optimizer_respects_budget_and_shapes() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    optimizer = BayesianOptimizer(bounds=bounds, acquisition="ei", seed=3)

    result = optimizer.run(objective=sphere, budget=20, n_initial=6)

    assert result.x_history.shape == (20, 2)
    assert result.y_history.shape == (20,)
    assert result.acquisition_history.shape == (14,)
    assert np.isclose(result.best_y, np.min(result.y_history))


def test_optimizer_improves_over_initial_set() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    optimizer = BayesianOptimizer(bounds=bounds, acquisition="ei", seed=11)

    result = optimizer.run(objective=sphere, budget=18, n_initial=6)

    initial_best = float(np.min(result.y_history[:6]))
    assert result.best_y <= initial_best


def test_optimizer_rejects_non_positive_budget() -> None:
    bounds = np.array([[0.0, 1.0]], dtype=np.float64)
    optimizer = BayesianOptimizer(bounds=bounds, acquisition="ei", seed=1)

    with pytest.raises(ValueError):
        optimizer.run(objective=lambda x: float(np.sum(x)), budget=0)
