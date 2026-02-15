from __future__ import annotations

import numpy as np

from bayesopt.acquisition import expected_improvement


def test_ei_prefers_lower_mean_when_variance_matches() -> None:
    mean = np.array([0.1, 0.5], dtype=np.float64)
    variance = np.array([0.2, 0.2], dtype=np.float64)

    scores = expected_improvement(mean=mean, variance=variance, best_y=0.2, xi=0.0)

    assert scores[0] > scores[1]


def test_ei_is_zero_with_zero_variance() -> None:
    mean = np.array([0.1, 0.3], dtype=np.float64)
    variance = np.array([0.0, 0.0], dtype=np.float64)

    ei = expected_improvement(mean=mean, variance=variance, best_y=0.2, xi=0.0)

    assert np.allclose(ei, np.array([0.0, 0.0], dtype=np.float64))
