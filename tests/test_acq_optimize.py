from __future__ import annotations

import numpy as np
import pytest

from bayesopt.acq_optimize import maximize_acquisition
from bayesopt.space import sample_uniform

pytest.importorskip("scipy")


def test_maximize_acquisition_returns_good_point() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    target = np.array([0.3, 0.7], dtype=np.float64)

    def acquisition(points: np.ndarray) -> np.ndarray:
        deltas = points - target[None, :]
        return -np.sum(deltas * deltas, axis=1)

    rng = np.random.default_rng(7)
    best_x, best_value = maximize_acquisition(
        acquisition_fn=acquisition,
        bounds=bounds,
        rng=rng,
        n_candidates=512,
        n_starts=6,
    )

    assert np.all(best_x >= bounds[:, 0])
    assert np.all(best_x <= bounds[:, 1])
    assert np.linalg.norm(best_x - target) < 0.1

    baseline_rng = np.random.default_rng(99)
    random_candidates = sample_uniform(baseline_rng, bounds, n_samples=32)
    baseline_best = float(np.max(acquisition(random_candidates)))
    assert best_value >= baseline_best
