from __future__ import annotations

import numpy as np

from bayesopt.acq_optimize import maximize_acquisition


def test_maximize_acquisition_returns_good_point() -> None:
    target = np.array([0.3, 0.7], dtype=np.float64)

    def acquisition(x: np.ndarray) -> float:
        deltas = x - target
        return -float(np.sum(deltas * deltas))

    acquisition.x0 = np.array([0.5, 0.5], dtype=np.float64)
    acquisition.bounds = [(0.0, 1.0), (0.0, 1.0)]
    acquisition.max_opt_iters = 60

    best_x, best_value = maximize_acquisition(acquisition_fn=acquisition)

    assert np.all(best_x >= 0.0)
    assert np.all(best_x <= 1.0)
    assert np.linalg.norm(best_x - target) < 0.1
    assert best_value >= acquisition(np.array([0.9, 0.1], dtype=np.float64))
