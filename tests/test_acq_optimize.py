from __future__ import annotations

import numpy as np

from bayesopt.acq_optimize import maximize_acquisition


def test_maximize_acquisition_returns_good_point() -> None:
    target = 0.3

    def acquisition(x: np.ndarray) -> float:
        delta = float(x[0]) - target
        return -(delta * delta)

    best_x, best_value = maximize_acquisition(
        acquisition_fn=acquisition,
        bounds=np.array([[0.0, 1.0]], dtype=np.float64),
        rng=np.random.default_rng(0),
    )

    assert best_x.shape == (1,)
    assert abs(float(best_x[0]) - target) < 0.1
    assert best_value >= acquisition(np.array([0.9], dtype=np.float64))
