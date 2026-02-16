from __future__ import annotations

from bayesopt.acquisition import expected_improvement


def test_ei_prefers_lower_mean_when_variance_matches() -> None:
    score_low = expected_improvement(mean=0.1, variance=0.2, best_y=0.2, xi=0.0)
    score_high = expected_improvement(mean=0.5, variance=0.2, best_y=0.2, xi=0.0)
    assert score_low > score_high


def test_ei_is_zero_with_zero_variance() -> None:
    assert expected_improvement(mean=0.1, variance=0.0, best_y=0.2, xi=0.0) == 0.0
    assert expected_improvement(mean=0.3, variance=0.0, best_y=0.2, xi=0.0) == 0.0
