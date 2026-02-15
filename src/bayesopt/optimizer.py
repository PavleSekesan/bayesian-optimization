from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bayesopt.acq_optimize import suggest_next_point
from bayesopt.gaussian_process import GaussianProcessRegressor
from bayesopt.kernels import RBFKernel
from bayesopt.space import from_unit_cube, sample_uniform, to_unit_cube, validate_bounds
from bayesopt.types import FloatArray, ObjectiveFunction


@dataclass(frozen=True)
class GaussianProcessConfig:
    length_scale: float = 0.5
    amplitude: float = 1.0
    noise_variance: float = 1e-6
    mean_value: float = 0.0
    jitter: float = 1e-8


@dataclass(frozen=True)
class AcquisitionConfig:
    xi: float = 0.01
    max_opt_iters: int = 80


@dataclass(frozen=True)
class OptimizationResult:
    best_x: FloatArray
    best_y: float
    x_history: FloatArray
    y_history: FloatArray
    acquisition_history: FloatArray


class BayesianOptimizer:
    def __init__(
        self,
        bounds: Sequence[tuple[float, float]] | FloatArray,
        gp_config: GaussianProcessConfig | None = None,
        acquisition_config: AcquisitionConfig | None = None,
        seed: int | None = None,
    ) -> None:
        validated_bounds = validate_bounds(bounds)
        self._bounds = validated_bounds
        self._dimension = validated_bounds.shape[0]
        self._rng = np.random.default_rng(seed)

        self._gp_config = gp_config if gp_config is not None else GaussianProcessConfig()
        self._acq_config = acquisition_config if acquisition_config is not None else AcquisitionConfig()

        self._unit_bounds = np.column_stack(
            (np.zeros(self._dimension, dtype=np.float64), np.ones(self._dimension, dtype=np.float64))
        )

    def run(
        self,
        objective: ObjectiveFunction,
        budget: int,
        n_initial: int | None = None,
    ) -> OptimizationResult:
        if budget <= 0:
            raise ValueError("budget must be positive.")

        initial_points = max(5, 2 * self._dimension) if n_initial is None else n_initial
        if initial_points <= 0:
            raise ValueError("n_initial must be positive.")
        initial_points = min(initial_points, budget)

        x_history = sample_uniform(self._rng, self._bounds, initial_points)
        y_history = np.asarray([float(objective(x)) for x in x_history], dtype=np.float64)

        gp = self._build_gp()
        acquisition_values_history: list[float] = []

        for _ in range(budget - initial_points):
            x_unit = np.asarray(to_unit_cube(x_history, self._bounds), dtype=np.float64)
            gp.fit(x_unit, y_history)

            best_y = float(np.min(y_history))
            next_unit, acquisition_value = suggest_next_point(
                gp=gp,
                bounds=self._unit_bounds,
                best_y=best_y,
                xi=self._acq_config.xi,
                max_opt_iters=self._acq_config.max_opt_iters,
            )

            next_x = np.asarray(from_unit_cube(next_unit, self._bounds), dtype=np.float64)
            next_x = self._ensure_novel_candidate(next_x, x_history)
            next_y = float(objective(next_x))

            x_history = np.vstack((x_history, next_x.reshape(1, -1)))
            y_history = np.append(y_history, next_y)
            acquisition_values_history.append(acquisition_value)

        best_index = int(np.argmin(y_history))
        return OptimizationResult(
            best_x=np.asarray(x_history[best_index], dtype=np.float64),
            best_y=float(y_history[best_index]),
            x_history=np.asarray(x_history, dtype=np.float64),
            y_history=np.asarray(y_history, dtype=np.float64),
            acquisition_history=np.asarray(acquisition_values_history, dtype=np.float64),
        )

    def _build_gp(self) -> GaussianProcessRegressor:
        kernel = RBFKernel(
            length_scale=self._gp_config.length_scale,
            amplitude=self._gp_config.amplitude,
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            noise_variance=self._gp_config.noise_variance,
            mean_value=self._gp_config.mean_value,
            jitter=self._gp_config.jitter,
        )

    def _ensure_novel_candidate(
        self,
        candidate: FloatArray,
        observed_points: FloatArray,
        tolerance: float = 1e-10,
        max_attempts: int = 64,
    ) -> FloatArray:
        if self._is_novel(candidate, observed_points, tolerance):
            return candidate

        for _ in range(max_attempts):
            sampled = sample_uniform(self._rng, self._bounds, 1)[0]
            if self._is_novel(sampled, observed_points, tolerance):
                return sampled

        return candidate

    @staticmethod
    def _is_novel(candidate: FloatArray, observed_points: FloatArray, tolerance: float) -> bool:
        deltas = np.max(np.abs(observed_points - candidate[None, :]), axis=1)
        return bool(np.all(deltas > tolerance))
