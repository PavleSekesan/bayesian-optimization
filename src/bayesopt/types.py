from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ObjectiveFunction = Callable[[FloatArray], float]


class Kernel(Protocol):
    def __call__(self, x: FloatArray, y: FloatArray) -> float:
        ...

    def matrix(self, x: FloatArray, y: FloatArray) -> FloatArray:
        ...

    def diagonal(self, x: FloatArray) -> FloatArray:
        ...
