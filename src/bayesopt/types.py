from __future__ import annotations

from typing import Callable, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ObjectiveFunction = Callable[[FloatArray], float]
AcquisitionKind = Literal["ei", "pi"]


class Kernel(Protocol):
    def __call__(self, x: FloatArray, y: FloatArray) -> float:
        ...

    def matrix(self, x: FloatArray, y: FloatArray) -> FloatArray:
        ...

    def diagonal(self, x: FloatArray) -> FloatArray:
        ...
