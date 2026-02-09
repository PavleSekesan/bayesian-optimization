from __future__ import annotations

import numpy as np

from bayesopt.kernels import RBFKernel


def test_rbf_matrix_is_symmetric() -> None:
    kernel = RBFKernel(length_scale=0.5, amplitude=1.5)
    x = np.array([[0.0, 0.0], [0.2, 0.5], [0.8, 0.1]], dtype=np.float64)

    gram = kernel.matrix(x, x)

    assert gram.shape == (3, 3)
    assert np.allclose(gram, gram.T, atol=1e-12)


def test_rbf_diagonal_matches_amplitude() -> None:
    kernel = RBFKernel(length_scale=1.0, amplitude=2.0)
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

    diag = kernel.diagonal(x)

    assert np.allclose(diag, np.array([2.0, 2.0, 2.0], dtype=np.float64))


def test_rbf_call_matches_matrix_entry() -> None:
    kernel = RBFKernel(length_scale=0.7, amplitude=1.1)
    x = np.array([[0.1, 0.2], [0.4, 0.8]], dtype=np.float64)

    matrix_value = kernel.matrix(x[:1], x[1:2])[0, 0]
    call_value = kernel(x[0], x[1])

    assert np.isclose(matrix_value, call_value)
