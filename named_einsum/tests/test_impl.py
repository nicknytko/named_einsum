"""Numerical tests of einsum."""
import numpy as np
import jax.numpy as jnp
import torch
import named_einsum
import named_einsum.numpy
import named_einsum.jax
import named_einsum.torch


def _close(val, true, tol=1e-5):
    return abs(val.item() - true) < tol


def test_wrapper():
    """Test wrapping numpy."""
    A = np.ones((10, 10, 10))
    assert _close(named_einsum.einsum(np.einsum, 'A[a,b,c] ->', A), 10**3)


def test_numpy():
    """Test Numpy einsum."""
    A = np.ones((10, 10, 10))
    assert _close(named_einsum.numpy.einsum('A[a, b, c] ->', A), 10**3)


def test_jax():
    """Test JAX einsum."""
    A = jnp.ones((10, 10, 10))
    assert _close(named_einsum.jax.einsum('A[a, b, c] ->', A), 10**3)


def test_torch():
    """Test Torch einsum."""
    A = torch.ones((10, 10, 10))
    assert _close(named_einsum.torch.einsum('A[a, b, c] ->', A), 10**3)


def test_matvec():
    """matrix-vector product."""
    A = np.eye(10, k=1)  # Shift matrix
    b = np.zeros(10)
    b[1:] = 1.
    out_expected = np.zeros(10)
    out_expected[:-1] = 1.

    assert np.max(abs(named_einsum.numpy.einsum('A[i,j], b[j] -> [i]', A, b) - out_expected)) < 1e-5


def test_product_axis():
    """Test a reduction product axis."""
    A = np.ones(10)
    B = np.ones(5)
    C = np.ones(7)
    out = named_einsum.numpy.einsum('A[i], B[j], C[k] -> C[i*j*k]', A, B, C)
    assert out.ndim == 1 and out.shape[0] == (10 * 5 * 7)
    assert np.all(out == 1.)


def test_khatri_rao_product():
    """Khatri-rao product, i.e., the column-wise Kronecker product."""
    # This example is taken from https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product

    C = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]).astype(np.float64)

    D = np.array([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
    ]).astype(np.float64)

    krp = named_einsum.numpy.einsum(
        'C[i, l], D[j, l] -> KRP[i*j, l]',
        C, D
    )

    out_expected = np.array([
        [1, 8, 21],
        [2, 10, 24],
        [3, 12, 27],
        [4, 20, 42],
        [8, 25, 48],
        [12, 30, 54],
        [7, 32, 63],
        [14, 40, 72],
        [21, 48, 81]
    ]).astype(np.float64)

    assert np.allclose(krp, out_expected)
