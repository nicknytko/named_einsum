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
