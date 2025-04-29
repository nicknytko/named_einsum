"""Wrapper for JAX einsum."""
import jax.numpy
from named_einsum import _einsum_partial


einsum = _einsum_partial(jax.numpy.einsum)
