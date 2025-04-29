"""Wrapper for Numpy einsum."""
import numpy as np
from named_einsum import _einsum_partial


einsum = _einsum_partial(np.einsum)
