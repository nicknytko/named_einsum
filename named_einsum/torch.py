"""Wrapper for Torch einsum."""
import torch
from named_einsum import _einsum_partial


einsum = _einsum_partial(torch.einsum)
