"""Library-specific errors and exceptions."""
from named_einsum.characters import VALID_CHARACTERS


class NamedEinsumError(Exception):
    """Base error type."""


class InconsistentAxisSizeError(NamedEinsumError):
    """An axis was found with differing (inconsistent) sizes across several tensors."""

    def __init__(self, axis, axis_sizes):
        self.axis = axis
        self.sizes = axis_sizes
        super().__init__(
            f'Axis "{axis}" has inconsistent sizes across tensors: ' +
            f'found {", ".join([str(size) for size in axis_sizes])}'
        )


class InconsistentShapeDefinitionError(NamedEinsumError):
    """A tensor was encountered with differing number of axes than what was stated."""

    def __init__(self, tensor_name, num_def_axes, num_found_axes):
        self.tensor_name = tensor_name
        self.num_def = num_def_axes
        self.num_found = num_found_axes
        super().__init__(
            f'Dimensionality for tensor "{tensor_name}" differs from definition.  ' +
            f'Expected {self.num_def} axes, found {self.num_found}.'
        )


class AmbiguousEllipsesError(NamedEinsumError):
    """Multiple ellipsis axes were found for a single variable."""

    def __init__(self, tensor_name):
        self.tensor_name = tensor_name
        super().__init__(
            f'Ambiguous shape was found for tensor {tensor_name}:  ' +
            'Multiple ellipses "..." were encountered'
        )


class AxisNotFoundError(NamedEinsumError):
    """An axis in the output tensor was not found in any input tensor."""

    def __init__(self, axis):
        self.axis = axis
        super().__init__(f'Output axis {axis} not found in any input tensor.')


class TooManyAxesError(NamedEinsumError):
    """Too many unique axes were encountered for a valid output einsum to be generated."""

    def __init__(self, axis_name, idx):
        self.axis = axis_name
        self.index = idx
        super().__init__(
            f'Axis {axis_name}, unique index {idx}, ' +
            f'exceeds available characters for einsum: {len(VALID_CHARACTERS)}'
        )
