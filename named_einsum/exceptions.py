from named_einsum.characters import valid_characters

class NamedEinsumError(Exception):
    pass


class InconsistentAxisSizeError(NamedEinsumError):
    def __init__(self, axis, tensor_names, axis_sizes):
        self.axis = axis
        self.sizes = axis_sizes
        self.tensors = tensor_names
        super().__init__(f'Axis "{axis}" has inconsistent sizes across tensors: {list(zip(tensor_names, axis_sizes))}')


class InconsistentShapeDefinitionError(NamedEinsumError):
    def __init__(self, axis, num_def_axes, num_found_axes):
        self.axis = axis
        self.num_def = num_def_axes
        self.num_found = num_found_axes
        super().__init__(f'Shape for axis "{axis}" differs from definition.  Expected {self.num_def} axes, found {self.num_found}.')


class AxisNotFoundError(NamedEinsumError):
    def __init__(self, axis):
        self.axis = axis
        super().__init__(f'Output axis {axis} not found in any input tensor.')


class TooManyAxesError(NamedEinsumError):
    def __init__(self, axis_name, idx):
        self.axis = axis_name
        self.index = idx
        super().__init__(f'Axis {axis_name}, unique index {idx}, exceeds available characters for einsum: {len(valid_characters)}')
