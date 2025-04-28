from named_einsum.lark_parser import Lark_StandAlone
from types import SimpleNamespace
import named_einsum.exceptions
from named_einsum.characters import valid_characters


class Variable:
    def __init__(self, name=None, axes=[]):
        self.name = name
        self.axes = axes

    @property
    def axis_names(self):
        return [axis for axis in self.axes if isinstance(axis, str)]


def _parse_variable(tree):
    assert(tree.data == 'variable')

    # Optionally capture the variable name, if provided
    if len(tree.children) == 2:
        name = tree.children[0].children[0].value
        tree_axes = tree.children[1]
    elif tree.children[0].data == 'name':
        name = tree.children[0].children[0].value
        tree_axes = SimpleNamespace(children=[])
    elif tree.children[0].data == 'axes':
        name = None
        tree_axes = tree.children[0]

    # Grab all axis names
    axes = []
    for tree_axis in tree_axes.children:
        if tree_axis.data == 'name':
            axes.append(tree_axis.children[0].value.lower())
        else:
            axes.append(...)

    return Variable(name=name, axes=axes)


def _idx_to_letter(axis, idx):
    if idx >= len(valid_characters):
        raise named_einsum.exceptions.TooManyAxesError(axis, idx)
    return valid_characters[idx]


def parse(inp):
    parser = Lark_StandAlone()
    tree = parser.parse(inp)

    assert(tree.data == 'einsum')
    assert(len(tree.children) == 2)
    assert(tree.children[0].data == 'input_variables')
    assert(tree.children[1].data == 'output_variable')

    tree_input_variables = tree.children[0]
    tree_output_variable = tree.children[1]

    input_variables = [_parse_variable(child) for child in tree_input_variables.children]
    for i, v in enumerate(input_variables):
        if v.name is None:
            v.name = f'input_{i}'

    output_variable = (None if len(tree_output_variable.children) == 0
                       else _parse_variable(tree_output_variable.children[0]))

    # Axis name to some letter
    unique_axis_idx = 0
    axis_mapping = {}

    input_axes = set()
    output_axes = set()

    # Find unique input axes
    for input_variable in input_variables:
        for axis_name in input_variable.axis_names:
            if axis_name not in axis_mapping:
                axis_mapping[axis_name] = _idx_to_letter(axis_name, unique_axis_idx)
                unique_axis_idx += 1
            input_axes.add(axis_name)

    # Ensure all output axes belong to some input
    if output_variable is not None:
        for axis_name in output_variable.axis_names:
            if axis_name not in input_axes:
                raise named_einsum.exceptions.AxisNotFoundError(axis_name)
            output_axes.add(axis_name)

    return SimpleNamespace(
        input_variables=input_variables,
        output_variable=output_variable,
        input_axes=input_axes,
        output_axes=output_axes,
        axis_mapping=axis_mapping
    )
