"""Parsing of named einsum expressions."""
from types import SimpleNamespace
import functools
from abc import ABC, abstractmethod

from named_einsum.lark_parser import Lark_StandAlone
import named_einsum.exceptions
from named_einsum.characters import VALID_CHARACTERS


class BaseAxis(ABC):
    """Base axis type representing some abstract axis einsum definition."""

    @property
    @abstractmethod
    def axis_names(self):
        """Returns a list of string axis represented by this object."""
        return []

    @property
    def flattened_axes(self):
        """Returns a list of NamedAxis/EllipsisAxis represented by this object."""
        return [self]

    @abstractmethod
    def einsum_repr(self, mapping):
        """Returns the output representation of this object in the einsum string."""
        return ''


class NamedAxis(BaseAxis):
    """A material axis that has an (optional) name."""

    def __init__(self, name):
        self.name = name

    @property
    def axis_names(self):
        """Returns the name of this axis."""
        return [self.name]

    def __repr__(self):
        """String representation of this named axis."""
        return f'(NamedAxis: {self.name})'

    def einsum_repr(self, mapping):
        """Returns the mapped letter of this axis."""
        return mapping[self.name]


class ProductAxis(BaseAxis):
    """An axis that is a product of several named axes."""

    def __init__(self, names):
        self.axes = [NamedAxis(name) for name in names]

    @property
    def axis_names(self):
        """Returns all axis names represented by this product axis."""
        return functools.reduce(
            lambda acc, axis: acc + axis.axis_names,
            self.axes, []
        )

    @property
    def flattened_axes(self):
        """Returns all NamedAxis objects represented by this product."""
        return self.axes

    @property
    def num_axes(self):
        """Returns the number of axes represented by this product."""
        return len(self.axes)

    def __repr__(self):
        """String representation of this product axis."""
        return f'(ProductAxis: {self.axes})'

    def einsum_repr(self, mapping):
        """Returns the individual mapped letters of the product axes."""
        return ''.join([axis.einsum_repr(mapping) for axis in self.axes])


class EllipsisAxis(BaseAxis):
    """A placeholder axis that can be expanded to represent some number of input/output axes."""

    @property
    def axis_names(self):
        """Returns all axis names (none) represented by this ellipse."""
        return []

    def __repr__(self):
        """Human-readable name for an ellipse axis."""
        return '(EllipsisAxis)'

    def einsum_repr(self, mapping):
        """Ellipse output in the einsum."""
        return '...'


class Variable:
    """A named variable and its axes."""

    def __init__(self, name, axes):
        self.name = name
        self.axes = axes

    @property
    def axis_names(self):
        """Returns all named axes."""
        return functools.reduce(
            lambda acc, axis: acc + axis.axis_names,
            self.axes, []
        )

    @property
    def flattened_axes(self):
        """Returns a list of all named/ellipsis axes."""
        return functools.reduce(
            lambda acc, axis: acc + axis.flattened_axes,
            self.axes, []
        )

    def __repr__(self):
        """String representation of this variable with its axes."""
        return f'({self.name}: {self.axes})'


def _parse_variable(tree):
    assert tree.data == 'variable'

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
    else:
        raise RuntimeError('Invalid parse tree for variable.')

    # Grab all axis names
    axes = []
    for tree_axis in tree_axes.children:
        if tree_axis.data == 'name':
            # Regular named axis
            axes.append(NamedAxis(tree_axis.children[0].value.lower()))
        elif tree_axis.data == 'product_axis':
            axes.append(ProductAxis([
                node.children[0].value.lower() for node in tree_axis.children
            ]))
        elif tree_axis.data == 'ellipsis':
            axes.append(EllipsisAxis())

    return Variable(name, axes)


def _idx_to_letter(axis, idx):
    if idx >= len(VALID_CHARACTERS):
        raise named_einsum.exceptions.TooManyAxesError(axis, idx)
    return VALID_CHARACTERS[idx]


def parse(inp):
    """Parse an input named einsum expression into a series of input and output variables."""
    parser = Lark_StandAlone()
    tree = parser.parse(inp)

    assert tree.data == 'einsum'
    assert len(tree.children) == 2
    assert tree.children[0].data == 'input_variables'
    assert tree.children[1].data == 'output_variable'

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
