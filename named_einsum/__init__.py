"""Main import for named_einsum."""
import functools
import named_einsum.parser
import named_einsum.exceptions
import autoray


def _generate_variable_subscripts(variable, mapping):
    out_str = ''

    for axis in variable.axes:
        out_str += axis.einsum_repr(mapping)

    return out_str


def parse(s):
    """Parse a readable einsum string into a list of variables."""
    return named_einsum.parser.parse(s)


def compile(parsed):
    """Compile a parsed string into an executable einsum statement."""
    input_var_strs = []
    for input_variable in parsed.input_variables:
        input_var_strs.append(
            _generate_variable_subscripts(input_variable, parsed.axis_mapping)
        )

    output_var_str = ('' if parsed.output_variable is None else
                      _generate_variable_subscripts(parsed.output_variable, parsed.axis_mapping))

    return ','.join(input_var_strs) + '->' + output_var_str


def shape_check(parsed, variables):
    """Check the shape of input variables against a parsed expression."""
    variable_shapes = {}
    material_variable_axes = []

    for (var_spec, var) in zip(parsed.input_variables, variables):
        ndim = var.ndim
        naxis = len(var_spec.axes)

        # Check if we have an ellipse axis
        ellipsis_axis_num = -1
        for i, axis in enumerate(var_spec.axes):
            if isinstance(axis, named_einsum.parser.EllipsisAxis):
                if ellipsis_axis_num != -1:
                    raise named_einsum.exceptions.AmbiguousEllipsesError(var_spec.name)
                ellipsis_axis_num = i

        # Check number of axes
        if ellipsis_axis_num == -1 and ndim != len(var_spec.axes):
            raise named_einsum.exceptions.InconsistentShapeDefinitionError(
                var_spec.name, len(var_spec.axes), ndim
            )

        # Fill out the ellipse axes to get a materialized list of axis that are
        # either a ProductAxis or NamedAxis
        mat_axis = []

        # Helper to check shape and materialize axes
        def check_axis(axis, size):
            mat_axis.append(axis)
            if isinstance(axis, named_einsum.parser.ProductAxis):
                # If we have a product axis, check that the product of axis shapes is consistent
                product_size = 1
                for subaxis in axis.axes:
                    axis_name = subaxis.name
                    if axis_name not in variable_shapes:
                        return
                    product_size *= variable_shapes[axis_name]
                if product_size != size:
                    raise named_einsum.exceptions.InconsistentAxisSizeError(
                        f'Product ({axis.einsum_repr})', [product_size, size]
                    )
            else:
                axis_name = axis.name

                # If this is the first time we encounter this axis, set it and return
                if axis_name not in variable_shapes:
                    variable_shapes[axis_name] = size
                    return

                # ... else check to make sure we are consistent
                previous_size = variable_shapes[axis_name]
                if previous_size != size:
                    raise named_einsum.exceptions.InconsistentAxisSizeError(
                        axis_name, (previous_size, size)
                    )

        # Finally, handle axis checking with ellipses
        shape_ptr = 0
        for i, axis in enumerate(var_spec.axes):
            if ellipsis_axis_num == -1 or i < ellipsis_axis_num:
                # Before ellipsis
                check_axis(axis, var.shape[shape_ptr])
                shape_ptr += 1
            elif ellipsis_axis_num != -1 and i >= (naxis - ellipsis_axis_num):
                # After ellipsis
                check_axis(axis, var.shape[shape_ptr])
                shape_ptr += 1
            else:
                # Current axis belongs in the ellipsis
                num_ellipse_axes = (ndim - naxis) + 1
                for j in range(num_ellipse_axes):
                    check_axis(
                        named_einsum.parser.NamedAxis(f'!ellipsis_{j}'),
                        var.shape[shape_ptr]
                    )
                    shape_ptr += 1

        material_variable_axes.append(mat_axis)

    # Reshape variables to reduce product axes
    reshaped_variables = []
    for (var_spec, var) in zip(material_variable_axes, variables):
        output_shape = []
        for axis in var_spec:
            output_shape.extend([variable_shapes[subaxis_name] for subaxis_name in axis.axis_names])
        reshaped_variables.append(var.reshape(tuple(output_shape)))

    return reshaped_variables


def compute_output_shape(parsed, var):
    """Compute the shape of the output variable if we have product axes."""
    if parsed.output_variable is None:
        # Scalar shape
        return (())

    var_spec = parsed.output_variable
    flat_axes = var_spec.flattened_axes
    naxis = len(flat_axes)

    # Check if we have an ellipse axis
    ellipsis_axis_num = -1
    for i, axis in enumerate(flat_axes):
        if isinstance(axis, named_einsum.parser.EllipsisAxis):
            if ellipsis_axis_num != -1:
                raise named_einsum.exceptions.AmbiguousEllipsesError(var_spec.name)
            ellipsis_axis_num = i

    # Fill out the ellipse axes to get a materialized list of axis that are
    # either a ProductAxis or NamedAxis
    if ellipsis_axis_num != -1:
        ellipse_axes = []
        for i in range(var.ndim - naxis + 1):
            ellipse_axes.append(named_einsum.parser.NamedAxis(
                f'!ellipsis_{i}'
            ))
        material_axes = []
        for axis in var_spec.axes:
            if isinstance(axis, named_einsum.parser.EllipsisAxis):
                material_axes.extend(ellipse_axes)
            else:
                material_axes.append(axis)
    else:
        material_axes = var_spec.axes

    # Now, collapse ProductAxes
    output_shape = []
    shape_ptr = 0
    for axis in material_axes:
        if isinstance(axis, named_einsum.parser.ProductAxis):
            size = functools.reduce(
                lambda x, y: x * y, var.shape[shape_ptr: shape_ptr + axis.num_axes], 1
            )
            output_shape.append(size)
            shape_ptr += axis.num_axes
        else:
            output_shape.append(var.shape[shape_ptr])
            shape_ptr += 1

    return tuple(output_shape)


@functools.cache
def translate(subscripts, return_parsed=False):
    """Translate a readable einsum string into something that can be executed by (i.e.) numpy."""
    parsed = parse(subscripts)
    if return_parsed:
        return compile(parsed), parsed
    return compile(parsed)


def einsum(subscripts, *args, **kwargs):
    """
    Wrapper routine for existing einsum functions.

    Parameters
    ----------
    fn : callable[[subscripts, arguments...], [output]]
      Existing einsum function to wrap
    subscripts : string
      Readable einsum subscripts string

    Returns
    -------
    array
      Output of einsum
    """
    compiled_subscripts, parsed_subscripts = translate(subscripts, True)
    reshaped_input = shape_check(parsed_subscripts, args)

    output = autoray.do('einsum', compiled_subscripts, *reshaped_input, **kwargs)
    output_shape = compute_output_shape(parsed_subscripts, output)

    return output.reshape(output_shape)


def feinsum(subscripts, *args, **kwargs):
    """
    Wrapper routine for existing einsum functions.
    "Fast" version, in that no shape checking is done.

    Parameters
    ----------
    fn : callable[[subscripts, arguments...], [output]]
      Existing einsum function to wrap
    subscripts : string
      Readable einsum subscripts string

    Returns
    -------
    array
      Output of einsum
    """
    compiled_subscripts, parsed_subscripts = translate(subscripts, True)
    return autoray.do('einsum', compiled_subscripts, *args, **kwargs)
