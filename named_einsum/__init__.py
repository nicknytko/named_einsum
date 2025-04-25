import functools
from named_einsum.parser import parse
from types import SimpleNamespace


def _generate_variable_subscripts(variable, mapping):
    out_str = ''

    for axis in variable.axes:
        if isinstance(axis, str):
            out_str += mapping[axis]
        elif isinstance(axis, type(...)):
            out_str += '...'

    return out_str


@functools.cache
def compile(s):
    parsed = parse(s)

    input_var_strs = []
    for input_variable in parsed.input_variables:
        input_var_strs.append(
            _generate_variable_subscripts(input_variable, parsed.axis_mapping)
        )

    output_var_str = ('' if parsed.output_variable is None else
                      _generate_variable_subscripts(parsed.output_variable, parsed.axis_mapping))

    return ','.join(input_var_strs) + '->' + output_var_str


def einsum(fn, subscripts, *args, **kwargs):
    compiled_subscripts = compile(subscripts)
    return fn(compiled_subscripts, *args, **kwargs)


def _einsum_partial(fn):
    def f(*args, **kwargs):
        return einsum(fn, *args, **kwargs)
    return f
