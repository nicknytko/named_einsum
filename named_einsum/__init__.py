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


def compile(parsed):
    '''Compile a parsed string into an executable einsum statement'''

    input_var_strs = []
    for input_variable in parsed.input_variables:
        input_var_strs.append(
            _generate_variable_subscripts(input_variable, parsed.axis_mapping)
        )

    output_var_str = ('' if parsed.output_variable is None else
                      _generate_variable_subscripts(parsed.output_variable, parsed.axis_mapping))

    return ','.join(input_var_strs) + '->' + output_var_str


def shape_check(parsed, variables):
    # todo...
    pass


def translate(subscripts):
    '''Translate a readable einsum string into something that can be executed by (i.e.) numpy'''

    return compile(parse(subscripts))


def einsum(fn, subscripts, *args, **kwargs):
    '''
    Wrapper for existing einsum functions

    parameters
    ----------
    fn : callable[[subscripts, arguments...], [output]]
      Existing einsum function to wrap
    subscripts : string
      Readable einsum subscripts string

    returns
    -------
    array
      Output of einsum
    '''

    parsed_subscripts = parse(subscripts)
    shape_check(parsed_subscripts, args)
    compiled_subscripts = compile(parsed_subscripts)
    return fn(compiled_subscripts, *args, **kwargs)


def _einsum_partial(fn):
    def f(*args, **kwargs):
        return einsum(fn, *args, **kwargs)
    return f
