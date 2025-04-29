"""Main import for named_einsum."""
import functools
import named_einsum.parser


def _generate_variable_subscripts(variable, mapping):
    out_str = ''

    for axis in variable.axes:
        if isinstance(axis, str):
            out_str += mapping[axis]
        elif isinstance(axis, type(...)):
            out_str += '...'

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


def shape_check(parsed, variables):  # noqa
    """Check the shape of input variables against a parsed expression."""
    # todo...


@functools.cache
def translate(subscripts, return_parsed=False):
    """Translate a readable einsum string into something that can be executed by (i.e.) numpy."""
    parsed = parse(subscripts)
    if return_parsed:
        return compile(parsed), parsed
    return compile(parsed)


def einsum(fn, subscripts, *args, **kwargs):
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
    shape_check(parsed_subscripts, args)
    return fn(compiled_subscripts, *args, **kwargs)


def _einsum_partial(fn):
    def f(*args, **kwargs):
        return einsum(fn, *args, **kwargs)
    return f
