# named_einsum

More readable einsums: go from the jumbled mess of

```Python
np.einsum('ab,cd,eb,fd,b,d,gbd -> gacef', ...)
```

to the much more informative

```Python
named_einsum.einsum('''
  phi_ix[basis_ix, quadrature_x],
  phi_iy[basis_iy, quadrature_y],
  phi_jx[basis_jx, quadrature_x],
  phi_jy[basis_jy, quadrature_y],
  weight_x[quadrature_x],
  weight_y[quadrature_y],
  jacobian_det[element, quadrature_x, quadrature_y]
  ->
  mass[element, basis_ix, basis_iy, basis_jx, basis_jy]
''', ...)
```

Einsum dispatching is automatically performed using [autoray](https://github.com/jcmgray/autoray) on
the underlying input types.

### Syntax

A variable can be described with the syntax

```
{variable_name}[ axis_1, axis_2, axis_3, ... ]
```

where `variable_name` is an optional name for the input/output variable, and axis names are some
mixture of letters, digits, and underscores (_).  For scalar variables, the axes can be omitted
entirely.

The remainder of the einsum remains the same, and has the syntax

```
input_variable_1, input_variable_2, ... -> output_variable
```

C-style comments using `//` are supported, i.e.,

```Python
'''
// This is a comment
var_1[axis_1, axis_2],

// This is another comment
var_2[axis_2, axis_3] ->

// This is a third comment
var_out[axis_1, axis_3]
'''
```

Product axes are supported in the output.  The syntax `i*j`, for example, means to flatten axes `i`
and `j` into a single axis in the output.  This is syntactic sugar for computing the intermediate
axes `i` and `j`, then flattening them together in the output.

```Python
A[i], B[j] -> C[i * j]
```

### Examples

Structured inner product

```Python
named_einsum.einsum('''
  u[element, basis],
  v[element, basis]
  ->
  u_times_v
''', u, v)
```

Khatri-Rao product

```Python
named_einsum.einsum('''
  A[i, l],
  B[j, l]
  ->
  KRP[i * j, l]
''', A, B)
```
