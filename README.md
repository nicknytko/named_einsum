# named_einsum

More readable einsums: go from the jumbled mess of

```Python
np.einsum('ab,cd,eb,fd,b,d,gbd -> gacef', ...)
```

to the much more informative

```Python
named_einsum.einsum(np.einsum, '''
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

### Syntax

A variable can be described with the syntax

```
{variable_name}[ axis_1, axis_2, axis_3, ... ]
```

where `variable_name` is an optional name for the input/output variable, and axis names are some mixture of letters, digits, and underscores (_).
For scalar variables, the axes can be omitted entirely.

The remainder of the einsum remains the same, and has the syntax

```
input_variable_1, input_variable_2, ... -> output_variable
```

### Examples

Structured inner product

```Python
named_einsum.einsum(np.einsum, '''
  u[element, basis],
  v[element, basis]
  ->
  u_times_v
''', u, v)
```
