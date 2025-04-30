"""Test valid compiled einsum strings."""
import named_einsum


def test_named_input():
    """Basic test of named variables."""
    assert named_einsum.translate('input_1[a], input_2[b] -> output[a,b]') == 'A,B->AB'


def test_scalar_out():
    """Test of scalar output."""
    assert named_einsum.translate('[a] ->') == 'A->'
    assert named_einsum.translate('[a] -> output') == 'A->'


def test_complex():
    """Test of a Tensor-product mass matrix evaluation."""
    assert (named_einsum.translate('''
    phi_ix[basis_ix, quad_x],
    phi_iy[basis_iy, quad_y],
    phi_jx[basis_jx, quad_x],
    phi_jy[basis_jy, quad_y],
    weight_x[quad_x],
    weight_y[quad_y],
    jacobian_det[element, quad_x, quad_y]
    ->
    mass[element, basis_ix, basis_iy, basis_jx, basis_jy]
    ''') == 'AB,CD,EB,FD,B,D,GBD->GACEF')


def test_ellipsis():
    """Test of axis broadcasting using ellipses..."""
    assert named_einsum.translate('[...,a,b], [a,b] -> [...]') == '...AB,AB->...'


def test_comment():
    """Test of comments in the einsum string."""
    assert named_einsum.translate(
        '''
        // this is a comment and shouldn't affect output!
        [a] ->
        '''
    ) == 'A->'
