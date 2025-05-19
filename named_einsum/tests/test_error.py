"""Tests for error handling."""
import named_einsum
import named_einsum.exceptions
import pytest
import numpy as np


def test_axis_not_found():
    """Test for axis in output variable not found in any input."""
    with pytest.raises(named_einsum.exceptions.AxisNotFoundError):
        named_einsum.translate('[A], [A] -> [B]')


def test_too_many_axis():
    """Test for too many axes to represent with an einsum."""
    named_einsum.translate('''
    [
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,
    aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao,ap,aq,ar,as,at,au,av,aw,ax,ay,az
    ] ->''')  # <-- this is OK, isomorphic to a-zA-Z

    with pytest.raises(named_einsum.exceptions.TooManyAxesError):
        named_einsum.translate('''
        [
        a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,
        aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao,ap,aq,ar,as,at,au,av,aw,ax,ay,az,
        ba
        ] ->''')  # <-- this is not OK


def test_invalid_parse():
    """Several tests for invalid expressions."""
    with pytest.raises(Exception):
        named_einsum.translate('var[1,2,3] ->')
    with pytest.raises(Exception):
        named_einsum.translate('var[1,2,3 ->')


def test_ambiguous_ellipse_shape():
    """Test for ambiguous ellipses."""
    with pytest.raises(named_einsum.exceptions.AmbiguousEllipsesError):
        parsed = named_einsum.parse('var[..., a, ...] ->')
        named_einsum.shape_check(parsed, np.empty((2, 2, 2)))


def test_inconsistent_shape():
    """Test for inconsistent axis sizes."""
    with pytest.raises(named_einsum.exceptions.InconsistentAxisSizeError):
        named_einsum.einsum(np.einsum, '[a], [a] ->', np.empty(5), np.empty(10))
