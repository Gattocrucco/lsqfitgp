# lsqfitgp/tests/test_copula.py
#
# Copyright (c) 2023, Giacomo Petrillo
#
# This file is part of lsqfitgp.
#
# lsqfitgp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# lsqfitgp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.

""" Test the Copula class """

import gvar
import pytest
from pytest import mark

from lsqfitgp import copula

from .. import util

@pytest.fixture
def name(request):
    return request.node.nodeid

def test_repr_paths():
    """ check that no object is represented more than once """
    d = {}
    assert repr(copula.Copula(d)) == 'Copula({})'
    
    d['a'] = copula.beta(1, 2)
    assert repr(copula.Copula(d)) == """Copula({'a': beta(1, 2)})"""

    d['b'] = d['a']
    assert repr(copula.Copula(d)) == """\
Copula({'a': beta(1, 2), 'b': <a>})"""
    
    c1 = copula.gamma(2, 2)
    d['c'] = copula.beta(d['a'], c1)
    assert repr(copula.Copula(d)) == """\
Copula({'a': beta(1, 2), 'b': <a>, 'c': beta(<a>, gamma(2, 2))})"""

    d['d'] = c1
    assert repr(copula.Copula(d)) == """\
Copula({'a': beta(1, 2), 'b': <a>, 'c': beta(<a>, gamma(2, 2)), 'd': <c.1>})"""

def test_dict_order():
    """ check that the insertion order of a dict is preserved (jax issue) """
    d = dict(b=copula.beta(1, 2), a=copula.beta(1, 2))
    assert list(copula.Copula(d)._variables) == list(d)

def test_repr_dict_order():
    """ check that the insertion order of a dict is preserved (pprint issue) """
    d = dict(b=copula.beta(1, 2), a=copula.beta(1, 2))
    assert repr(copula.Copula(d)) == """\
Copula({'b': beta(1, 2), 'a': beta(1, 2)})"""

def test_repr_long():
    """ check that long reprs are split on newlines """
    d = {str(i): copula.beta(1, 2) for i in range(100)}
    assert repr(copula.Copula(d)).find('\n') >= 0

def test_add_distribution(name, rng):
    """ check that add_distribution works """
    c = copula.Copula({'a': copula.beta(1, 2)})
    c.add_distribution(name)
    in_samples = rng.standard_normal(c.in_shape)
    bd = gvar.BufferDict({f'{name}(x)': in_samples})
    out_samples_1 = bd['x']['a']
    out_samples_2 = c.partial_invfcn(in_samples)['a']
    util.assert_equal(out_samples_1, out_samples_2)

@mark.parametrize('broadcast_shape', [(), (2,), (2,3)])
@mark.parametrize('shape', [(), (2,), (2,3)])
@mark.parametrize('use_gvar', [False, True])
@mark.parametrize('double', [False, True])
def test_partial_invfcn(rng, broadcast_shape, shape, use_gvar, double):
    """ check that a Distr used through a Copula works the same """
    
    distr = copula.beta(1, 2, shape=shape)
    c = copula.Copula({'a': distr})
    if double:
        c = copula.Copula(c)
    
    in_samples = rng.standard_normal(broadcast_shape + c.in_shape)
    if use_gvar:
        in_samples = gvar.gvar(in_samples, rng.gamma(5, 1/5, in_samples.shape))
    
    out_samples_1 = c.partial_invfcn(in_samples)['a']
    out_samples_2 = distr.partial_invfcn(in_samples.reshape(broadcast_shape + distr.in_shape))
    
    if use_gvar:
        util.assert_same_gvars(out_samples_1, out_samples_2)
    else:
        util.assert_equal(out_samples_1, out_samples_2)

@mark.parametrize('broadcast_shape', [(), (2,), (2,3)])
def test_dependencies(rng, broadcast_shape):
    """ check that dependencies are respected """
    d = {}
    d['a'] = copula.beta(1, 2)
    d['b'] = copula.beta(d['a'], 2)
    c = copula.Copula(d)
    in_samples = rng.standard_normal(broadcast_shape + c.in_shape)
    out_samples_1 = c.partial_invfcn(in_samples)
    out_samples_2a = d['a'].partial_invfcn(in_samples[..., 0])
    out_samples_2b = d['b'].invfcn(in_samples[..., 1], out_samples_1['a'], d['b'].params[1])
    util.assert_equal(out_samples_1['a'], out_samples_2a)
    util.assert_equal(out_samples_1['b'], out_samples_2b)

@mark.parametrize('double', [False, True])
@mark.parametrize('attr', ['shape', 'distrshape', 'dtype'])
def test_out_attrs(double, attr):
    d = {
        'a': copula.beta(1, 2),
        'b': copula.beta(1, 2, shape=(2,)),
    }
    c = copula.Copula(d)
    if double:
        c = copula.Copula(c)
    assert getattr(c, attr) == {k: getattr(v, attr) for k, v in d.items()}
