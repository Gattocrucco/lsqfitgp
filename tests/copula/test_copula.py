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

import numpy as np
import gvar
import jax
from jax import numpy as jnp
import pytest
from pytest import mark
from scipy import stats, special

import lsqfitgp as lgp

from .. import util

def test_repr():
    c = lgp.copula.Copula()
    assert repr(c) == 'Copula()'
    
    c['a'] = lgp.copula.beta(1, 2)
    assert repr(c) == """Copula({
    'a': beta(1, 2),
})"""
    
    c['b'] = c['a']
    assert repr(c) == """Copula({
    'a': beta(1, 2),
    'b': <a>,
})"""
    
    c1 = lgp.copula.gamma(2, 2)
    c['c'] = lgp.copula.beta(c['a'], c1)
    assert repr(c) == """Copula({
    'a': beta(1, 2),
    'b': <a>,
    'c': beta(<a>, gamma(2, 2)),
})"""

    c['d'] = c1
    assert repr(c) == """Copula({
    'a': beta(1, 2),
    'b': <a>,
    'c': beta(<a>, gamma(2, 2)),
    'd': <c.1>,
})"""

def test_no_overwrite():
    c = lgp.copula.Copula()
    c['a'] = lgp.copula.beta(1, 2)
    with pytest.raises(KeyError):
        c['a'] = lgp.copula.beta(1, 2)

def test_no_circular_references():
    c = lgp.copula.Copula()
    with pytest.raises(ValueError):
        c['b'] = c
    c['b'] = lgp.copula.Copula()
    with pytest.raises(ValueError):
        c['b']['c'] = c
    with pytest.raises(ValueError):
        c['b']['c'] = c['b']

def test_immutable():
    c = lgp.copula.Copula()
    c = c.freeze()
    with pytest.raises(TypeError):
        c['a'] = lgp.copula.beta(1, 2)
