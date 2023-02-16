# lsqfitgp/tests/test_deriv.py
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

import sys

import numpy as np
from jax import numpy as jnp
import gvar
from scipy import stats
import pytest

sys.path.insert(0, '.')
import lsqfitgp as lgp

import util

def test_manyargs():
    with pytest.raises(ValueError):
        lgp.Deriv(1, 2)

def test_alienargs():
    with pytest.raises(TypeError):
        lgp.Deriv((None,))

def test_manyintegers():
    with pytest.raises(ValueError):
        lgp.Deriv((1, 2))

def test_alienarg():
    with pytest.raises(TypeError):
        lgp.Deriv(object)

def test_orphan():
    with pytest.raises(ValueError):
        lgp.Deriv((1,))
    with pytest.raises(ValueError):
        lgp.Deriv(('ciao', 1))

def test_length():
    assert len(lgp.Deriv([1, 'ciao', 2, 'pippo'])) == 2

def test_compare():
    assert not lgp.Deriv() == 'cippa'

def test_repr():
    assert repr(lgp.Deriv()) == '{}'
