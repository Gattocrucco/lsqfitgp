# lsqfitgp/tests/test_gvar.py
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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
import gvar
from jax import tree_util
import jax

sys.path.insert(0, '.')
from lsqfitgp import _patch_gvar
import util

def check_jacobian(nprim, shape, pnz=1):
    z, s = np.random.randn(2, nprim)
    x = gvar.gvar(z, np.abs(s))
    t = np.random.randn(*shape, nprim)
    t[np.random.binomial(1, 1 - pnz, t.shape).astype(bool)] = 0
    y = t @ x
    jac, indices = _patch_gvar.jacobian(y)
    y2 = _patch_gvar.from_jacobian(gvar.mean(y), jac, indices)
    util.assert_same_gvars(y, y2, atol=1e-16)

def test_jacobian():
    check_jacobian(10, (5,))
    check_jacobian(10, (2, 3))
    check_jacobian(0, (5,))
    check_jacobian(10, (0,))
    check_jacobian(10, ())
    check_jacobian(0, ())
    check_jacobian(10, (5,), 0)
    check_jacobian(100, (5,), 0.5)

def test_bdtree():
    x = gvar.BufferDict(a=[1, 2], b=[3, 4])
    l, t = tree_util.tree_flatten(x)
    y = tree_util.tree_unflatten(t, l)
    assert np.all(x.buf == y.buf)
    assert x.keys() == y.keys()
    for k in x:
        assert x.slice_shape(k) == y.slice_shape(k)

def test_bdtree_dtype():
    x = gvar.BufferDict(dict(a=0), dtype=bool)
    l, t = tree_util.tree_flatten(x)
    y = tree_util.tree_unflatten(t, l)
    assert x.dtype == y.dtype

def test_tracer():
    x = gvar.BufferDict(a=0., b=1.)
    @jax.jit
    def f(x):
        return tree_util.tree_map(lambda x: x, x)
    y = f(x)
    assert x == y
