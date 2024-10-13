# lsqfitgp/tests/test_gvar.py
#
# Copyright (c) 2022, 2023, 2024, Giacomo Petrillo
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

import numpy as np
import gvar
from jax import tree_util
import jax
from pytest import mark

from lsqfitgp import _gvarext

from . import util

def check_jacobian(nprim, shape, rng, *, pnz=1):
    z, s = rng.standard_normal((2, nprim))
    x = gvar.gvar(z, np.abs(s))
    t = rng.standard_normal((*shape, nprim))
    t[rng.binomial(1, 1 - pnz, t.shape).astype(bool)] = 0
    y = t @ x
    jac, indices = _gvarext.jacobian(y)
    y2 = _gvarext.from_jacobian(gvar.mean(y), jac, indices)
    util.assert_same_gvars(y, y2, atol=1e-16)

def test_jacobian(rng):
    check_jacobian(10, (5,), rng)
    check_jacobian(10, (2, 3), rng)
    check_jacobian(0, (5,), rng)
    check_jacobian(10, (0,), rng)
    check_jacobian(10, (), rng)
    check_jacobian(0, (), rng)
    check_jacobian(10, (5,), rng, pnz=0)
    check_jacobian(100, (5,), rng, pnz=0.5)

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

@mark.parametrize('shape', [(), (2,)])
def test_bdtree_compare(rng, shape):
    x = gvar.BufferDict(a=rng.standard_normal(shape))
    _, t1 = tree_util.tree_flatten(x)
    _, t2 = tree_util.tree_flatten(x)
    assert t1 == t2

@mark.parametrize('shape', [(), (2,)])
def test_double_jit(rng, shape):
    """ Call a jitted function twice with the same BufferDict to trigger the
    machinery that checks if a compiled version already exists """
    x = gvar.BufferDict(a=rng.standard_normal(shape))
    @jax.jit
    def f(x):
        return x
    y1 = f(x)
    y2 = f(x)
    util.assert_equal_bufferdict(x, y1)
    util.assert_equal_bufferdict(x, y2)

def test_tracer():
    x = gvar.BufferDict(a=0., b=1.)
    @jax.jit
    def f(x):
        return tree_util.tree_map(lambda x: x, x)
    y = f(x)
    assert x == y

def test_uformat():

    def check(n, s, string, *args, **kw):
        defaults = dict(minnegexp=2, minposexp=0)
        defaults.update(kw)
        f = _gvarext.uformat(n, s, *args, **defaults)
        assert f == string

    arglist = [
        (1, 0.2, "1.00 pm 0.20", 1.5, " pm "),
        (1, 0.3, "1.00 pm 0.30", 1.5, " pm "),
        (1, 0.31, "1.00 pm 0.31", 1.5, " pm "),
        (1, 0.32, "1.0 pm 0.3", 1.5, " pm "),
        (-1, 0.34, "-1.00 pm 0.34", 2, " pm "),
        (0, 0, "0 pm 0", 2, " pm "),
        (123456, 0, "123456. pm 0", 2, " pm "),
        (12345.6, 0, "12345.6 pm 0", 2, " pm "),
        (12345.67, 0, "12345.7 pm 0", 2, " pm "),
        (1e8, 0, "1.00000e+08 pm 0", 2, " pm "),
        (1e-2, 0, "0.0100000 pm 0", 2, " pm "),
        (1e-1, 0, "0.100000 pm 0", 2, " pm "),
        (12345.99, 0, "12346.0 pm 0", 2, " pm "),
        (0, 0.001, "(0.0 pm 1.0)e-3", 2, " pm "),
        (0, 0.01, "(0.0 pm 1.0)e-2", 2, " pm "),
        (0, 0.1, "0.00 pm 0.10", 2, " pm "),
        (0, 1, "0.0 pm 1.0", 2, " pm "),
        (0, 10, "0 pm 10", 2, " pm "),
        (0, 100, "(0.0 pm 1.0)e+2", 2, " pm "),
        (0, 1000, "(0.0 pm 1.0)e+3", 2, " pm "),
        (0, 0.0196, "(0.0 pm 2.0)e-2", 2, " pm "),
        (0, 0.196, "0.00 pm 0.20", 2, " pm "),
        (0, 1.96, "0.0 pm 2.0", 2, " pm "),
        (0, 19.6, "0 pm 20", 2, " pm "),
        (0, 196, "(0.0 pm 2.0)e+2", 2, " pm "),
        (0, 0.00996, "(0.0 pm 1.0)e-2", 2, " pm "),
        (0, 0.0996, "0.00 pm 0.10", 2, " pm "),
        (0, 0.996, "0.0 pm 1.0", 2, " pm "),
        (0, 9.96, "0 pm 10", 2, " pm "),
        (0, 99.6, "(0.0 pm 1.0)e+2", 2, " pm "),
        (0.025, 3, "0.0 pm 3.0", 2, " pm "),
        (0.0251, 0.3, "0.03 pm 0.30", 2, " pm "),
        (0.025, 0.03, "(2.5 pm 3.0)e-2", 2, " pm "),
        (0.025, 0.003, "(2.50 pm 0.30)e-2", 2, " pm "),
        (0.0025, 0.003, "(2.5 pm 3.0)e-3", 2, " pm "),
        (0.251, 3, "0.3 pm 3.0", 2, " pm "),
        (2.5, 3, "2.5 pm 3.0", 2, " pm "),
        (25, 3, "25.0 pm 3.0", 2, " pm "),
        (2500, 300, "(2.50 pm 0.30)e+3", 2, " pm "),
        (1, 0.99, "1.0 pm 1.0", 1.5, " pm "),
        (np.inf, 1.0, "inf pm 1.00000", 2, " pm "),
        (-np.inf, 1.0, "-inf pm 1.00000", 2, " pm "),
        (0, np.inf, "0 pm inf", 2, " pm "),

        (1, 0.2, "1.00(20)", 1.5, None),
        (1, 0.3, "1.00(30)", 1.5, None),
        (1, 0.31, "1.00(31)", 1.5, None),
        (1, 0.32, "1.0(3)", 1.5, None),
        (-1, 0.34, "-1.00(34)", 2, None),
        (0, 0, "0(0)", 2, None),
        (123456, 0, "123456.(0)", 2, None),
        (12345.6, 0, "12345.6(0)", 2, None),
        (12345.67, 0, "12345.7(0)", 2, None),
        (1e8, 0, "1.00000e+08(0)", 2, None),
        (1e-2, 0, "0.0100000(0)", 2, None),
        (1e-1, 0, "0.100000(0)", 2, None),
        (12345.99, 0, "12346.0(0)", 2, None),
        (0, 0.001, "0.0(1.0)e-3", 2, None),
        (0, 0.01, "0.0(1.0)e-2", 2, None),
        (0, 0.1, "0.00(10)", 2, None),
        (0, 1, "0.0(1.0)", 2, None),
        (0, 10, "0(10)", 2, None),
        (0, 100, "0.0(1.0)e+2", 2, None),
        (0, 1000, "0.0(1.0)e+3", 2, None),
        (0, 0.0196, "0.0(2.0)e-2", 2, None),
        (0, 0.196, "0.00(20)", 2, None),
        (0, 1.96, "0.0(2.0)", 2, None),
        (0, 19.6, "0(20)", 2, None),
        (0, 196, "0.0(2.0)e+2", 2, None),
        (0, 0.00996, "0.0(1.0)e-2", 2, None),
        (0, 0.0996, "0.00(10)", 2, None),
        (0, 0.996, "0.0(1.0)", 2, None),
        (0, 9.96, "0(10)", 2, None),
        (0, 99.6, "0.0(1.0)e+2", 2, None),
        (0.025, 3, "0.0(3.0)", 2, None),
        (0.0251, 0.3, "0.03(30)", 2, None),
        (0.025, 0.03, "2.5(3.0)e-2", 2, None),
        (0.025, 0.003, "2.50(30)e-2", 2, None),
        (0.0025, 0.003, "2.5(3.0)e-3", 2, None),
        (0.251, 3, "0.3(3.0)", 2, None),
        (2.5, 3, "2.5(3.0)", 2, None),
        (25, 3, "25.0(3.0)", 2, None),
        (2500, 300, "2.50(30)e+3", 2, None),
        (1, 0.99, "1.0(1.0)", 1.5, None),
        (np.inf, 1.0, "inf(1.00000)", 2, None),
        (-np.inf, 1.0, "-inf(1.00000)", 2, None),
        (0, np.inf, "0(inf)", 2, None),
    ]

    for args in arglist:
        check(*args)

def test_gvar_format():
    arglist = [
        # mean, sdev, spec, result
        (1, 12, '1p', '1(12)'),
        (1, 1234, 'p', '1(1234)'),
        (1, 1234, '#p', 'ooo(12oo)'),
        (1, 1234, ':2p', '0.0(1.2)e+3'),
        (1, 1234, ':2u', '(0.0 ± 1.2)e+3'),
        (1, 1234, ':2U', '(0.0 ± 1.2)×10³'),
        (1, 1234, '$:2U', '0.0×10³ ± 1.2×10³'),
        (1, 1234, '+$:2U', '+0.0×10³ ± 1.2×10³'),
        (-1, 1234, ':2u', '(-0.0 ± 1.2)e+3'),
        (-1, 1234, '-:2u', '-(0.0 ± 1.2)e+3'),
    ]
    for mean, sdev, spec, result in arglist:
        g = gvar.gvar(mean, sdev)
        pre = format(g)
        with _gvarext.gvar_format(spec):
            assert format(g) == result
        assert format(g) == pre

    # TODO check without lsqfitgp formatting rules

# TODO test gvar_gufunc
