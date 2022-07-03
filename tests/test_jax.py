# lsqfitgp/tests/test_jax.py
#
# Copyright (c) 2022, Giacomo Petrillo
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

import jax
from jax import test_util
import numpy as np
from scipy import special, linalg
import pytest

from lsqfitgp import _patch_jax
import util

def test_sinc():
    x = np.linspace(-0.1, 0.1, 1000)
    s1 = np.sinc(x)
    s2 = _patch_jax.sinc(x)
    np.testing.assert_allclose(s2, s1, atol=1e-15, rtol=1e-15)

def test_jvmodx2():
    nu = np.linspace(-5, 5, 20)
    x = np.linspace(1e-15, 0.1, 1000)
    for v in nu:
        s1 = (x / 2) ** -v * special.jv(v, x)
        s2 = _patch_jax.jvmodx2(v, x ** 2)
        np.testing.assert_allclose(s2, s1, atol=1e-15, rtol=1e-14)
        np.testing.assert_allclose(_patch_jax.jvmodx2(v, 0), 1 / special.gamma(v + 1), rtol=1e-14)
        test_util.check_grads(lambda x: _patch_jax.jvmodx2(v, x ** 2), (x,), 2)

def test_kvmodx2():
    nu = np.linspace(-5, 5, 20)
    x = np.linspace(1e-15, 0.1, 1000)
    xsoft = np.linspace(1, 10, 1000)
    for v in nu:
        s1 = 2 / special.gamma(v) * (x / 2) ** v * special.kv(v, x)
        s2 = _patch_jax.kvmodx2(v, x ** 2)
        np.testing.assert_allclose(s2, s1, atol=1e-15, rtol=1e-14)
        np.testing.assert_allclose(_patch_jax.kvmodx2(v, 0), 1, rtol=1e-14)
        test_util.check_grads(lambda x: _patch_jax.kvmodx2(v, x ** 2), (xsoft,), 2)
        # for no in range(5):
        #     np.testing.assert_allclose(_patch_jax.kvmodx2(v, 1e-15, no), _patch_jax.kvmodx2(v, 0, no), equal_nan=False)
    xz = np.linspace(0, 1, 1000)
    np.testing.assert_equal(_patch_jax.kvmodx2(0, xz), np.where(xz, 0, 1))

def test_kvmodx2_hi():
    x = np.linspace(1e-15, 10, 1000)
    x2 = x ** 2
    for p in range(5):
        v = p + 1/2
        f1 = lambda x: _patch_jax.kvmodx2_hi(x, p)
        f2 = lambda x: _patch_jax.kvmodx2(v, x)
        for _ in range(3):
            s1 = f1(x2)
            s2 = f2(x2)
            np.testing.assert_allclose(s1, s2, atol=0, rtol=1e-14)
            f1 = _patch_jax.elementwise_grad(f1)
            f2 = _patch_jax.elementwise_grad(f2)

def randpoly(n):
    while True:
        a = np.random.randn(n)
        if a[0] != 0:
            return a

def test_companion():
    a = randpoly(20)
    c1 = _patch_jax.companion(a)
    c2 = linalg.companion(a)
    assert c1.dtype == c2.dtype
    util.assert_equal(c1, c2)
    with pytest.raises(AssertionError):
        a[0] = 0
        _patch_jax.companion(a)

def test_polyroots():
    for n in [2, 20]:
        a = randpoly(n)
        r1 = np.polynomial.polynomial.polyroots(a[::-1])
        r2 = _patch_jax.polyroots(a[::-1])
        np.testing.assert_allclose(r1, r2, atol=1e-15, rtol=1e-12)
