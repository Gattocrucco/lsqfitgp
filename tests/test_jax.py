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
from pytest import mark
import mpmath

from lsqfitgp import _patch_jax
import util

gen = np.random.default_rng(202207201419)

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
        if v >= 0.5: # negative diverges, and below 0.5 d/dx at 0 is inf
            for no in range(5):
                np.testing.assert_allclose(_patch_jax.kvmodx2(v, 1e-15, no), _patch_jax.kvmodx2(v, 0, no), equal_nan=False)
        # TODO really need to check at 0 in other cases
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
        a = gen.random.standard_normal(n)
        if a[0] != 0:
            return a

def test_exp1_imag_and_ci():
    xs = [
        np.linspace(1e-15, 100, 1000),
        np.logspace(2, 20, 1000),
    ]
    for x in xs:
        y1 = _patch_jax.exp1_imag(x)
        y2 = special.exp1(-1j * x)
        np.testing.assert_allclose(y1, y2, atol=0, rtol=1e-14)
        
        y1 = _patch_jax.ci(x)
        _, y2 = special.sici(x)
        np.testing.assert_allclose(y1, y2, atol=1e-15, rtol=1e-15)

def test_expm1x():
    x = np.linspace(-2, 2, 10000)
    y = _patch_jax.expm1x(x)
    y2 = x * x / 2 * special.hyp1f1(1, 3, x)
    np.testing.assert_array_max_ulp(y, y2, 8)
    y = _patch_jax.expm1x(x.astype('f'))
    np.testing.assert_array_max_ulp(y, y2.astype('f'), 4)
    test_util.check_grads(_patch_jax.expm1x, (x,), 2)

zeta = np.vectorize(lambda *args: float(mpmath.zeta(*args)))

def test_hurwitz_zeta():
    s = np.linspace(-10, 0, 100)[:, None]
    a = np.linspace(0, 1, 100)
    z1 = zeta(s, a)
    z2 = _patch_jax.hurwitz_zeta(s, a)
    eps = np.finfo(float).eps
    tol = 60 * eps * np.max(np.abs(z1), 1)
    maxdiff = np.max(np.abs(z2 - z1), 1)
    assert np.all(maxdiff < tol)

def test_hurwitz_zeta_vectorized():
    s = np.linspace(-10, 0, 100)
    a = np.linspace(0, 1, 100)
    z1 = _patch_jax.hurwitz_zeta(s, a)
    z2 = np.vectorize(_patch_jax.hurwitz_zeta)(s, a)
    np.testing.assert_array_max_ulp(z1, z2, 500)
    # TODO what?? 500 ULP?? what??

def test_gamma():
    x = np.linspace(-100, 100, 1000)
    g1 = special.gamma(x)
    g2 = _patch_jax.gamma(x)
    np.testing.assert_array_max_ulp(g2, g1, 1500)

@np.vectorize
def periodic_zeta_real(x, s):
    with mpmath.workdps(32):
        arg = mpmath.exp(2j * mpmath.pi * x)
        return float(mpmath.polylog(s, arg).real)
        # the doubled working precision serves two purposes:
        # 1) compute accurately arg, for example, with x=1 the result would
        #    have a small imaginary part which changes the result significantly
        # 2) polylog is inaccurate for s near an integer (mpmath issue #634)

@mark.parametrize('sgn', [
    pytest.param(1, id='pos'),
    pytest.param(-1, id='neg'),
])
@mark.parametrize('d', [
    pytest.param(0, id='at'),
    pytest.param(1e-4, id='close'),
    pytest.param(1e-13, id='veryclose'),
])
@mark.parametrize('s', [
    pytest.param(0.5 + np.arange(1, 15), id='half'),
    pytest.param(np.arange(2, 15, 2), id='even'),
    pytest.param(np.arange(3, 15, 2), id='odd', marks=mark.xfail),
])
def test_periodic_zeta(s, d, sgn):
    if d == 0 and sgn < 0:
        pytest.skip()
    x = np.linspace(-1, 2, 52)
    s = s[:, None] + sgn * d
    z1 = periodic_zeta_real(x, s)
    z2 = _patch_jax.periodic_zeta_real(x, s)
    eps = np.finfo(float).eps
    tol = 60 * eps * np.max(np.abs(z1), 1)
    maxdiff = np.max(np.abs(z2 - z1), 1)
    assert np.all(maxdiff < tol)

@mark.parametrize('s', [
    pytest.param((1 - 1e-15) * np.linspace(-1, 1, 101), id='widerange'),
    pytest.param(0.5 * np.linspace(-1, 1, 101), id='medrange'),
    pytest.param(1e-8 * np.linspace(-1, 1, 101), id='shortrange'),
    pytest.param(1e-14 * np.linspace(-1, 1, 101), id='tinyrange'),
])
def test_zeta_zero(s):
    with mpmath.workdps(40):
        z1 = np.array([float(mpmath.zeta(s) - mpmath.zeta(0)) for s in s])
    z2 = _patch_jax._zeta_zero(s)
    np.testing.assert_array_max_ulp(z1, z2, 2)

@mark.parametrize('s', [
    pytest.param(1, id='pos'),
    pytest.param(-1, id='neg'),
])
@mark.parametrize('e', [
    pytest.param(0.5 * np.linspace(0, 1, 51), id='medrange'),
    pytest.param(1e-8 * np.linspace(0, 1, 51), id='shortrange'),
    pytest.param(1e-14 * np.linspace(0, 1, 51), id='tinyrange'),
])
@mark.parametrize('x', [
    pytest.param(2, id='nearpole'),
    pytest.param(np.arange(3, 15), id='farpole'),
])
def test_gamma_incr(x, e, s):
    x = np.reshape(x, (-1, 1))
    @np.vectorize
    def func(x, e):
        with mpmath.workdps(40):
            x = mpmath.mpf(float(x))
            e = mpmath.mpf(float(e))
            denom = mpmath.gamma(x) * mpmath.gamma(1 + e)
            return float(mpmath.gamma(x + e) / denom - 1)
    e = e * s
    e = np.where(x == 1, np.abs(e), e)
    g1 = func(x, e)
    g2 = _patch_jax._gamma_incr(x, e)
    np.testing.assert_array_max_ulp(g2, g1, 5)

@mark.parametrize('s', [
    pytest.param(1, id='pos'),
    pytest.param(-1, id='neg'),
])
@mark.parametrize('x', [
    pytest.param(0.5 * np.linspace(0, 1, 51), id='medrange'),
    pytest.param(1e-8 * np.linspace(0, 1, 51), id='shortrange'),
    pytest.param(1e-14 * np.linspace(0, 1, 51), id='tinyrange'),
])
def test_gammaln1(x, s):
    @np.vectorize
    def func(x):
        with mpmath.workdps(40):
            x = mpmath.mpf(float(x))
            return float(mpmath.loggamma(1 + x))
    x = x * s
    g1 = func(x)
    g2 = _patch_jax._gammaln1(x)
    np.testing.assert_array_max_ulp(g2, g1, 2)

@mark.parametrize('s', [
    pytest.param(1, id='pos'),
    pytest.param(-1, id='neg'),
])
@mark.parametrize('a', [
    pytest.param(0.5 * np.linspace(0.04, 1, 25), id='medrange'),
    pytest.param(1e-8 * np.linspace(0.04, 1, 25), id='shortrange'),
    pytest.param(1e-14 * np.linspace(0.04, 1, 25), id='tinyrange'),
])
@mark.parametrize('x', [
    pytest.param(0, id='xzero'),
    pytest.param(np.logspace(-100, -3, 25), id='xnearzero'),
    pytest.param(0.5, id='xhalf'),
    pytest.param(np.linspace(0.02, 0.48, 24), id='xother'),
])
@mark.parametrize('q', [
    pytest.param(0, id='qzero'),
    pytest.param(np.arange(2, 11, 2), id='qeven')
])
def test_power_diff(x, q, a, s):
    if np.any(q == 0) and np.any(s > 0):
        pytest.skip()
    x = np.reshape(x, (-1, 1, 1))
    q = np.reshape(q, (-1, 1))
    a = a * s # don't use inplace *= since arguments are not copied
    @np.vectorize
    def func(x, q, a):
        with mpmath.workdps(40):
            x = mpmath.mpf(float(x))
            q = mpmath.mpf(float(q))
            a = mpmath.mpf(float(a))
            num = mpmath.gamma(1 + q - a)
            denom = mpmath.gamma(1 - a) * mpmath.gamma(1 + q)
            zeta = mpmath.zeta(a)
            term = 2 * num / denom * zeta * x ** q
            power = x ** (q - a)
            return float(power + term)
    p1 = func(x, q, a)
    p2 = _patch_jax._power_diff(x, q, a)
    if not np.any(q) and np.any(x):
        tol = 3 * np.max(np.abs(p1), 0) * np.finfo(float).eps
        assert np.all(np.abs(p1 - p2) < tol)
    else:
        tol = np.finfo(float).eps ** 2 / 2
        p1 = np.maximum(p1, tol)
        p2 = np.maximum(p2, tol)
        np.testing.assert_array_max_ulp(p2, p1, 16)

# TODO test expn
