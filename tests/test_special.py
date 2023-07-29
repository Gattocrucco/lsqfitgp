# lsqfitgp/tests/test_jax.py
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

# TODO maybe use sympy as reference, since it's better maintained than mpmath

import jax
from jax import test_util
import numpy as np
from scipy import special, linalg
import pytest
from pytest import mark
import mpmath

from lsqfitgp import _special, _patch_jax
from . import util

gen = np.random.default_rng(202207201419)

def test_sinc():
    x = np.linspace(-0.1, 0.1, 1000)
    s1 = np.sinc(x)
    s2 = _special.sinc(x)
    util.assert_allclose(s2, s1, atol=1e-15, rtol=1e-15)

def test_jvmodx2():
    nu = np.linspace(-5, 5, 20)
    x = np.linspace(1e-15, 0.1, 1000)
    for v in nu:
        s1 = (x / 2) ** -v * special.jv(v, x)
        s2 = _special.jvmodx2(v, x ** 2)
        util.assert_allclose(s2, s1, atol=1e-15, rtol=1e-14)
        util.assert_allclose(_special.jvmodx2(v, 0), 1 / special.gamma(v + 1), rtol=1e-14)
        test_util.check_grads(lambda x: _special.jvmodx2(v, x ** 2), (x,), 2)

def test_kvmodx2():
    nu = np.linspace(-5, 5, 20)
    x = np.linspace(1e-15, 0.1, 1000)
    xsoft = np.linspace(1, 10, 1000)
    for v in nu:
        s1 = 2 / special.gamma(v) * (x / 2) ** v * special.kv(v, x)
        s2 = _special.kvmodx2(v, x ** 2)
        util.assert_allclose(s2, s1, atol=1e-15, rtol=1e-14)
        util.assert_allclose(_special.kvmodx2(v, 0), 1, rtol=1e-14)
        test_util.check_grads(lambda x: _special.kvmodx2(v, x ** 2), (xsoft,), 2)
        if v >= 0.5: # negative diverges, and below 0.5 d/dx at 0 is inf
            for no in range(5):
                util.assert_allclose(_special.kvmodx2(v, 1e-15, no), _special.kvmodx2(v, 0, no), rtol=1e-11)
        # TODO really need to check at 0 in other cases
    xz = np.linspace(0, 1, 1000)
    np.testing.assert_equal(_special.kvmodx2(0, xz), np.where(xz, 0, 1))

def test_kvmodx2_hi():
    x = np.linspace(1e-15, 10, 1000)
    x2 = x ** 2
    for p in range(5):
        v = p + 1/2
        f1 = lambda x: _special.kvmodx2_hi(x, p)
        f2 = lambda x: _special.kvmodx2(v, x)
        for _ in range(3):
            s1 = f1(x2)
            s2 = f2(x2)
            util.assert_allclose(s1, s2, atol=0, rtol=1e-14)
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
        y1 = _special.exp1_imag(x)
        y2 = special.exp1(-1j * x)
        util.assert_allclose(y1, y2, atol=0, rtol=1e-14)
        
        y1 = _special.ci(x)
        _, y2 = special.sici(x)
        util.assert_allclose(y1, y2, atol=1e-15, rtol=1e-15)

def test_expm1x():
    x = np.linspace(-2, 2, 10000)
    y = _special.expm1x(x)
    y2 = x * x / 2 * special.hyp1f1(1, 3, x)
    np.testing.assert_array_max_ulp(y, y2, 11)
    y = _special.expm1x(x.astype('f'))
    np.testing.assert_array_max_ulp(y, y2.astype('f'), 4)
    test_util.check_grads(_special.expm1x, (x,), 2)

zeta = np.vectorize(lambda *args: float(mpmath.zeta(*args)))

def test_hurwitz_zeta():
    s = np.linspace(-10, 0, 100)[:, None]
    a = np.linspace(0, 1, 100)
    z1 = zeta(s, a)
    z2 = _special.hurwitz_zeta(s, a)
    eps = np.finfo(float).eps
    tol = 350 * eps * np.max(np.abs(z1), 1)
    maxdiff = np.max(np.abs(z2 - z1), 1)
    assert np.all(maxdiff < tol)

def test_hurwitz_zeta_vectorized():
    s = np.linspace(-10, 0, 100)
    a = np.linspace(0, 1, 100)
    z1 = _special.hurwitz_zeta(s, a)
    z2 = np.vectorize(_special.hurwitz_zeta)(s, a)
    np.testing.assert_array_max_ulp(z1, z2, 780)
    # TODO what?? 675 ULP?? what?? => 780 with latest ubuntu release!

def test_gamma():
    x = np.linspace(-100, 100, 1000)
    g1 = special.gamma(x)
    g2 = _special.gamma(x)
    np.testing.assert_array_max_ulp(g2, g1, 1500)

def _periodic_zeta_mpmath(x, s):
    with mpmath.workdps(32):
        arg = mpmath.exp(2j * mpmath.pi * x)
        return complex(mpmath.polylog(s, arg))
        # the doubled working precision serves two purposes:
        # 1) compute accurately arg, for example, with x=1 the result would
        #    have a small imaginary part which changes the result significantly
        # 2) polylog is inaccurate for s near an integer (mpmath issue #634)

def _zeta_mpmath(s):
    return complex(mpmath.zeta(s))

@np.vectorize
def periodic_zeta(x, s):
    if int(x) == x:
        return _zeta_mpmath(s)
        # patch for mpmath.polylog(s, 1) != zeta(s)
    else:
        return _periodic_zeta_mpmath(x, s)

@mark.parametrize('i', [
    pytest.param(False, id='real'),
    pytest.param(True, id='imag'),
])
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
    pytest.param(1 + 1e-15, id='near1'),
    pytest.param(0.5 + np.arange(1, 15), id='half'),
    pytest.param(np.arange(2, 15, 2), id='even'),
    pytest.param(np.arange(3, 15, 2), id='odd'),
])
def test_periodic_zeta(s, d, sgn, i):
    if d == 0 and sgn < 0:
        pytest.skip()

    x = np.linspace(-1, 2, 52)
    s = np.atleast_1d(s)[:, None] + sgn * d
    
    if np.any(s <= 1):
        pytest.skip()
    
    z1 = periodic_zeta(x, s)
    z1 = z1.imag if i else z1.real
    z2 = _special.periodic_zeta(x, s, i)
    
    eps = np.finfo(float).eps
    tol = 110 * eps * np.max(np.abs(z1), 1)
    maxdiff = np.max(np.abs(z2 - z1), 1)
    assert np.all(maxdiff < tol)

@mark.parametrize('i', [
    pytest.param(False, id='real'),
    pytest.param(True, id='imag'),
])
def test_periodic_zeta_deriv(i):
    x = np.linspace(-1, 2, 52)
    s = np.linspace(2.01, 16, 20)[:, None]

    z1 = 2j * np.pi * periodic_zeta(x, s - 1)
    z1 = z1.imag if i else z1.real
    z2 = _patch_jax.elementwise_grad(_special.periodic_zeta)(x, s, i)
    
    eps = np.finfo(float).eps
    tol = 100 * eps * np.max(np.abs(z1), 1)
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
    z2 = _special.zeta_zero(s)
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
    pytest.param(2, id='2'),
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
    g2 = _special.gamma_incr(x, e)
    np.testing.assert_array_max_ulp(g2, g1, 7)

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
    g2 = _special.gammaln1(x)
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
    pytest.param(np.linspace(0.02, 0.5, 25), id='xother'),
])
@mark.parametrize('q', [
    pytest.param(0, id='qzero'),
    pytest.param(1, id='qone'),
    pytest.param(np.arange(2, 11, 2), id='qeven'),
    pytest.param(np.arange(3, 11, 2), id='qodd'),
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
    p2 = _special.zeta_series_power_diff(x, q, a)
    if np.all(q <= 1) and np.any(x):
        tol = 20 * np.max(np.abs(p1), 0) * np.finfo(float).eps
        maxdiff = np.max(np.abs(p1 - p2), 0)
        assert np.all(maxdiff < tol)
    else:
        tol = np.finfo(float).eps ** 2 / 2
        cond = np.abs(p1) < tol
        cp1 = np.where(cond, tol, p1)
        cp2 = np.where(cond, tol, p2)
        np.testing.assert_array_max_ulp(cp2, cp1, 22)

@mark.parametrize('s', [
    pytest.param(-2, id='-2'),
    pytest.param(-np.arange(4, 11, 2), id='small'),
    pytest.param(-np.arange(12, 101, 2), id='large'),
])
def test_zeta_zeros(s):
    @np.vectorize
    def func(s):
        with mpmath.workdps(20):
            return float(mpmath.diff(mpmath.zeta, s))
    def handwritten(s):
        pi = 2 * (2 * np.pi) ** (s - 1)
        n = np.around(s)
        sgn = np.where(n % 4, -1, 1)
        cos = np.pi / 2 * sgn
        gamma = _special.gamma(1 - s)
        zeta = _special.zeta(1 - s)
        return pi * cos * gamma * zeta
    z0 = handwritten(s)
    z1 = func(s)
    ulp = 25 if np.all(s >= -10) else 1113
    np.testing.assert_array_max_ulp(z0, z1, ulp)
    eps = 1e-30
    z2 = _special.zeta(eps, s) / eps
    np.testing.assert_array_max_ulp(z2, z1, ulp)

def test_zeta():
    n = np.arange(-10, 60)[:, None]
    s = np.linspace(-0.5, 0.5, 101)
    @np.vectorize
    def func(s, n):
        with mpmath.workdps(32):
            n = mpmath.mpmathify(n)
            s = mpmath.mpmathify(s)
            return float(mpmath.zeta(s + n)) if n + s != 1 else np.inf
    z1 = func(s, n)
    z2 = _special.zeta(s, n)
    np.testing.assert_array_max_ulp(z2, z1, 72)

def bernoulli_poly_handwritten(n, x):
    return [
        lambda x: 1,
        lambda x: x - 1/2,
        lambda x: x**2 - x + 1/6,
        lambda x: x**3 - 3/2 * x**2 + 1/2 * x,
        lambda x: x**4 - 2 * x**3 + x**2 - 1/30,
        lambda x: x**5 - 5/2 * x**4 + 5/3 * x**3 - 1/6 * x,
        lambda x: x**6 - 3 * x**5 + 5/2 * x**4 - 1/2 * x**2 + 1/42
    ][n](x)

def check_bernoulli(n, x):
    r1 = bernoulli_poly_handwritten(n, x)
    r2 = _special.periodic_bernoulli(n, x)
    util.assert_allclose(r1, r2, atol=1e-15, rtol=1e-9)

def test_bernoulli():
    for n in range(7):
        x = gen.uniform(0, 1, size=100)
        check_bernoulli(n, x)

@np.vectorize
def expint(n, z):
    return complex(mpmath.expint(n, z))

def test_expn():
    x = gen.uniform(0, 30, 10)
    n = np.arange(2, 12)
    result_64 = np.array([_special.expn_imag(n, x) for n in n])
    result_32 = np.array([_special.expn_imag(n.astype('i4'), x.astype('f')) for n in n])
    assert result_64.dtype == 'complex128'
    assert result_32.dtype == 'complex64'
    sol = expint(n[:, None], -1j * x)
    util.assert_allclose(result_64, sol, rtol=1e-7) # TODO quite bad
    util.assert_allclose(result_32, sol, atol=0.01) # TODO very bad!!

def test_kvp():
    test_util.check_grads(lambda z: _special.kv(3.2, z), (1.5,), 2)

# TODO these tests currently take 4 min out of 17 total. I guess the bottleneck
# is mpmath. I should produce and commit a cache of values. Maybe the "cache"
# fixture from pytest is appropriate?
