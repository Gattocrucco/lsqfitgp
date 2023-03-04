# lsqfitgp/tests/test_jax.py
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
import gvar
import jax
from jax import numpy as jnp
from pytest import mark

sys.path.insert(0, '.')
from lsqfitgp import _patch_jax
import lsqfitgp as lgp
import util

rng = np.random.default_rng(202303031632)

def test_elementwise_grad_1():
    def f(x):
        return 2 * x
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(f)(x)
    y2 = jax.vmap(jax.grad(f))(x)
    util.assert_equal(y, y2)

def test_elementwise_grad_2():
    def f(x, z):
        return 2 * x * z
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(f, 0)(x, x)
    y2 = jax.vmap(jax.grad(f, 0))(x, x)
    util.assert_equal(y, y2)

def test_elementwise_grad_3():
    def f(x):
        return 2 * x
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(_patch_jax.elementwise_grad(f))(x)
    y2 = jax.vmap(jax.grad(jax.grad(f)))(x)
    util.assert_equal(y, y2)

def test_elementwise_grad_4():
    def f(x, z):
        return 2 * x * z
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(_patch_jax.elementwise_grad(f, 0), 0)(x, x)
    y2 = jax.vmap(jax.grad(jax.grad(f, 0), 0))(x, x)
    util.assert_equal(y, y2)

def test_elementwise_grad_5():
    def f(x, z):
        return 2 * x * z
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(_patch_jax.elementwise_grad(f, 0), 1)(x, x)
    y2 = jax.vmap(jax.grad(jax.grad(f, 0), 1))(x, x)
    util.assert_equal(y, y2)

def test_elementwise_grad_6():
    def f(x, z):
        return 2 * x * z
    x = np.arange(8.)
    with jax.checking_leaks():
        y = jax.jacrev(_patch_jax.elementwise_grad(f, 0), 1)(x, x)
    y2 = jax.jacrev(jax.vmap(jax.grad(f, 0)), 1)(x, x)
    util.assert_equal(y, y2)

def test_vectorize():
    def func(x, y):
        return x @ y
    kw = dict(signature='(p),(p)->()')
    func1 = jnp.vectorize(func, **kw)
    func2 = _patch_jax.vectorize(func, **kw)
    x, y = rng.standard_normal((2, 20, 7))
    args = (x[None, :, :], y[:, None, :])
    p1 = func1(*args)
    p2 = func2(*args)
    util.assert_equal(p1, p2)

@mark.parametrize('maxnbytes', [1, 10 * 8, 1000 * 8])
def test_batcher(maxnbytes):
    def func(x, y):
        return x * y
    batched = _patch_jax.batchufunc(func, maxnbytes=maxnbytes)
    x, y = rng.standard_normal((2, 20, 7))
    args = (x[None, :, :], y[:, None, :])
    p1 = func(*args)
    p2 = batched(*args)
    util.assert_equal(p1, p2)

@mark.parametrize('maxnbytes', [1, 10 * 8, 1000 * 8])
def test_batcher_structured(maxnbytes):
    def func(x, y):
        return jnp.sum(x['x'] * y['x'], axis=-1)
    batched = _patch_jax.batchufunc(func, maxnbytes=maxnbytes)
    xy = rng.standard_normal((2, 20, 7)).view([('x', float, 7)]).squeeze(-1)
    xy = lgp.StructuredArray(xy)
    x, y = xy
    args = (x[None, :], y[:, None])
    p1 = func(*args)
    p2 = batched(*args)
    util.assert_allclose(p1, p2, atol=2e-15)
