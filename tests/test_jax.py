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

sys.path.insert(0, '.')
from lsqfitgp import _patch_jax
import util

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
