# lsqfitgp/tests/bayestree/test_bart.py
#
# Copyright (c) 2024, 2026, Giacomo Petrillo
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

"""Test `lgp.bayestree.bart`."""

import pytest
from jax import numpy as jnp
from jax import random

from lsqfitgp import bayestree
from tests import util


def gen_X(key, p, n):
    """Generate random covariates uniform in [-2, 2] with shape (p, n)."""
    return random.uniform(key, (p, n), float, -2, 2)


def f(X):
    """Return the true regression function."""
    T = 2
    return jnp.sum(jnp.cos(2 * jnp.pi / T * X), axis=0)


def gen_y(key, X):
    """Generate outcomes as `f(X)` plus Gaussian noise."""
    sigma = 0.1
    return f(X) + sigma * random.normal(key, X.shape[1:])


@pytest.fixture
def n():
    """Return the number of data points."""
    return 101


@pytest.fixture
def p():
    """Return the number of covariates."""
    return 11


@pytest.fixture
def X(n, p, key):
    """Return random covariates."""
    key = random.fold_in(key, 0xD9B0963D)
    return gen_X(key, p, n)


@pytest.fixture
def y(X, key):
    """Return random outcomes."""
    key = random.fold_in(key, 0x1391BC96)
    return gen_y(key, X)


@pytest.fixture
def kw():
    """Return additional keyword arguments for `bart` (none)."""
    return dict()


def test_scale_shift(X, y, kw):
    """Check that `bart` is equivariant to an affine transformation of the outcome."""
    X = X.T
    bart1 = bayestree.bart(X, y, **kw)

    offset = 0.4703189
    scale = 0.5294714
    tilde_y = offset + y * scale
    bart2 = bayestree.bart(X, tilde_y, **kw)

    m1, cov1 = bart1.pred()
    m2, cov2 = bart2.pred()
    util.assert_allclose(m2, offset + m1 * scale, rtol=1e-10)
    util.assert_allclose(cov2, cov1 * scale**2, rtol=1e-11)
