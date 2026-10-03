# lsqfitgp/tests/copula/test_beta.py
#
# Copyright (c) 2023, 2026, Giacomo Petrillo
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

"""Test the `copula._beta` module."""

import numpy as np
import pytest
from jax import test_util
from scipy import stats

from lsqfitgp.copula import _beta


@pytest.fixture
def aby():
    """Return the parameters `a`, `b` and the probability `y`."""
    return 2.5, 1.3, 0.3


def test_grad(aby):
    """Check the first and second derivatives of `betaincinv` w.r.t. `y`."""
    a, b, y = aby
    test_util.check_grads(lambda y: _beta.betaincinv(a, b, y), (y,), 2)


@pytest.mark.xfail(reason='missing derivs in jax for betainc')
def test_grad_ab(aby):
    """Check the derivatives of `betaincinv` w.r.t. all the arguments."""
    test_util.check_grads(_beta.betaincinv, aby, 1)


def test_dtype(aby):
    """Check the dtype of `betaincinv` with 32 and 64 bit float and int inputs."""
    assert _beta.betaincinv(*aby).dtype == np.float64
    assert _beta.betaincinv(*map(np.float32, aby)).dtype == np.float32
    assert (
        _beta.betaincinv(*(np.ceil(x).astype(np.int64) for x in aby)).dtype
        == np.float64
    )
    assert (
        _beta.betaincinv(*(np.ceil(x).astype(np.int32) for x in aby)).dtype
        == np.float32
    )


def test_ppf():
    """Check `beta.ppf` against scipy."""
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_allclose(
        stats.beta.ppf(q, a, b), _beta.beta.ppf(q, a, b), rtol=1e-6
    )
