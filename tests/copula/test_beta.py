# lsqfitgp/tests/copula/test_beta.py
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

"""Test the copula.beta module"""

from jax import test_util
from scipy import stats
import pytest
import numpy as np
from pytest import mark

from lsqfitgp.copula import _beta

@pytest.fixture
def aby():
    return 2.5, 1.3, 0.3

def test_grad(aby):
    a, b, y = aby
    test_util.check_grads(lambda y: _beta.betaincinv(a, b, y), (y,), 2)

@mark.xfail(reason='missing derivs in jax for betainc')
def test_grad_ab(aby):
    test_util.check_grads(_beta.betaincinv, aby, 1)

def test_dtype(aby):
    assert _beta.betaincinv(*aby).dtype == np.float64
    assert _beta.betaincinv(*map(np.float32, aby)).dtype == np.float32
    assert _beta.betaincinv(*map(lambda x: np.ceil(x).astype(np.int64), aby)).dtype == np.float64
    assert _beta.betaincinv(*map(lambda x: np.ceil(x).astype(np.int32), aby)).dtype == np.float32

def test_ppf():
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_allclose(stats.beta.ppf(q, a, b), _beta.beta.ppf(q, a, b), rtol=1e-6)
