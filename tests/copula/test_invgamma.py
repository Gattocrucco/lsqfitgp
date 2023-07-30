# lsqfitgp/tests/copula/test_invgamma.py
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

""" Test the copula.invgamma module """

from jax import test_util
from scipy import stats
import pytest
import numpy as np

from lsqfitgp.copula import _invgamma

def test_grad():
    test_util.check_grads(_invgamma.gammainccinv, (2.5, 0.3), 1)
    test_util.check_grads(_invgamma.gammaincinv, (2.5, 0.3), 1)

@pytest.mark.xfail
def test_hess():
    test_util.check_grads(_invgamma.gammainccinv, (2.5, 0.3), 2)
    test_util.check_grads(_invgamma.gammaincinv, (2.5, 0.3), 2)

def test_ppf():
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_array_max_ulp(
        stats.invgamma.ppf(q, a, scale=b),
        _invgamma.invgamma.ppf(q, a, scale=b),
    )

def test_isf():
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_allclose(
        _invgamma.invgamma.isf(q, a, scale=b),
        stats.invgamma.isf(q, a, scale=b),
        atol=0, rtol=1e-7,
    )

def test_logpdf():
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_allclose(
        _invgamma.invgamma.logpdf(q, a, scale=b),
        stats.invgamma.logpdf(q, a, scale=b), 
        rtol=1e-5,
    )

def test_cdf():
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_allclose(
        _invgamma.invgamma.cdf(q, a, scale=b),
        stats.invgamma.cdf(q, a, scale=b), 
        rtol=1e-6,
    )
