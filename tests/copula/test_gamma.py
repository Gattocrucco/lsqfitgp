# lsqfitgp/tests/copula/test_gamma.py
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

""" Test the copula._gamma module """

from jax import test_util
from scipy import stats
from pytest import mark
import pytest
import numpy as np

from lsqfitgp.copula import _gamma

@mark.parametrize('degree', [
    pytest.param(1, id='grad'),
    pytest.param(2, id='hess', marks=mark.xfail),
])
@mark.parametrize('func', ['gammaincinv', 'gammainccinv'])
def test_deriv(degree, func):
    test_util.check_grads(getattr(_gamma, func), (2.5, 0.3), degree)

@pytest.fixture(params=['gamma', 'invgamma'])
def distr(request):
    return request.param

def test_ppf(distr):
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_array_max_ulp(
        getattr(stats, distr).ppf(q, a, scale=b),
        getattr(_gamma, distr).ppf(q, a, scale=b),
    )

def test_isf(distr):
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_allclose(
        getattr(_gamma, distr).isf(q, a, scale=b),
        getattr(stats, distr).isf(q, a, scale=b),
        atol=0, rtol=1e-7,
    )

def test_logpdf():
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_allclose(
        _gamma.invgamma.logpdf(q, a, scale=b),
        stats.invgamma.logpdf(q, a, scale=b), 
        rtol=1e-5,
    )

def test_cdf():
    q = 0.43
    a = 3.6
    b = 2.1
    np.testing.assert_allclose(
        _gamma.invgamma.cdf(q, a, scale=b),
        stats.invgamma.cdf(q, a, scale=b), 
        rtol=1e-6,
    )
