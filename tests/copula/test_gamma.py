# lsqfitgp/tests/copula/test_gamma.py
#
# Copyright (c) 2023, 2024, 2026, Giacomo Petrillo
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

"""Test the `copula._gamma` module."""

import jax
import numpy as np
import pytest
from jax import test_util
from scipy import stats

from lsqfitgp.copula import _gamma
from tests import util


@pytest.mark.parametrize(
    'degree',
    [
        pytest.param(1, id='grad'),
        pytest.param(
            2,
            id='hess',
            marks=pytest.mark.xfail(reason='missing second derivs for gammainc in jax'),
        ),
    ],
)
@pytest.mark.parametrize('func', ['gammaincinv', 'gammainccinv'])
def test_deriv(degree, func):
    """Check the derivatives of the inverse incomplete gamma functions."""
    test_util.check_grads(getattr(_gamma, func), (2.5, 0.3), degree)


@pytest.mark.parametrize('func', ['gammaincinv', 'gammainccinv'])
def test_deriv_int_alpha(func):
    """Check that the derivative w.r.t. `y` works with integer `a`."""
    jax.grad(getattr(_gamma, func), 1)(1, 0.5)


@pytest.fixture(params=['gamma', 'invgamma'])
def distr(request):
    """Return the name of a distribution."""
    return request.param


def test_ppf(distr):
    """Check `ppf` against scipy."""
    q = 0.43
    a = 3.6
    np.testing.assert_array_max_ulp(
        getattr(stats, distr).ppf(q, a), getattr(_gamma, distr).ppf(q, a)
    )


def test_isf(distr):
    """Check `isf` against scipy."""
    q = 0.43
    a = 3.6
    util.assert_allclose(
        getattr(_gamma, distr).isf(q, a),
        getattr(stats, distr).isf(q, a),
        atol=0,
        rtol=1e-7,
    )


def test_logpdf():
    """Check `invgamma.logpdf` against scipy."""
    q = 0.43
    a = 3.6
    util.assert_allclose(
        _gamma.invgamma.logpdf(q, a), stats.invgamma.logpdf(q, a), rtol=1e-5
    )


def test_cdf():
    """Check `invgamma.cdf` against scipy."""
    q = 0.43
    a = 3.6
    util.assert_allclose(_gamma.invgamma.cdf(q, a), stats.invgamma.cdf(q, a), rtol=1e-6)


def test_log_asymp():
    """Check `_loggammaisf_normcdf_large_neg_x` against the log of the non-log one."""
    args = -40, 1
    y = _gamma._gammaisf_normcdf_large_neg_x(*args)
    logy = _gamma._loggammaisf_normcdf_large_neg_x(*args)
    util.assert_allclose(y, np.exp(logy), rtol=1e-15)
