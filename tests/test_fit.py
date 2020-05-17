# lsqfitgp/tests/test_fit.py
#
# Copyright (c) 2020, Giacomo Petrillo
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
from scipy import stats
import pytest

sys.path.insert(0, '.')
import lsqfitgp as lgp

def flat(g):
    """convert dictionary or array to 1D array"""
    if hasattr(g, 'buf'):
        return g.buf
    elif hasattr(g, 'keys'):
        return gvar.BufferDict(g).buf
    else:
        return np.reshape(g, -1)

def quad(A, v):
    """compute v.T @ A^-1 @ v"""
    w, u = np.linalg.eigh(A)
    utv = u.T @ v
    eps = len(A) * 1e-12 * np.max(w)
    return (utv.T / np.maximum(w, eps)) @ utv

def chisq_test(g, alpha=2e-5):
    """chisquare test on g being 0"""
    g = flat(g)
    mean = gvar.mean(g)
    cov = gvar.evalcov(g)
    q = quad(cov, mean)
    n = len(mean)
    assert stats.chi2(n).sf(q) > alpha / 2
    assert stats.chi2(n).cdf(q) > alpha / 2

def check_fit(hyperprior, gpfactory, dataerr=None):
    """do a fit with empbayes_fit and check the fitted hyperparameters
    are compatible with the ones used to generate the data"""
    
    # generate hyperparameters
    truehp = next(gvar.raniter(hyperprior))
    
    # generate data
    gp = gpfactory(truehp)
    data = next(gvar.raniter(gp.prior()))
    if dataerr:
        mean = dataerr * np.random.randn(len(data.buf))
        sdev = np.full_like(mean, dataerr)
        data += gvar.BufferDict(data, buf=gvar.gvar(mean, sdev))
        
    # run fit
    fit = lgp.empbayes_fit(hyperprior, gpfactory, data, raises=False)
    
    # check fit result against hyperparameters
    chisq_test(fit.p - truehp)

@pytest.mark.xfail
def test_period():
    hp = {
        'log(scale)': gvar.log(gvar.gvar(1, 0.1))
    }
    x = np.linspace(0, 20, 10)
    def gpfactory(hp):
        gp = lgp.GP(lgp.Periodic(scale=hp['scale']))
        gp.addx(x, 'x')
        return gp
    for _ in range(10):
        check_fit(hp, gpfactory)

def test_scale():
    hp = {
        'log(scale)': gvar.log(gvar.gvar(3, 0.2))
    }
    x = np.linspace(0, 10, 10)
    def gpfactory(hp):
        gp = lgp.GP(lgp.ExpQuad(scale=hp['scale']))
        gp.addx(x, 'x')
        return gp
    for _ in range(10):
        check_fit(hp, gpfactory)

def test_sdev():
    hp = {
        'log(sdev)': gvar.log(gvar.gvar(1, 1))
    }
    x = np.linspace(0, 5, 10)
    def gpfactory(hp):
        gp = lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2)
        gp.addx(x, 'x')
        return gp
    for _ in range(10):
        check_fit(hp, gpfactory)
