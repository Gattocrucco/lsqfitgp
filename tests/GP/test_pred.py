# lsqfitgp/tests/GP/test_pred.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

import itertools
import time

import numpy as np
import gvar
from pytest import mark
import pytest

import lsqfitgp as lgp
from .. import util

# TODO
# - add gpkw for setting checksym, checkpos, solvers
# - this is quite slow, would it be faster with jit?

def pred(seed, err, kw):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=20)
    xpred = rng.uniform(-10, 10, size=100)

    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(x, 'data')
    gp.addx(xpred, 'pred')
    
    y = np.tanh(x)
    if err:
        datagp = lgp.GP(0.1 ** 2 * lgp.Cauchy(scale=0.3))
        datagp.addx(x, 'data')
        y = y + datagp.prior('data')
    
    result = gp.pred({'data': y}, 'pred', **kw)
    if isinstance(result, tuple) and len(result) == 2:
        mean, cov = result
    elif isinstance(result, np.ndarray):
        mean = gvar.mean(result)
        cov = gvar.evalcov(result)
    
    return mean, cov

@mark.parametrize('kw1,kw2', itertools.combinations([
    dict(fromdata=fromdata, raw=raw, keepcorr=keepcorr)
    for fromdata, raw, keepcorr in itertools.product([False, True], repeat=3)
    if not (raw and keepcorr)
], 2))
@mark.parametrize('err', [False, True])
def test_pred(err, kw1, kw2, rng):
    if err and kw1['fromdata'] != kw2['fromdata']:
        pytest.skip()
    for _ in range(10):
        high = np.iinfo(np.uint64).max
        seed = rng.integers(high, dtype=np.uint64, endpoint=True)
        m1, cov1 = pred(seed, err, kw1)
        m2, cov2 = pred(seed, err, kw2)
        util.assert_allclose(m1, m2, rtol=1e-5 if err else 1e-6)
        util.assert_close_matrices(cov1, cov2, rtol=1e-5 if err else 1e-1)
        # TODO 1e-1 looks too inaccurate

def test_double_pred(rng):
    n = 50
    gp = lgp.GP(lgp.ExpQuad())
    ax, bx = rng.standard_normal((2, n))
    gp.addx(ax, 'a')
    gp.addx(bx, 'b')
    m = rng.standard_normal((n, n))
    ay = gvar.gvar(rng.standard_normal(n), m.T @ m)
    m1, cov1 = gp.predfromdata({'a': ay}, 'b', raw=True)
    m2, cov2 = gp.predfromfit(gp.predfromdata({'a': ay}, ['a']), 'b', raw=True)
    util.assert_allclose(m2, m1, rtol=1e-7)
    util.assert_close_matrices(cov2, cov1, rtol=1e-5)
