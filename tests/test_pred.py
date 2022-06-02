# lsqfitgp/tests/test_pred.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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
import itertools
import time

import numpy as np
import gvar

sys.path = ['.'] + sys.path
import lsqfitgp as lgp

# TODO add gpkw for setting checksym, checkpos, solvers

def pred(kw, seed, err):
    np.random.seed(seed)
    x = np.random.uniform(-5, 5, size=20)
    xpred = np.random.uniform(-10, 10, size=100)

    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(x, 'data')
    gp.addx(xpred, 'pred')
    
    y = np.tanh(x)
    if err:
        datagp = lgp.GP(0.1 ** 2 * lgp.RatQuad(scale=0.3))
        datagp.addx(x, 'data')
        y = y + datagp.prior('data')
    
    result = gp.pred({'data': y}, 'pred', **kw)
    if isinstance(result, tuple) and len(result) == 2:
        mean, cov = result
    elif isinstance(result, np.ndarray):
        mean = gvar.mean(result)
        cov = gvar.evalcov(result)
    
    return mean, cov

def assert_close(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-05, atol=1e-08)

def assert_close_cov(a, b, stol, mtol):
    assert np.sqrt(np.sum((a - b) ** 2) / a.size) < stol
    assert np.median(np.abs(a - b)) < mtol

kwargs = [
    dict(fromdata=fromdata, raw=raw, keepcorr=keepcorr)
    for fromdata in [False, True]
    for raw in [False, True]
    for keepcorr in [False, True]
    if not (keepcorr and raw)
]

def postfix(kw1, kw2):
    postfix = '_'
    postfix += '_'.join(k + '_' + str(v) for (k, v) in kw1.items())
    postfix += '___'
    postfix += '_'.join(k + '_' + str(v) for (k, v) in kw2.items())
    return postfix

def makeseed():
    return int(1e6 * time.time()) % (1 << 32)

for kw1, kw2 in itertools.combinations(kwargs, 2):
    
    fundef = """
def {}():
    for _ in range(10):
        seed = makeseed()
        m1, cov1 = pred({}, seed, False)
        m2, cov2 = pred({}, seed, False)
        assert_close(m1, m2)
        assert_close_cov(cov1, cov2, 6e-3, 2e-5)
""".format('test_pred_noerr' + postfix(kw1, kw2), kw1, kw2)
    
    exec(fundef)
    
    fundef = """
def {}():
    for _ in range(10):
        seed = makeseed()
        m1, cov1 = pred({}, seed, True)
        m2, cov2 = pred({}, seed, True)
        assert_close(m1, m2)
        assert_close_cov(cov1, cov2, 6e-3, 2e-5)
""".format('test_pred_err' + postfix(kw1, kw2), kw1, kw2)
    
    if kw1['fromdata'] == kw2['fromdata']:
        exec(fundef)

def test_double_pred():
    n = 50
    gp = lgp.GP(lgp.ExpQuad())
    ax, bx = np.random.randn(2, n)
    gp.addx(ax, 'a')
    gp.addx(bx, 'b')
    m = np.random.randn(n, n)
    ay = gvar.gvar(np.random.randn(n), m.T @ m)
    m1, cov1 = gp.predfromdata({'a': ay}, 'b', raw=True)
    m2, cov2 = gp.predfromfit(gp.predfromdata({'a': ay}, ['a']), 'b', raw=True)
    assert_close(m2, m1)
    assert_close_cov(cov2, cov1, 1e-3, 1e-6)
