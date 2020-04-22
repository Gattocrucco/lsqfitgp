from __future__ import division

import sys
import itertools
import time

import numpy as np
import gvar

sys.path = ['.'] + sys.path
import lsqfitgp as lgp

def pred(kw, seed):
    np.random.seed(int(seed))
    x = np.random.uniform(-5, 5, size=20)
    xpred = np.random.uniform(-10, 10, size=100)

    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(x, 'data')
    gp.addx(xpred, 'pred')
    
    datagp = lgp.GP(0.1 ** 2 * lgp.RatQuad(scale=0.3))
    datagp.addx(x, 'data')
    y = np.tanh(x) + datagp.prior('data')
    
    result = gp.pred({'data': y}, 'pred', **kw)
    if isinstance(result, tuple) and len(result) == 2:
        mean, cov = result
    elif isinstance(result, np.ndarray):
        mean = gvar.mean(result)
        cov = gvar.evalcov(result)
    
    return mean, cov

def assert_close(x, y):
    assert np.allclose(x, y)

def assert_close_cov(a, b, stol, mtol):
    assert np.sqrt(np.sum((a - b) ** 2) / a.size) < stol
    assert np.median(np.abs(a - b)) < mtol

for fromdata in [True, False]:
    kwargs = [
        dict(fromdata=fromdata, raw=raw, keepcorr=keepcorr)
        for raw in [False, True]
        for keepcorr in [False, True]
        if not (keepcorr and raw)
    ]

    for kw1, kw2 in itertools.combinations(kwargs, 2):
        name = 'test_pred_'
        name += '_'.join(k + '_' + str(v) for (k, v) in kw1.items())
        name += '___'
        name += '_'.join(k + '_' + str(v) for (k, v) in kw2.items())
        fundef = """
def {}():
    seed = time.time()
    m1, cov1 = pred({}, seed)
    m2, cov2 = pred({}, seed)
    assert_close(m1, m2)
    assert_close_cov(cov1, cov2, 3e-2, 3e-5)
""".format(name, kw1, kw2)
        exec(fundef)
