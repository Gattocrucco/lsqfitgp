import sys
import itertools

import numpy as np
import gvar

sys.path = ['.'] + sys.path
import lsqfitgp as lgp

def pred_noerr(kw, x, xpred):
    kernel = lgp.ExpQuad()
    gp = lgp.GP(kernel)
    gp.addx(x, 'data')
    gp.addx(xpred, 'pred')
    y = np.tanh(x)
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

noerr_kwargs = [
    dict(fromdata=fromdata, raw=raw, keepcorr=keepcorr)
    for fromdata in [False, True]
    for raw in [False, True]
    for keepcorr in [False, True]
    if not (keepcorr and raw)
]

for kw1, kw2 in itertools.combinations(noerr_kwargs, 2):
    name = 'test_pred_noerr_'
    name += '_'.join(k + '_' + str(v) for (k, v) in kw1.items())
    name += '___'
    name += '_'.join(k + '_' + str(v) for (k, v) in kw2.items())
    fundef = """
def {}():
    x = np.random.uniform(-5, 5, size=20)
    xpred = np.random.uniform(-10, 10, size=100)
    m1, cov1 = pred_noerr({}, x, xpred)
    m2, cov2 = pred_noerr({}, x, xpred)
    assert_close(m1, m2)
    assert_close_cov(cov1, cov2, 1e-2, 2e-6)
""".format(name, kw1, kw2)
    exec(fundef)
