# lsqfitgp/tests/test_GP.py
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
import copy

import pytest
import numpy as np
from jax import numpy as jnp
import gvar

sys.path.insert(0, '.')
import lsqfitgp as lgp

import util

def test_prior_raw_shape():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(np.arange(20).reshape(2, 10), 'x')
    cov = gp.prior(raw=True)
    assert cov['x', 'x'].shape == (2, 10, 2, 10)
    
    cov = gp.prior('x', raw=True)
    assert cov.shape == (2, 10, 2, 10)

def test_no_checksym():
    covs = []
    for checksym in [False, True]:
        gp = lgp.GP(lgp.ExpQuad(), checksym=checksym)
        gp.addx({'x': np.arange(20)})
        covs.append(gp.prior('x', raw=True))
    util.assert_equal(*covs)

def test_transf_scalar():
    gp = lgp.GP(lgp.ExpQuad() + lgp.RatQuad())
    x = np.arange(20)
    gp.addx(x, 'x')
    cov1 = gp.prior('x', raw=True)
    
    gp = lgp.GP(lgp.where(lambda f: f, lgp.ExpQuad(dim='f1'), lgp.RatQuad(dim='f1'), dim='f0'))
    y = np.empty((2, len(x)), '?,f8')
    y['f0'] = np.reshape([0, 1], (2, 1))
    y['f1'] = x
    gp.addx(y[0], 'y0')
    gp.addx(y[1], 'y1')
    gp.addtransf({'y0': 1, 'y1': 1}, 'x')
    cov2 = gp.prior('x', raw=True)
    
    util.assert_equal(cov1, cov2)

def test_transf_vector():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx([0, 1], 'x')
    gp.addtransf({'x': [-1, 1]}, 'y')
    prior = gp.prior()
    y1 = prior['y']
    y2 = prior['x'][1] - prior['x'][0]
    np.testing.assert_allclose(gvar.mean(y1), gvar.mean(y2))
    np.testing.assert_allclose(gvar.sdev(y1), gvar.sdev(y2))
    z = y1 - y2
    np.testing.assert_allclose(gvar.mean(z), 0)
    np.testing.assert_allclose(gvar.sdev(z), 0)

def test_cov():
    A = np.random.randn(20, 20)
    M1 = A.T @ A
    gp = lgp.GP()
    gp.addcov(M1, 'M')
    M2 = gp.prior('M', raw=True)
    util.assert_equal(M1, M2)

def test_proctransf():
    x = np.arange(20)
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    gp.addproc(lgp.RatQuad(), 'b')
    gp.addx(x, 'ax', proc='a')
    gp.addx(x, 'bx', proc='b')
    gp.addtransf({'ax': 2, 'bx': 3}, 'abx1')
    gp.addproctransf({'a': 2, 'b': 3}, 'ab')
    gp.addx(x, 'abx2', proc='ab')
    prior = gp.prior(['abx1', 'abx2'], raw=True)
    util.assert_equal(prior['abx1', 'abx1'], prior['abx2', 'abx2'])
    util.assert_equal(prior['abx1', 'abx2'], prior['abx2', 'abx1'].T)
    util.assert_equal(prior['abx1', 'abx2'], prior['abx1', 'abx1'])

def test_kernelop():
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    f = lambda x: x
    gp.addproc(lgp.ExpQuad() * lgp.Rescaling(stdfun=f), 'b1')
    gp.addkernelop('rescale', f, 'b2', 'a')
    x = np.arange(20)
    gp.addx(x, 'x1', proc='b1')
    gp.addx(x, 'x2', proc='b2')
    prior = gp.prior(['x1', 'x2'], raw=True)
    util.assert_equal(prior['x1', 'x1'], prior['x2', 'x2'])
    util.assert_equal(prior['x2', 'x1'], prior['x1', 'x2'].T)
    util.assert_equal(prior['x1', 'x2'], np.zeros(2 * x.shape))

def test_not_kernel():
    with pytest.raises(TypeError):
        gp = lgp.GP(0)

def test_two_procs():
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    with pytest.raises(KeyError):
        gp.addproc(lgp.ExpQuad(), 'a')

def test_default_kernel():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addproc(key='a')
    x = np.arange(20)
    gp.addx(x, 'x1')
    gp.addx(x, 'x2', proc='a')
    prior = gp.prior(raw=True)
    util.assert_equal(prior['x1', 'x1'], prior['x2', 'x2'])
    util.assert_equal(prior['x1', 'x2'], np.zeros(2 * x.shape))

def test_no_proc():
    gp = lgp.GP()
    with pytest.raises(KeyError):
        gp.addproctransf({'a': 1}, 'b')

def test_invalid_factor():
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    with pytest.raises(TypeError):
        gp.addproctransf({'a': None}, 'b')

def test_existing_proc():
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    with pytest.raises(KeyError):
        gp.addproctransf({'a': 1}, 'a')

def test_empty_proc():
    gp = lgp.GP()
    gp.addproctransf({}, 'a')
    x = np.arange(20)
    gp.addx(x, 'x', proc='a')
    cov = gp.prior('x', raw=True)
    util.assert_equal(cov, np.zeros(2 * x.shape))

def test_no_op():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addkernelop('cippa', None, 'a')

def test_already_defined():
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    with pytest.raises(KeyError):
        gp.addkernelop('diff', 1, 'a', 'a')

def test_proc_not_found():
    gp = lgp.GP()
    with pytest.raises(KeyError):
        gp.addkernelop('diff', 1, 'a', 'b')

def test_addprocderiv():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addprocderiv(1, 'a')
    x = np.arange(20)
    gp.addx(x, 0, proc='a')
    gp.addx(x, 1, deriv=1)
    prior = gp.prior(raw=True)
    util.assert_equal(prior[0, 0], prior[1, 1])
    util.assert_equal(prior[0, 0], prior[0, 1])

def test_addprocxtransf():
    gp = lgp.GP(lgp.ExpQuad())
    f = lambda x: x ** 2
    gp.addprocxtransf(f, 'a')
    x = np.linspace(0, 4, 20)
    gp.addx(x, 0, proc='a')
    gp.addx(f(x), 1)
    prior = gp.prior(raw=True)
    util.assert_equal(prior[0, 0], prior[1, 1])
    util.assert_equal(prior[0, 0], prior[0, 1])

def test_addprocrescale():
    gp = lgp.GP(lgp.ExpQuad())
    s = lambda x: x ** 2
    gp.addprocrescale(s, 'a')
    x = np.linspace(0, 2, 20)
    gp.addx(x, 'base')
    gp.addtransf({'base': s(x) * np.eye(len(x))}, 0)
    gp.addx(x, 1, proc='a')
    prior = gp.prior([0, 1], raw=True)
    np.testing.assert_allclose(prior[0, 0], prior[1, 1], rtol=1e-15)
    np.testing.assert_allclose(prior[0, 0], prior[0, 1], rtol=1e-15)

def test_missing_proc():
    gp = lgp.GP()
    with pytest.raises(KeyError):
        gp.addx(0, 0, proc='cippa')

def test_no_key():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addx(0)
    with pytest.raises(ValueError):
        gp.addcov(1)

def test_redundant_key():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addx({0: 0}, 0)
    with pytest.raises(ValueError):
        gp.addcov({0: 1}, 0)

def test_none_key():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addx({None: 0})
    
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    with pytest.raises(ValueError):
        gp.addtransf({0: 1}, None)

    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addcov({None: 1})

def test_nonsense_x():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(TypeError):
        gp.addx(None, 0)
    with pytest.raises(TypeError):
        gp.addcov(None, 0)

def test_key_already_used():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    with pytest.raises(KeyError):
        gp.addx(0, 0)

    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    with pytest.raises(KeyError):
        gp.addtransf({0: 1}, 0)

    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    with pytest.raises(KeyError):
        gp.addcov(1, 0)

def test_not_array():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(TypeError):
        gp.addx({0: {}})

def test_not_empty():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addx([], 0)

def test_incompatible_dtypes():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    with pytest.raises(TypeError):
        gp.addx(np.array((0, 0), 'f8,f8'), 1)
    
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(np.array((0, 0), 'f8,f8'), 0)
    with pytest.raises(TypeError):
        gp.addx(np.array((0, 0), 'i8,i8'), 1)

def test_explicit_deriv():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addx(0, 0, deriv='x')

def test_missing_field():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addx(np.array((0, 0), 'f8,f8'), 0, deriv='x')

def test_missing_key():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(KeyError):
        gp.addtransf({0: 1}, 1)

def test_nonsense_tensors():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    with pytest.raises(TypeError):
        gp.addtransf({0: 'a'}, 1)
    with pytest.raises(ValueError):
        gp.addtransf({0: np.inf}, 1)
    gp.addx([0, 1], 1)
    with pytest.raises(ValueError):
        gp.addtransf({1: [1, 2, 3]}, 2)

def test_fail_broadcast():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx([0, 1], 0)
    gp.addx([0, 1, 2], 1)
    with pytest.raises(ValueError):
        gp.addtransf({0: 1, 1: 1}, 2)

def test_addcov_wrong_blocks():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addcov(np.zeros((1, 1, 1)), 0)
    with pytest.raises(ValueError):
        gp.addcov(np.zeros((1, 2, 2, 1)), 0)
    with pytest.raises(ValueError):
        gp.addcov(np.random.randn(10, 10), 0)
    with pytest.raises(KeyError):
        gp.addcov({
            (0, 0): 1,
            (0, 1): 0,
        })
    with pytest.raises(ValueError):
        gp.addcov({
            (0, 0): np.ones((2, 2)),
            (1, 1): np.ones((3, 3)),
            (0, 1): np.ones((3, 2)),
        })
    with pytest.raises(ValueError):
        gp.addcov({
            (0, 0): np.ones((2, 2)),
            (1, 1): np.ones((3, 3)),
            (0, 1): np.ones((2, 3)),
            (1, 0): np.zeros((3, 2)),
        })

def test_addcov_no_checksym():
    gp = lgp.GP(lgp.ExpQuad(), checksym=False)
    gp.addcov({
        (0, 0): np.ones((2, 2)),
        (1, 1): np.ones((3, 3)),
        (0, 1): np.ones((2, 3)),
        (1, 0): np.zeros((3, 2)),
    })

def test_addcov_missing_block():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addcov({
        (0, 0): np.ones((2, 2)),
        (1, 1): np.ones((3, 3)),
        (0, 1): np.ones((2, 3)),
    })
    prior = gp.prior(raw=True)
    util.assert_equal(prior[0, 1], prior[1, 0].T)

def test_new_proc():
    gp = lgp.GP()
    gp._procs[0] = None
    gp.addx(0, 0, proc=0)
    with pytest.raises(TypeError):
        gp.prior(0, raw=True)

def test_partial_derivative():
    gp = lgp.GP(lgp.ExpQuad())
    x = np.arange(20)
    y = np.zeros(len(x), 'f8,f8')
    y['f0'] = x
    gp.addx(y, 0, deriv='f0')
    cov1 = gp.prior(0, raw=True)
    
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(x, 0, deriv=1)
    cov2 = gp.prior(0, raw=True)
    
    util.assert_equal(cov1, cov2)

def test_addproctransf_function():
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    x = np.arange(20)
    gp.addx(x, 'base', proc='a')
    s = lambda x: x ** 2
    gp.addtransf({'base': s(x) * np.eye(len(x))}, 0)
    gp.addproctransf({'a': s}, 'b')
    gp.addx(x, 1, proc='b')
    prior = gp.prior([0, 1], raw=True)
    util.assert_equal(prior[0, 0], prior[1, 1])
    util.assert_equal(prior[0, 0], prior[0, 1])

def test_zero_covblock():
    gp = lgp.GP()
    a = np.random.randn(10, 10)
    m = a.T @ a
    gp.addcov(m, 0)
    gp.addcov(m, 1)
    prior = gp.prior(raw=True)
    util.assert_equal(prior[0, 1], np.zeros_like(m))

def test_addcov_checks():
    gp = lgp.GP()
    a = np.random.randn(10, 10)
    b = np.copy(a)
    b[0, 0] = np.inf
    m = b.T @ b
    with pytest.raises(ValueError):
        gp.addcov(a, 0)
    with pytest.raises(ValueError):
        gp.addcov(m, 0)
    
    gp = lgp.GP(checksym=False)
    gp.addcov(a, 0)
    
    gp = lgp.GP(checkfinite=False)
    gp.addcov(m, 0)

def test_makecovblock_checks():
    a = np.random.randn(10, 10)
    b = np.copy(a)
    b[0, 0] = np.inf
    m = b.T @ b
    
    gp = lgp.GP(checksym=False)
    gp.addcov(a, 0)
    gp._checksym = True
    with pytest.raises(RuntimeError):
        gp.prior(raw=True)

    gp = lgp.GP(checkfinite=False)
    gp.addcov(m, 0)
    gp._checkfinite = True
    with pytest.raises(RuntimeError):
        gp.prior(raw=True)
    
    gp = lgp.GP(checksym=False, checkpos=False)
    gp.addcov(a, 0)
    gp.prior(raw=True)
    
    gp = lgp.GP(checkfinite=False, checkpos=False)
    gp.addcov(m, 0)
    gp.prior(raw=True)

def test_covblock_checks():
    a, b, c, d = np.random.randn(4, 10, 10)
    m = a.T @ a
    n = b.T @ b
    gp = lgp.GP(checksym=False, checkpos=False)
    gp.addcov({
        (0, 0): m,
        (1, 1): n,
        (0, 1): c,
        (1, 0): d,
    })
    gp2 = copy.deepcopy(gp)
    gp._checksym = True
    with pytest.raises(RuntimeError):
        gp.prior(raw=True)
    gp2.prior(raw=True)

def test_solver_cache():
    gp = lgp.GP(lgp.ExpQuad())
    x = np.linspace(0, 1, 10)
    y = np.linspace(1, 2, 10)
    z = np.zeros_like(x)
    gp.addx(x, 0)
    gp.addx(y, 1)
    m1, c1 = gp.predfromdata({0: z}, 1, raw=True)
    m2, c2 = gp.predfromdata({0: z}, 1, raw=True)
    util.assert_equal(m1, m2)
    util.assert_equal(c1, c2)

def test_checkpos():
    a = np.random.randn(20, 20)
    m = a.T @ a
    w, v = np.linalg.eigh(m)
    w[np.arange(len(w)) % 2] *= -1
    m = (v * w) @ v.T
    
    gp = lgp.GP()
    gp.addcov(m, 0)
    with pytest.raises(np.linalg.LinAlgError):
        gp.prior()
    
    gp._checkpositive = False
    gp.prior()

def test_priorpoints_cache():
    gp = lgp.GP(lgp.ExpQuad())
    x = np.arange(20)
    gp.addx(x, 0)
    gp.addx(x, 1)
    prior = gp.prior()
    cov = gvar.evalcov(prior)
    util.assert_equal(cov[0, 0], cov[1, 1])
    util.assert_equal(cov[0, 0], cov[0, 1])

def test_priortransf():
    gp = lgp.GP(lgp.ExpQuad())
    x, y = np.arange(40).reshape(2, -1)
    gp.addx(x, 0)
    gp.addx(y, 1)
    gp.addtransf({0: x, 1: y}, 2)
    cov1 = gp.prior(2, raw=True)
    u = gp.prior(2)
    cov2 = gvar.evalcov(u)
    np.testing.assert_allclose(cov1, cov2, rtol=1e-15)

def test_new_element():
    gp = lgp.GP()
    gp._elements[0] = None
    with pytest.raises(TypeError):
        gp.prior()

def test_given_checks():
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = np.random.randn(3, 20)
    gp.addx(x, 0)
    gp.addx(y, 1)
    with pytest.raises(TypeError):
        gp.predfromdata(0, 1)
    with pytest.raises(TypeError):
        gp.predfromdata({0: z}, 1, givencov=0)
    with pytest.raises(KeyError):
        gp.predfromdata({2: z}, 1)
    with pytest.raises(ValueError):
        gp.predfromdata({0: z[:-1]}, 1)
    with pytest.raises(TypeError):
        gp.predfromdata({0: np.empty_like(z, str)}, 1)

def test_zero_givencov():
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = np.random.randn(3, 20)
    gp.addx(x, 0)
    gp.addx(y, 1)
    cov = np.zeros(2 * x.shape)
    m1, c1 = gp.predfromdata({0: z}, 1, {(0, 0): cov}, raw=True)
    m2, c2 = gp.predfromdata({0: z}, 1, raw=True)
    util.assert_equal(m1, m2)
    util.assert_equal(c1, c2)

def test_pred_checks():
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = np.random.randn(3, 20)
    gp.addx(x, 0)
    gp.addx(y, 1)
    with pytest.raises(ValueError):
        gp.pred({0: z}, 1)
    with pytest.raises(ValueError):
        gp.predfromdata({0: z}, 1, raw=True, keepcorr=True)
    with pytest.raises(ValueError):
        gp.predfromdata({0: np.full_like(z, np.nan)}, 1)
    with pytest.raises(ValueError):
        gp.predfromdata({0: z}, 1, {(0, 0): np.full(2 * x.shape, np.nan)})
    a = np.random.randn(20, 20)
    with pytest.raises(ValueError):
        gp.predfromdata({0: z}, 1, {(0, 0): a})
    gp._checkfinite = False
    gp.predfromdata({0: np.full_like(z, np.nan)}, 1)

def test_pred_all():
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = np.random.randn(3, 20)
    gp.addx(x, 0)
    gp.addx(y, 1)
    m1, c1 = gp.predfromdata({0: z}, raw=True)
    m2, c2 = gp.predfromdata({0: z}, [0, 1], raw=True)
    util.assert_equal(m1, m2)
    util.assert_equal(c1, c2)

def test_marginal_likelihood_separate():
    gp = lgp.GP(lgp.ExpQuad())
    x, y = np.random.randn(2, 20)
    gp.addx(x, 0)
    ml1 = gp.marginal_likelihood({0: y})
    l, r = gp.marginal_likelihood({0: y}, separate=True)
    ml2 = -1/2 * (l + r @ r)
    np.testing.assert_allclose(ml2, ml1, rtol=1e-15)

def test_marginal_likelihood_checks():
    gp = lgp.GP(lgp.ExpQuad())
    x, y = np.random.randn(2, 20)
    gp.addx(x, 0)
    z = np.full_like(x, np.nan)
    with pytest.raises(ValueError):
        gp.marginal_likelihood({0: z})
    m = np.full(2 * x.shape, np.nan)
    with pytest.raises(ValueError):
        gp.marginal_likelihood({0: y}, {(0, 0): m})
    a = np.random.randn(*2 * x.shape)
    with pytest.raises(ValueError):
        gp.marginal_likelihood({0: y}, {(0, 0): a})
    c = a.T @ a
    with pytest.warns(UserWarning):
        gp.marginal_likelihood({0: gvar.gvar(y, c)}, {(0, 0): c})

def test_marginal_likelihood_gvar():
    gp = lgp.GP(lgp.ExpQuad())
    x, y = np.random.randn(2, 20)
    gp.addx(x, 0)
    a = np.random.randn(20, 20)
    m = a.T @ a
    ml1 = gp.marginal_likelihood({0: gvar.gvar(y, m)})
    ml2 = gp.marginal_likelihood({0: y}, {(0, 0): m})
    np.testing.assert_allclose(ml2, ml1, rtol=1e-15)

def test_singleton():
    dp = lgp._GP.DefaultProcess
    assert repr(dp) == 'DefaultProcess'
    with pytest.raises(NotImplementedError):
        dp()
