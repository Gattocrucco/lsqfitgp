# lsqfitgp/tests/test_GP.py
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

import copy
import sys
import itertools

import pytest
import numpy as np
from jax import numpy as jnp
import gvar
import jax

sys.path = ['.'] + sys.path
import lsqfitgp as lgp
import util

gen = np.random.default_rng(202208251700)

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
    gp = lgp.GP(lgp.ExpQuad() + lgp.Cauchy())
    x = np.arange(20)
    gp.addx(x, 'x')
    cov1 = gp.prior('x', raw=True)
    
    gp = lgp.GP(lgp.where(lambda f: f, lgp.ExpQuad(dim='f1'), lgp.Cauchy(dim='f1'), dim='f0'))
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
    util.assert_same_gvars(y1, y2, atol=1e-12)

def test_cov():
    A = gen.standard_normal((20, 20))
    M1 = A.T @ A
    gp = lgp.GP()
    gp.addcov(M1, 'M')
    M2 = gp.prior('M', raw=True)
    util.assert_equal(M1, M2)

def test_compare_transfs():
    x = np.arange(20)
    def preparegp():
        gp = lgp.GP()
        gp.addproc(lgp.ExpQuad(), 'a')
        gp.addproc(lgp.Cauchy(), 'b')
        gp.addx(x, 'ax', proc='a')
        gp.addx(x, 'bx', proc='b')
        return gp
    def finalizegp(gp):
        gp.addx(x, 2, proc='ab1')
        gp.addx(x, 3, proc='ab2')
    def checkgp(gp):
        keys = [0, 1, 2, 3]
        prior = gp.prior(keys, raw=True)
        for k1 in keys:
            for k2 in keys:
                util.assert_allclose(prior[0, 0], prior[k1, k2], atol=1e-15, rtol=1e-15)
    
    # with functions
    gp = preparegp()
    fa = lambda x: jnp.sin(x) + 0.5 * jnp.cos(x ** 2)
    fb = lambda x: 1 / (1 + jnp.exp(-x))
    gp.addtransf({'ax': np.diag(fa(x)), 'bx': np.diag(fb(x))}, 0)
    gp.addlintransf(lambda a, b: fa(x) * a + fb(x) * b, ['ax', 'bx'], 1)
    gp.addproctransf({'a': fa, 'b': fb}, 'ab1')
    gp.addproclintransf(lambda a, b: lambda x: fa(x) * a(x) + fb(x) * b(x), ['a', 'b'], 'ab2')
    finalizegp(gp)
    checkgp(gp)
    
    # with scalars
    gp = preparegp()
    gp.addtransf({'ax': 2, 'bx': 3}, 0)
    gp.addlintransf(lambda a, b: 2 * a + 3 * b, ['ax', 'bx'], 1)
    gp.addproctransf({'a': 2, 'b': 3}, 'ab1')
    gp.addproclintransf(lambda a, b: lambda x: 2 * a(x) + 3 * b(x), ['a', 'b'], 'ab2')
    finalizegp(gp)
    checkgp(gp)

def test_lintransf_checks():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    gp.addx(0, 1)
    with pytest.raises(KeyError):
        gp.addlintransf(lambda x, y: x + y, [0, 1], 0)
    with pytest.raises(ValueError):
        gp.addlintransf(lambda x, y: x + y, [0, 1], None)
    with pytest.raises(KeyError):
        gp.addlintransf(lambda x, y: x + y, [0, 2], 2)
    with pytest.raises(RuntimeError):
        gp.addlintransf(lambda x, y: 1 + x + y, [0, 1], 2)
    with pytest.raises(RuntimeError):
        gp.addlintransf(lambda x, y: 1, [0, 1], 2)
    gp.addlintransf(lambda x, y: 1 + x + y, [0, 1], 2, checklin=False)
    gp._checklin = False
    gp.addlintransf(lambda x, y: 1 + x + y, [0, 1], 3)
    with pytest.raises(RuntimeError):
        gp.addlintransf(lambda x, y: 1 + x + y, [0, 1], 4, checklin=True)

def test_proclintransf_checks():
    def makegp(**kw):
        gp = lgp.GP(**kw)
        gp.addproc(lgp.ExpQuad(), 0)
        gp.addproc(lgp.ExpQuad(), 1)
        return gp
    gp = makegp()
    with pytest.raises(KeyError):
        gp.addproclintransf(lambda f, g: lambda x: f(x) + g(x), [0, 1], 0)
    with pytest.raises(KeyError):
        gp.addproclintransf(lambda f, g: lambda x: f(x) + g(x), [0, 2], 2)
    with pytest.raises(RuntimeError):
        gp.addproclintransf(lambda f, g: lambda x: 1, [0, 1], 2, checklin=True)
    with pytest.raises(RuntimeError):
        gp.addproclintransf(lambda f, g: lambda x: 1 + f(x) + g(x), [0, 1], 2, checklin=True)
    with pytest.raises(RuntimeError):
        gp.addproclintransf(lambda f, g: lambda x: f(x) + g(x)[None, :], [0, 1], 2, checklin=True)
    gp.addproclintransf(lambda f, g: lambda x: 1 + f(x) + g(x), [0, 1], 2)
    gp.addproclintransf(lambda f, g: lambda x: f(x) + g(x), [0, 1], 3, checklin=True)
    gp = makegp(checklin=True)
    with pytest.raises(RuntimeError):
        gp.addproclintransf(lambda f, g: lambda x: 1, [0, 1], 2, checklin=None)

def test_proclintransf_mockup():
    def makegp(**kw):
        gp = lgp.GP(**kw)
        gp.addproc(lgp.ExpQuad(), 0)
        gp.addproc(lgp.ExpQuad(), 1)
        return gp
    gp = makegp()
    with pytest.raises(RuntimeError):
        gp.addproclintransf(lambda f, g: lambda x: 1 + f(x['dim1']) + g(x['dim2']), [0, 1], 2, checklin=True)
    gp.addproclintransf(lambda f, g: lambda x: f(x['dim1']) + g(x['dim2']), [0, 1], 2, checklin=True)

def test_lintransf_matmul():
    gp = lgp.GP(lgp.ExpQuad())
    x = np.arange(20)
    gp.addx(x, 0)
    m = gen.standard_normal((30, len(x)))
    gp.addlintransf(lambda x: m @ x, [0], 1)
    prior = gp.prior([0, 1], raw=True)
    util.assert_allclose(m @ prior[0, 0] @ m.T, prior[1, 1], rtol=1e-12)
    util.assert_allclose(m @ prior[0, 1], prior[1, 1], rtol=1e-12)
    util.assert_allclose(prior[1, 0] @ m.T, prior[1, 1], rtol=1e-12)

def test_prior_gvar():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(gen.standard_normal(20), 0)
    gp.addx(gen.standard_normal(15), 1)
    gp.addtransf({0: gen.standard_normal((15, 20))}, 2)
    gp.addlintransf(lambda x, y: jnp.real(jnp.fft.rfft(x + y)), [1, 2], 3)
    m = gen.standard_normal((40, 40))
    gp.addcov(m @ m.T, 4)
    gp.addtransf({4: gen.standard_normal((8, 40)), 3: np.pi}, 5)
    covs = gp.prior(raw=True)
    prior = gp.prior()
    gmeans = gvar.mean(prior)
    for g in gmeans.values():
        np.testing.assert_equal(g, np.zeros_like(g))
    gcovs = gvar.evalcov(prior)
    for k, cov in covs.items():
        gcov = gcovs[k]
        util.assert_close_matrices(cov, gcov, atol=1e-15, rtol=1e-9)

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
    gp.addproclintransf(lambda: lambda x: 0, [], 'b')
    x = np.arange(20)
    gp.addx(x, 'ax', proc='a')
    gp.addx(x, 'bx', proc='b')
    cov = gp.prior(raw=True)
    util.assert_equal(cov['ax', 'ax'], np.zeros(2 * x.shape))
    util.assert_equal(cov['bx', 'bx'], np.zeros(2 * x.shape))

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
    gp = lgp.GP()
    f = lambda x: x ** 2
    gp.addproc(lgp.ExpQuad(), 0)
    gp.addprocxtransf(f, 'a', 0)
    gp.addproclintransf(lambda g: lambda x: g(f(x)), [0], 'b')
    x = np.linspace(0, 4, 20)
    gp.addx(x, 0, proc='a')
    gp.addx(x, 1, proc='b')
    gp.addx(f(x), 2, proc=0)
    prior = gp.prior(raw=True)
    for i in range(3):
        for j in range(3):
            util.assert_equal(prior[0, 0], prior[i, j])

def test_addprocrescale():
    gp = lgp.GP()
    s = lambda x: x ** 2
    gp.addproc(lgp.ExpQuad(), 0)
    gp.addprocrescale(s, 'a', 0)
    gp.addproclintransf(lambda f: lambda x: s(x) * f(x), [0], 'b')
    x = np.linspace(0, 2, 20)
    gp.addx(x, 'base', proc=0)
    gp.addtransf({'base': s(x) * np.eye(len(x))}, 0)
    gp.addx(x, 1, proc='a')
    gp.addx(x, 2, proc='b')
    prior = gp.prior(raw=True)
    for i in range(3):
        for j in range(3):
            util.assert_allclose(prior[0, 0], prior[i, j], rtol=1e-15, atol=1e-15)

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

# TODO this fails in numpy 1.20, may work as versions are bumped
# def test_bad_array():
#     gp = lgp.GP(lgp.ExpQuad())
#     with pytest.raises(ValueError):
#         gp.addx({0: [[1, 2], 3]})

def test_not_empty():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addx([], 0)

def test_incompatible_dtypes():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    with pytest.raises(TypeError):
        gp.addx(np.zeros(1, 'd,d'), 1)
    
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(np.zeros(1, 'd,d'), 0)
    # gp.addx(np.zeros(1, 'i,i'), 1) # succeeds only if numpy >= 1.23
    with pytest.raises(TypeError):
        gp.addx(np.zeros(1, 'd,d,d'), 2)

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
        gp.addcov(gen.standard_normal((10, 10)), 0)
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

def test_zero_covblock():
    gp = lgp.GP()
    a = gen.standard_normal((10, 10))
    m = a.T @ a
    gp.addcov(m, 0)
    gp.addcov(m, 1)
    prior = gp.prior(raw=True)
    util.assert_equal(prior[0, 1], np.zeros_like(m))

def test_addcov_checks():
    a = gen.standard_normal((10, 10))
    b = np.copy(a)
    b[0, 0] = np.inf
    m = b.T @ b

    gp = lgp.GP()
    with pytest.raises(ValueError):
        gp.addcov(a, 0)
    with pytest.raises(ValueError):
        gp.addcov(m, 0)
    
    gp = lgp.GP(checksym=False)
    gp.addcov(a, 0)
    
    gp = lgp.GP(checkfinite=False)
    gp.addcov(m, 0)
    
    a = a @ a.T
    gp = lgp.GP()
    dec = lgp.GP.decompose(a)
    with pytest.raises(TypeError):
        gp.addcov({(0, 0): a}, decomps=dec)
    with pytest.raises(KeyError):
        gp.addcov({(0, 0): a}, decomps={1: dec})
    with pytest.raises(TypeError):
        gp.addcov({(0, 0): a}, decomps={0: a})
    b = gen.standard_normal((20, 20))
    b = b @ b.T
    bd = lgp.GP.decompose(b)
    with pytest.raises(ValueError):
        gp.addcov({(0, 0): a}, decomps={0: bd})

def test_makecovblock_checks():
    a = gen.standard_normal((10, 10))
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
    a, b, c, d = gen.standard_normal((4, 10, 10))
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
    a = gen.standard_normal((20, 20))
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
    util.assert_allclose(cov1, cov2, rtol=1e-15)

def test_new_element():
    gp = lgp.GP()
    gp._elements[0] = None
    with pytest.raises(Exception):
        gp.prior()

def test_given_checks():
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = gen.standard_normal((3, 20))
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
    x, y, z = gen.standard_normal((3, 20))
    gp.addx(x, 0)
    gp.addx(y, 1)
    cov = np.zeros(2 * x.shape)
    m1, c1 = gp.predfromdata({0: z}, 1, {(0, 0): cov}, raw=True)
    m2, c2 = gp.predfromdata({0: z}, 1, raw=True)
    util.assert_equal(m1, m2)
    util.assert_equal(c1, c2)

def test_pred_checks():
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = gen.standard_normal((3, 20))
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
    a = gen.standard_normal((20, 20))
    with pytest.raises(ValueError):
        gp.predfromdata({0: z}, 1, {(0, 0): a})
    gp._checkfinite = False
    gp.predfromdata({0: np.full_like(z, np.nan)}, 1)

def test_pred_all():
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = gen.standard_normal((3, 20))
    gp.addx(x, 0)
    gp.addx(y, 1)
    m1, c1 = gp.predfromdata({0: z}, raw=True)
    m2, c2 = gp.predfromdata({0: z}, [0, 1], raw=True)
    util.assert_equal(m1, m2)
    util.assert_equal(c1, c2)

def test_marginal_likelihood_separate():
    gp = lgp.GP(lgp.ExpQuad())
    x, y = gen.standard_normal((2, 20))
    gp.addx(x, 0)
    ml1 = gp.marginal_likelihood({0: y})
    l, r = gp.marginal_likelihood({0: y}, separate=True)
    ml2 = -1/2 * (l + r @ r)
    util.assert_allclose(ml2, ml1, rtol=1e-15)

def test_marginal_likelihood_checks():
    gp = lgp.GP(lgp.ExpQuad())
    x, y = gen.standard_normal((2, 20))
    gp.addx(x, 0)
    z = np.full_like(x, np.nan)
    with pytest.raises(ValueError):
        gp.marginal_likelihood({0: z})
    m = np.full(2 * x.shape, np.nan)
    with pytest.raises(ValueError):
        gp.marginal_likelihood({0: y}, {(0, 0): m})
    a = gen.standard_normal(2 * x.shape)
    with pytest.raises(ValueError):
        gp.marginal_likelihood({0: y}, {(0, 0): a})
    c = a.T @ a
    with pytest.warns(UserWarning):
        gp.marginal_likelihood({0: gvar.gvar(y, c)}, {(0, 0): c})

def test_marginal_likelihood_gvar():
    gp = lgp.GP(lgp.ExpQuad())
    x, y = gen.standard_normal((2, 20))
    gp.addx(x, 0)
    a = gen.standard_normal((20, 20))
    m = a.T @ a
    ml1 = gp.marginal_likelihood({0: gvar.gvar(y, m)})
    ml2 = gp.marginal_likelihood({0: y}, {(0, 0): m})
    util.assert_allclose(ml2, ml1, rtol=1e-15)

def test_singleton():
    dp = lgp.GP.DefaultProcess
    assert repr(dp) == 'DefaultProcess'
    with pytest.raises(NotImplementedError):
        dp()

def test_addtransf_abstract():
    def func():
        gp = lgp.GP(lgp.ExpQuad())
        gp.addx(0, 0)
        gp.addtransf({0: np.inf}, 1)
        return gp.prior(1, raw=True)
    with pytest.raises(ValueError):
        func()
    assert jax.jit(func)().item() == np.inf

def test_addlintransf_abstract():
    def func():
        gp = lgp.GP(lgp.ExpQuad())
        gp.addx(0, 0)
        gp.addlintransf(lambda x: x + 1, [0], 1)
        return gp.prior(1, raw=True)
    with pytest.raises(RuntimeError):
        func()
    assert jax.jit(func)().item() == 3

def test_addcov_abstract():
    def func():
        gp = lgp.GP(lgp.ExpQuad())
        gp.addcov({
            (0, 0): 1,
            (1, 1): 1,
            (0, 1): 1,
            (1, 0): 0,
        })
        return gp.prior([0, 1], raw=True)
    with pytest.raises(ValueError):
        func()
    cov = jax.jit(func)()
    assert cov[0, 1] == 1 or cov[0, 1] == 0

def test_marginal_likelihood_abstract():
    def func():
        gp = lgp.GP(lgp.ExpQuad())
        gp.addx(gen.standard_normal(10), 0)
        return gp.marginal_likelihood({0: np.full(10, np.nan)})
    
    with pytest.raises(ValueError):
        func()
    assert np.isnan(jax.jit(func)())
    
    def func(cov):
        gp = lgp.GP(lgp.ExpQuad())
        gp.addx(gen.standard_normal(10), 0)
        return gp.marginal_likelihood({0: gen.standard_normal(10)}, {(0, 0): cov})
    
    covnan = np.full((10, 10), np.nan)
    with pytest.raises(ValueError):
        func(covnan)
    assert np.isnan(jax.jit(func)(covnan))
    
    covasym = gen.standard_normal((10, 10))
    with pytest.raises(ValueError):
        func(covasym)
    jax.jit(func)(covasym)

def test_addcov_decomps():
    a = gen.standard_normal((10, 10))
    a = a @ a.T
    blocks = {
        (0, 0): a[:5, :5],
        (0, 1): a[:5, 5:],
        (1, 0): a[5:, :5],
        (1, 1): a[5:, 5:],
    }
    dec = lgp.GP.decompose(blocks[0, 0])
    dec1 = lgp.GP.decompose(blocks[1, 1])
    b = jnp.asarray(gen.standard_normal(5))
    
    def makez(**kw):
        gp = lgp.GP()
        gp.addcov(blocks, **kw)
        return gp.predfromdata({0: b}, 1)
    
    z1 = makez()
    z2 = makez(decomps={0: dec})
    z3 = makez(decomps={0: dec1})
    
    util.assert_similar_gvars(z1, z2)
    with pytest.raises(AssertionError):
        util.assert_similar_gvars(z1, z3)
    
    def makez(**kw):
        gp = lgp.GP()
        gp.addcov(blocks[0, 0], 0, **kw)
        gp.addcov(blocks[1, 1], 1)
        return gp.predfromdata({0: b}, 1)
    
    z1 = makez()
    z2 = makez(decomps=dec)
    
    util.assert_similar_gvars(z1, z2)

def test_matrices():
    shapes = [
        (),
        (10,),
        (3, 7),
    ]
    for sout in shapes:
        tensors = []
        gp = lgp.GP(lgp.ExpQuad())
        for i, sin in enumerate(shapes):
            tensor = gen.standard_normal(sout + sin)
            x = gen.standard_normal(sin)
            gp.addx(x, i)
            tensors.append(tensor)
            def transf(*args):
                out = 0
                for x, t, s in zip(args, tensors, shapes):
                    out += jnp.tensordot(t, x, axes=len(s))
                return out
            gp.addlintransf(transf, list(range(len(tensors))), 100 + i)
            
            matrices = gp._elements[100 + i].matrices(gp)
            tensors2 = [m.reshape(t.shape) for m, t in zip(matrices, tensors)]
            for t, t2 in zip(tensors, tensors2):
                util.assert_equal(t2, t)

def test_transf_outer():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(np.arange(5), 0)
    t = np.arange(5)
    gp.addtransf({0: t}, 1, axes=0)
    cov = gp.prior(raw=True)
    c0 = cov[0, 0]
    c1 = np.einsum('i,jl,k', t, c0, t)
    util.assert_equal(c1, cov[1, 1])

def test_transf_checks():
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError):
        gp.addtransf({}, 2)

def test_givencov_decomp():

    def genpd(n, rank=None, size=()):
        if not isinstance(size, tuple):
            size = (size,)
        if rank is None:
            rank = n
        m = gen.standard_normal(size + (n, rank))
        return m @ np.swapaxes(m, -2, -1)
    
    def decs(gp, keys, covrank=None):
        elems = gp._elements
        shapes = [elems[key].shape for key in keys]
        given = {k: np.zeros(s) for k, s in zip(keys, shapes)}
        size = sum(elems[key].size for key in keys)
        cov = genpd(size, covrank)
        slices = gp._slices(keys)
        givencov1 = {
            (ka, kb): cov[sla, slb].reshape(sa + sb)
            for (ka, sla, sa), (kb, slb, sb)
            in itertools.product(zip(keys, slices, shapes), repeat=2)
        }
        givencov2 = gp.decompose(cov)
        dec1, _ = gp._prior_decomp(given, givencov1)
        dec2, _ = gp._prior_decomp(given, givencov2)
        classes = (lgp._linalg.Woodbury, lgp._linalg.Woodbury2)
        assert not isinstance(dec1, classes)
        assert isinstance(dec2, classes)
        return dec1, dec2
    
    # generic matrix
    a = genpd(10)
    gp = lgp.GP()
    gp.addcov(a, 0)
    dec1, dec2 = decs(gp, [0])
    util.assert_close_decomps(dec2, dec1, rtol=1e-11)
    
    # short sandwich
    b = gen.standard_normal((len(a) // 2, len(a)))
    gp.addtransf({0: b}, 1)
    dec1, dec2 = decs(gp, [1])
    util.assert_close_decomps(dec2, dec1, rtol=1e-12)
    
    # tall sandwich
    c = gen.standard_normal((len(a) * 2, len(a)))
    gp.addtransf({0: c}, 2)
    dec1, dec2 = decs(gp, [2])
    util.assert_close_decomps(dec2, dec1, rtol=1e-9)
    
    # short and tall sandwich
    dec1, dec2 = decs(gp, [1, 2])
    util.assert_close_decomps(dec2, dec1, rtol=1e-10)

    # two generic matrices
    d = genpd(20)
    gp.addcov(d, 3)
    dec1, dec2 = decs(gp, [0, 3])
    util.assert_close_decomps(dec2, dec1, rtol=1e-8)
    
    # matrix, short and tall sandwich
    dec1, dec2 = decs(gp, [0, 1, 2])
    util.assert_close_decomps(dec2, dec1, rtol=1e-7)
    
    # short and tall sandwich, starting from different matrices
    e = gen.standard_normal((2 * len(d), len(d)))
    gp.addtransf({3: e}, 4)
    dec1, dec2 = decs(gp, [1, 4])
    util.assert_close_decomps(dec2, dec1, rtol=1e-9)
    
    # sum of two matrices
    f = genpd(len(a))
    gp.addcov(f, 5)
    gp.addtransf({0: 1, 5: 1}, 6)
    dec1, dec2 = decs(gp, [6])
    util.assert_close_decomps(dec2, dec1, rtol=1e-11)
    
    # the same matrix, twice
    gp.addcov({(k, q): a for k in [7, 8] for q in [7, 8]})
    dec1, dec2 = decs(gp, [7, 8])
    util.assert_close_decomps(dec2, dec1, rtol=1e-12)
    
    # low rank givencov
    dec1, dec2 = decs(gp, [0], len(a) // 2)
    util.assert_close_decomps(dec2, dec1, rtol=1) # TODO wildly inaccurate
    
def test_nochecksym_structured():
    gp = lgp.GP(lgp.ExpQuad(), checksym=False)
    gp.addx(np.zeros(1, 'd,d'), 0)
    gp.prior(0)

def test_nochecksym_tracer():
    def fun():
        gp = lgp.GP(lgp.ExpQuad(), checksym=False)
        gp.addx(np.zeros(1), 0)
        return gp.prior(0, raw=True)
    jax.jit(fun)()

def test_decompose_nd():
    cov = np.array(2)
    d1 = lgp.GP.decompose(cov)
    d2 = lgp.GP.decompose(cov.reshape(1, 1))
    d3 = lgp.GP.decompose(cov.reshape(1, 1, 1, 1))
    util.assert_close_decomps(d1, d2)
    util.assert_close_decomps(d1, d3)

def test_pred_woodbury():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    gp.addx(1, 1)
    cov = 2
    covdec = gp.decompose(cov)
    y1 = gp.predfromdata({0: 1}, 1, {(0, 0): cov})
    y2 = gp.predfromdata({0: 1}, 1, covdec)
    util.assert_similar_gvars(y1, y2, rtol=1e-15)

def test_pred_ambiguous_error_covariance():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    gp.addx(1, 1)
    with pytest.raises(ValueError):
        gp.predfromdata({0: gvar.gvar(0, 1)}, 1, {(0, 0): 2})

def test_pred_gvars_givencov():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    gp.addx(1, 1)
    mean, sdev = 1, 2
    y1 = gp.predfromdata({0: gvar.gvar(mean, sdev)}, 1)
    y2 = gp.predfromdata({0: mean}, 1, {(0, 0): sdev ** 2})
    y3 = gp.predfromdata({0: mean}, 1, gp.decompose(sdev ** 2))
    util.assert_similar_gvars(y1, y2)
    util.assert_similar_gvars(y1, y3)

def test_pred_fromfit_decomp():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(0, 0)
    gp.addx(1, 1)
    mean, sdev = 1, 2
    y0 = gp.predfromdata({0: gvar.gvar(mean, sdev)}, 0)
    y1 = gp.predfromfit({0: y0}, 1)
    y2 = gp.predfromfit({0: gvar.mean(y0)}, 1, {(0, 0): gvar.var(y0)})
    y3 = gp.predfromfit({0: gvar.mean(y0)}, 1, gp.decompose(gvar.var(y0)))
    util.assert_similar_gvars(y1, y2)
    util.assert_similar_gvars(y1, y3)
