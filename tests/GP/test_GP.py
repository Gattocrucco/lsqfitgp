# lsqfitgp/tests/GP/test_GP.py
#
# Copyright (c) 2020, 2022, 2023, 2026, Giacomo Petrillo
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

"""Test the `GP` class."""

import copy
import itertools

import gvar
import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp

import lsqfitgp as lgp
from lsqfitgp import _linalg
from tests import util


def test_prior_raw_shape():
    """Check the shape of the raw prior covariance of a 2D array of points."""
    gp = lgp.GP(lgp.ExpQuad()).addx(np.arange(20).reshape(2, 10), 'x')

    cov = gp.prior(raw=True)
    assert cov['x', 'x'].shape == (2, 10, 2, 10)

    cov = gp.prior('x', raw=True)
    assert cov.shape == (2, 10, 2, 10)


@pytest.mark.parametrize('shape', [(20,), (2, 3)])
def test_halfmatrix(shape, rng):
    """Check that computing half the covariance matrix gives the same result."""
    covs = []
    x = rng.standard_normal(shape)
    for checksym in [False, True]:
        gp = lgp.GP(lgp.ExpQuad(), checksym=checksym, halfmatrix=not checksym)
        gp = gp.addx({'x': x})
        covs.append(gp.prior('x', raw=True))
    util.assert_equal(*covs)


def test_transf_scalar():
    """Check that `addtransf` summing the parts of a `'cond'` kernel gives the sum."""
    gp = lgp.GP(lgp.ExpQuad() + lgp.Cauchy())
    x = np.arange(20)
    gp = gp.addx(x, 'x')
    cov1 = gp.prior('x', raw=True)

    gp = lgp.GP(
        lgp.ExpQuad(dim='f1').linop('cond', lgp.Cauchy(dim='f1'), lambda x: x['f0'])
    )
    y = np.empty((2, len(x)), '?,f8')
    y['f0'] = np.reshape([0, 1], (2, 1))
    y['f1'] = x
    gp = gp.addx(y[0], 'y0').addx(y[1], 'y1').addtransf({'y0': 1, 'y1': 1}, 'x')
    cov2 = gp.prior('x', raw=True)

    util.assert_equal(cov1, cov2)


def test_transf_vector():
    """Check `addtransf` with a vector coefficient against the explicit operation."""
    gp = lgp.GP(lgp.ExpQuad()).addx([0, 1], 'x').addtransf({'x': [-1, 1]}, 'y')
    prior = gp.prior()
    y1 = prior['y']
    y2 = prior['x'][1] - prior['x'][0]
    util.assert_same_gvars(y1, y2, atol=1e-12)


def test_cov(rng):
    """Check that `addcov` stores the given covariance matrix unchanged."""
    A = rng.standard_normal((20, 20))
    M1 = A.T @ A
    gp = lgp.GP().addcov(M1, 'M')
    M2 = gp.prior('M', raw=True)
    util.assert_equal(M1, M2)


def test_compare_transfs():
    """Check that `addtransf`, `addlintransf`, `deftransf` and `deflintransf` agree."""
    x = np.arange(20)

    def preparegp():
        return (
            lgp.GP()
            .defproc('a', lgp.ExpQuad())
            .defproc('b', lgp.Cauchy())
            .addx(x, 'ax', proc='a')
            .addx(x, 'bx', proc='b')
        )

    def finalizegp(gp):
        return gp.addx(x, 2, proc='ab1').addx(x, 3, proc='ab2')

    def checkgp(gp):
        keys = [0, 1, 2, 3]
        prior = gp.prior(keys, raw=True)
        for k1 in keys:
            for k2 in keys:
                util.assert_allclose(prior[0, 0], prior[k1, k2], atol=1e-15, rtol=1e-15)

    # with functions
    gp = preparegp()
    fa = lambda x: jnp.sin(x) + 0.5 * jnp.cos(x**2)
    fb = lambda x: 1 / (1 + jnp.exp(-x))
    gp = (
        gp.addtransf({'ax': np.diag(fa(x)), 'bx': np.diag(fb(x))}, 0)
        .addlintransf(lambda a, b: fa(x) * a + fb(x) * b, ['ax', 'bx'], 1)
        .deftransf('ab1', {'a': fa, 'b': fb})
        .deflintransf(
            'ab2', lambda a, b: lambda x: fa(x) * a(x) + fb(x) * b(x), ['a', 'b']
        )
    )
    gp = finalizegp(gp)
    checkgp(gp)

    # with scalars
    gp = (
        preparegp()
        .addtransf({'ax': 2, 'bx': 3}, 0)
        .addlintransf(lambda a, b: 2 * a + 3 * b, ['ax', 'bx'], 1)
        .deftransf('ab1', {'a': 2, 'b': 3})
        .deflintransf('ab2', lambda a, b: lambda x: 2 * a(x) + 3 * b(x), ['a', 'b'])
    )
    gp = finalizegp(gp)
    checkgp(gp)


def test_lintransf_checks():
    """Check that `addlintransf` raises on invalid keys and nonlinear functions."""
    gp = lgp.GP(lgp.ExpQuad()).addx(0, 0).addx(0, 1)
    with pytest.raises(KeyError):
        gp.addlintransf(lambda x, y: x + y, [0, 1], 0)
    with pytest.raises(ValueError, match='key can not be None'):
        gp.addlintransf(lambda x, y: x + y, [0, 1], None)
    with pytest.raises(KeyError):
        gp.addlintransf(lambda x, y: x + y, [0, 2], 2)
    with pytest.raises(RuntimeError):
        gp.addlintransf(lambda x, y: 1 + x + y, [0, 1], 2)
    with pytest.raises(RuntimeError):
        gp.addlintransf(lambda _x, _y: 1, [0, 1], 2)
    gp = gp.addlintransf(lambda x, y: 1 + x + y, [0, 1], 2, checklin=False)
    gp._checklin = False
    gp = gp.addlintransf(lambda x, y: 1 + x + y, [0, 1], 3)
    with pytest.raises(RuntimeError):
        gp.addlintransf(lambda x, y: 1 + x + y, [0, 1], 4, checklin=True)


def test_proclintransf_checks():
    """Check that `deflintransf` raises on invalid processes and nonlinear functions."""

    def makegp(**kw):
        return lgp.GP(**kw).defproc(0, lgp.ExpQuad()).defproc(1, lgp.ExpQuad())

    gp = makegp()
    with pytest.raises(KeyError):
        gp.deflintransf(0, lambda f, g: lambda x: f(x) + g(x), [0, 1])
    with pytest.raises(KeyError):
        gp.deflintransf(2, lambda f, g: lambda x: f(x) + g(x), [0, 2])
    with pytest.raises(RuntimeError):
        gp.deflintransf(2, lambda _f, _g: lambda _x: 1, [0, 1], checklin=True)
    with pytest.raises(RuntimeError):
        gp.deflintransf(
            2, lambda f, g: lambda x: 1 + f(x) + g(x), [0, 1], checklin=True
        )
    with pytest.raises(RuntimeError):
        gp.deflintransf(
            2, lambda f, g: lambda x: f(x) + g(x)[None, :], [0, 1], checklin=True
        )
    gp = gp.deflintransf(2, lambda f, g: lambda x: 1 + f(x) + g(x), [0, 1])
    gp = gp.deflintransf(3, lambda f, g: lambda x: f(x) + g(x), [0, 1], checklin=True)
    gp = makegp(checklin=True)
    with pytest.raises(RuntimeError):
        gp.deflintransf(2, lambda _f, _g: lambda _x: 1, [0, 1], checklin=None)


def test_proclintransf_mockup():
    """Check the linearity check of `deflintransf` on functions indexing fields."""

    def makegp(**kw):
        return lgp.GP(**kw).defproc(0, lgp.ExpQuad()).defproc(1, lgp.ExpQuad())

    gp = makegp()
    with pytest.raises(RuntimeError):
        gp.deflintransf(
            2,
            lambda f, g: lambda x: 1 + f(x['dim1']) + g(x['dim2']),
            [0, 1],
            checklin=True,
        )
    gp.deflintransf(
        2, lambda f, g: lambda x: f(x['dim1']) + g(x['dim2']), [0, 1], checklin=True
    )


def test_lintransf_matmul(rng):
    """Check the prior of a matrix multiplication defined with `addlintransf`."""
    gp = lgp.GP(lgp.ExpQuad())
    x = np.arange(20)
    gp = gp.addx(x, 0)
    m = rng.standard_normal((30, len(x)))
    gp = gp.addlintransf(lambda x: m @ x, [0], 1)
    prior = gp.prior([0, 1], raw=True)
    util.assert_allclose(m @ prior[0, 0] @ m.T, prior[1, 1], rtol=1e-11)
    util.assert_allclose(m @ prior[0, 1], prior[1, 1], rtol=1e-11)
    util.assert_allclose(prior[1, 0] @ m.T, prior[1, 1], rtol=1e-11)


def test_prior_gvar(rng):
    """Check that the gvar prior has zero mean and the covariance of the raw prior."""
    gp = (
        lgp.GP(lgp.ExpQuad())
        .addx(rng.standard_normal(20), 0)
        .addx(rng.standard_normal(15), 1)
        .addtransf({0: rng.standard_normal((15, 20))}, 2)
        .addlintransf(lambda x, y: jnp.real(jnp.fft.rfft(x + y)), [1, 2], 3)
    )
    m = rng.standard_normal((40, 40))
    gp = gp.addcov(m @ m.T, 4).addtransf({4: rng.standard_normal((8, 40)), 3: np.pi}, 5)
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
    """Check that `deflinop` with `'rescale'` equals multiplying by `Rescaling`."""
    gp = lgp.GP().defproc('a', lgp.ExpQuad())
    f = lambda x: x
    gp = gp.defproc('b1', lgp.ExpQuad() * lgp.Rescaling(stdfun=f)).deflinop(
        'b2', 'rescale', f, 'a'
    )
    x = np.arange(20)
    gp = gp.addx(x, 'x1', proc='b1').addx(x, 'x2', proc='b2')
    prior = gp.prior(['x1', 'x2'], raw=True)
    util.assert_allclose(prior['x1', 'x1'], prior['x2', 'x2'], rtol=1e-15)
    util.assert_equal(prior['x2', 'x1'], prior['x1', 'x2'].T)
    util.assert_equal(prior['x1', 'x2'], np.zeros(2 * x.shape))


def test_not_kernel():
    """Check that `GP` raises `TypeError` if the covariance function is not a kernel."""
    with pytest.raises(TypeError):
        lgp.GP(0)


def test_two_procs():
    """Check that defining twice a process with the same key raises `KeyError`."""
    gp = lgp.GP().defproc('a', lgp.ExpQuad())
    with pytest.raises(KeyError):
        gp.defproc('a', lgp.ExpQuad())


def test_default_kernel():
    """Check that `defproc` without kernel defines a process with the default kernel."""
    gp = lgp.GP(lgp.ExpQuad()).defproc('a')
    x = np.arange(20)
    gp = gp.addx(x, 'x1').addx(x, 'x2', proc='a')
    prior = gp.prior(raw=True)
    util.assert_equal(prior['x1', 'x1'], prior['x2', 'x2'])
    util.assert_equal(prior['x1', 'x2'], np.zeros(2 * x.shape))


def test_no_proc():
    """Check that `deftransf` raises `KeyError` if the source process does not exist."""
    gp = lgp.GP()
    with pytest.raises(KeyError):
        gp.deftransf('b', {'a': 1})


def test_invalid_factor():
    """Check that `deftransf` accepts scalar factors and rejects `None`."""
    gp = (
        lgp.GP()
        .defproc('a', lgp.ExpQuad())
        .deftransf('b', {'a': 0.0})
        .deftransf('c', {'a': np.array(0.0)})
    )
    with pytest.raises(TypeError):
        gp.deftransf('d', {'a': None})


def test_existing_proc():
    """Check that `deftransf` raises `KeyError` if the target process already exists."""
    gp = lgp.GP().defproc('a', lgp.ExpQuad())
    with pytest.raises(KeyError):
        gp.deftransf('a', {'a': 1})


def test_empty_proc():
    """Check that transformations of zero processes have null covariance."""
    x = np.arange(20)
    cov = (
        lgp.GP()
        .deftransf('a', {})
        .deflintransf('b', lambda: lambda _x: 0, [])
        .addx(x, 'ax', proc='a')
        .addx(x, 'bx', proc='b')
        .prior(raw=True)
    )
    util.assert_equal(cov['ax', 'ax'], np.zeros(2 * x.shape))
    util.assert_equal(cov['bx', 'bx'], np.zeros(2 * x.shape))


def test_already_defined():
    """Check that `deflinop` raises `KeyError` if the target process already exists."""
    gp = lgp.GP().defproc('a', lgp.ExpQuad())
    with pytest.raises(KeyError):
        gp.deflinop('a', 'diff', 1, 'a')


def test_proc_not_found():
    """Check that `deflinop` raises `KeyError` if the source process does not exist."""
    gp = lgp.GP()
    with pytest.raises(KeyError):
        gp.deflinop('b', 'diff', 1, 'a')


def test_defderiv():
    """Check that `defderiv` is equivalent to the `deriv` argument of `addx`."""
    x = np.arange(20)
    prior = (
        lgp.GP(lgp.ExpQuad())
        .defderiv('a', 1, lgp.GP.DefaultProcess)
        .addx(x, 0, proc='a')
        .addx(x, 1, deriv=1)
        .prior(raw=True)
    )
    util.assert_equal(prior[0, 0], prior[1, 1])
    util.assert_equal(prior[0, 0], prior[0, 1])


def test_defxtransf():
    """Check that `defxtransf` is equivalent to transforming the input points."""
    f = lambda x: x**2
    x = np.linspace(0, 4, 20)
    prior = (
        lgp.GP()
        .defproc(0, lgp.ExpQuad())
        .defxtransf('a', f, 0)
        .deflintransf('b', lambda g: lambda x: g(f(x)), [0])
        .addx(x, 0, proc='a')
        .addx(x, 1, proc='b')
        .addx(f(x), 2, proc=0)
        .prior(raw=True)
    )
    for i in range(3):
        for j in range(3):
            util.assert_equal(prior[0, 0], prior[i, j])


def test_defrescale():
    """Check that `defrescale` is equivalent to multiplying by a function."""
    s = lambda x: x**2
    x = np.linspace(0, 2, 20)
    prior = (
        lgp.GP()
        .defproc(0, lgp.ExpQuad())
        .defrescale('a', s, 0)
        .deflintransf('b', lambda f: lambda x: s(x) * f(x), [0])
        .addx(x, 'base', proc=0)
        .addtransf({'base': s(x) * np.eye(len(x))}, 0)
        .addx(x, 1, proc='a')
        .addx(x, 2, proc='b')
        .prior(raw=True)
    )
    for i in range(3):
        for j in range(3):
            util.assert_allclose(prior[0, 0], prior[i, j], rtol=1e-15, atol=1e-15)


def test_missing_proc():
    """Check that `addx` raises `KeyError` if the process does not exist."""
    gp = lgp.GP()
    with pytest.raises(KeyError):
        gp.addx(0, 0, proc='cippa')


def test_no_key():
    """Check that `addx` and `addcov` raise `ValueError` if the key is missing."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError, match='x is not dictionary but key is None'):
        gp.addx(0)
    with pytest.raises(ValueError, match='covblocks is not dictionary'):
        gp.addcov(1)


def test_redundant_key():
    """Check that `addx` and `addcov` raise `ValueError` if the key is given twice."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError, match='can not specify key if x is a dictionary'):
        gp.addx({0: 0}, 0)
    with pytest.raises(ValueError, match='can not specify key if covblocks'):
        gp.addcov({0: 1}, 0)


def test_none_key():
    """Check that `None` is rejected as key."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError, match='None key in x not allowed'):
        gp.addx({None: 0})

    gp = lgp.GP(lgp.ExpQuad()).addx(0, 0)
    with pytest.raises(ValueError, match='key can not be None'):
        gp.addtransf({0: 1}, None)

    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError, match='None key in covblocks not allowed'):
        gp.addcov({None: 1})


def test_nonsense_x():
    """Check that `addcov` raises `TypeError` on `None`."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(TypeError):
        gp.addcov(None, 0)


def test_key_already_used():
    """Check that reusing a key raises `KeyError`."""
    gp = lgp.GP(lgp.ExpQuad()).addx(0, 0)
    with pytest.raises(KeyError):
        gp.addx(0, 0)

    gp = lgp.GP(lgp.ExpQuad()).addx(0, 0)
    with pytest.raises(KeyError):
        gp.addtransf({0: 1}, 0)

    gp = lgp.GP(lgp.ExpQuad()).addx(0, 0)
    with pytest.raises(KeyError):
        gp.addcov(1, 0)


# def test_bad_array():
#     gp = lgp.GP(lgp.ExpQuad())
#     with pytest.raises(ValueError):
#         gp.addx({0: [[1, 2], 3]})


def test_incompatible_dtypes():
    """Check that `addx` raises `TypeError` on inputs with incompatible dtypes."""
    gp = lgp.GP(lgp.ExpQuad()).addx(0, 0)
    with pytest.raises(TypeError):
        gp.addx(np.zeros(1, 'd,d'), 1)

    gp = lgp.GP(lgp.ExpQuad()).addx(np.zeros(1, 'd,d'), 0)
    # gp.addx(np.zeros(1, 'i,i'), 1) # succeeds only if numpy >= 1.23
    with pytest.raises(TypeError):
        gp.addx(np.zeros(1, 'd,d,d'), 2)


def test_explicit_deriv():
    """Check that a derivative by field name on a plain input raises `ValueError`."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError, match='x has no fields but derivative has'):
        gp.addx(0, 0, deriv='x')


def test_missing_field():
    """Check that a derivative w.r.t. a missing field raises `ValueError`."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError, match="deriv field 'x' not in x"):
        gp.addx(np.array((0, 0), 'f8,f8'), 0, deriv='x')


def test_missing_key():
    """Check that `addtransf` raises `KeyError` on a missing key."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(KeyError):
        gp.addtransf({0: 1}, 1)


def test_nonsense_tensors():
    """Check that `addtransf` rejects non-numerical, infinite or misshaped factors."""
    gp = lgp.GP(lgp.ExpQuad()).addx(0, 0)
    with pytest.raises(TypeError):
        gp.addtransf({0: 'a'}, 1)
    with pytest.raises(ValueError, match=r'tensors\[0\] contains infs/nans'):
        gp.addtransf({0: np.inf}, 1)
    gp = gp.addx([0, 1], 1)
    with pytest.raises(ValueError, match='can not be multiplied with shape'):
        gp.addtransf({1: [1, 2, 3]}, 2)


def test_fail_broadcast():
    """Check that `addtransf` raises `ValueError` if the shapes do not broadcast."""
    gp = lgp.GP(lgp.ExpQuad()).addx([0, 1], 0).addx([0, 1, 2], 1)
    with pytest.raises(ValueError, match=r'with shapes \[\(2,\), \(3,\)\]$'):
        gp.addtransf({0: 1, 1: 1}, 2)


def test_addcov_wrong_blocks(rng):
    """Check that `addcov` rejects invalid or asymmetric covariance blocks."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError, match='odd number of axes'):
        gp.addcov(np.zeros((1, 1, 1)), 0)
    with pytest.raises(ValueError, match='of diagonal block 0 is not symmetric'):
        gp.addcov(np.zeros((1, 2, 2, 1)), 0)
    with pytest.raises(ValueError, match=r'^diagonal block 0 is not symmetric'):
        gp.addcov(rng.standard_normal((10, 10)), 0)
    with pytest.raises(KeyError):
        gp.addcov({(0, 0): 1, (0, 1): 0})
    with pytest.raises(ValueError, match=r'is not \(2, 3\) as expected'):
        gp.addcov(
            {(0, 0): np.ones((2, 2)), (1, 1): np.ones((3, 3)), (0, 1): np.ones((3, 2))}
        )
    with pytest.raises(
        ValueError, match=r'^block \(0, 1\) is not the transpose of block \(1, 0\)$'
    ):
        gp.addcov(
            {
                (0, 0): np.ones((2, 2)),
                (1, 1): np.ones((3, 3)),
                (0, 1): np.ones((2, 3)),
                (1, 0): np.zeros((3, 2)),
            }
        )


def test_addcov_no_checksym():
    """Check that `addcov` accepts asymmetric blocks with `checksym=False`."""
    gp = lgp.GP(lgp.ExpQuad(), checksym=False)
    gp = gp.addcov(
        {
            (0, 0): np.ones((2, 2)),
            (1, 1): np.ones((3, 3)),
            (0, 1): np.ones((2, 3)),
            (1, 0): np.zeros((3, 2)),
        }
    )


def test_addcov_missing_block():
    """Check that `addcov` fills a missing off-diagonal block with the transpose."""
    gp = lgp.GP(lgp.ExpQuad())
    gp = gp.addcov(
        {(0, 0): np.ones((2, 2)), (1, 1): np.ones((3, 3)), (0, 1): np.ones((2, 3))}
    )
    prior = gp.prior(raw=True)
    util.assert_equal(prior[0, 1], prior[1, 0].T)


def test_new_proc():
    """Check that `prior` raises `TypeError` on a process of unknown type."""
    gp = lgp.GP()
    gp._procs[0] = None
    gp = gp.addx(0, 0, proc=0)
    with pytest.raises(TypeError):
        gp.prior(0, raw=True)


def test_partial_derivative():
    """Check that a derivative w.r.t. a field equals one w.r.t. a plain input."""
    gp = lgp.GP(lgp.ExpQuad())
    x = np.arange(20)
    y = np.zeros(len(x), 'f8,f8')
    y['f0'] = x
    gp = gp.addx(y, 0, deriv='f0')
    cov1 = gp.prior(0, raw=True)

    gp = lgp.GP(lgp.ExpQuad()).addx(x, 0, deriv=1)
    cov2 = gp.prior(0, raw=True)

    util.assert_equal(cov1, cov2)


def test_zero_covblock(rng):
    """Check that blocks added by separate `addcov` calls are uncorrelated."""
    gp = lgp.GP()
    a = rng.standard_normal((10, 10))
    m = a.T @ a
    gp = gp.addcov(m, 0).addcov(m, 1)
    prior = gp.prior(raw=True)
    util.assert_equal(prior[0, 1], np.zeros_like(m))


def test_addcov_checks(rng):
    """Check the input validation of `addcov`, including the `decomps` argument."""
    a = rng.standard_normal((10, 10))
    b = np.copy(a)
    b[0, 0] = np.inf
    m = b.T @ b

    gp = lgp.GP()
    with pytest.raises(ValueError, match='diagonal block 0 is not symmetric'):
        gp.addcov(a, 0)
    with pytest.raises(ValueError, match=r'block \(0, 0\) not finite'):
        gp.addcov(m, 0)

    gp = lgp.GP(checksym=False).addcov(a, 0)

    gp = lgp.GP(checkfinite=False).addcov(m, 0)

    a = a @ a.T
    gp = lgp.GP()
    dec = lgp.GP.decompose(a)
    with pytest.raises(TypeError):
        gp.addcov({(0, 0): a}, decomps=dec)
    with pytest.raises(KeyError):
        gp.addcov({(0, 0): a}, decomps={1: dec})
    with pytest.raises(TypeError):
        gp.addcov({(0, 0): a}, decomps={0: a})
    b = rng.standard_normal((20, 20))
    b = b @ b.T
    bd = lgp.GP.decompose(b)
    with pytest.raises(
        ValueError, match='decomposition matrix size 20 != diagonal block size 10'
    ):
        gp.addcov({(0, 0): a}, decomps={0: bd})


def test_makecovblock_checks(rng):
    """Check that the symmetry and finiteness checks run when computing the prior."""
    a = rng.standard_normal((10, 10))
    b = np.copy(a)
    b[0, 0] = np.inf
    m = b.T @ b

    gp = lgp.GP(checksym=False).addcov(a, 0)
    gp._checksym = True
    with pytest.raises(RuntimeError):
        gp.prior(raw=True)

    gp = lgp.GP(checkfinite=False).addcov(m, 0)
    gp._checkfinite = True
    with pytest.raises(RuntimeError):
        gp.prior(raw=True)

    gp = lgp.GP(checksym=False, checkpos=False).addcov(a, 0)
    gp.prior(raw=True)

    gp = lgp.GP(checkfinite=False, checkpos=False).addcov(m, 0)
    gp.prior(raw=True)


def test_covblock_checks(rng):
    """Check that the symmetry check of the prior catches asymmetric blocks."""
    a, b, c, d = rng.standard_normal((4, 10, 10))
    m = a.T @ a
    n = b.T @ b
    gp = lgp.GP(checksym=False, checkpos=False)
    gp = gp.addcov({(0, 0): m, (1, 1): n, (0, 1): c, (1, 0): d})
    gp2 = copy.deepcopy(gp)
    gp._checksym = True
    with pytest.raises(RuntimeError):
        gp.prior(raw=True)
    gp2.prior(raw=True)


def test_solver_cache():
    """Check that repeating a prediction gives identical results."""
    gp = lgp.GP(lgp.ExpQuad())
    x = np.linspace(0, 1, 10)
    y = np.linspace(1, 2, 10)
    z = np.zeros_like(x)
    gp = gp.addx(x, 0).addx(y, 1)
    m1, c1 = gp.predfromdata({0: z}, 1, raw=True)
    m2, c2 = gp.predfromdata({0: z}, 1, raw=True)
    util.assert_equal(m1, m2)
    util.assert_equal(c1, c2)


def test_checkpos(rng):
    """Check that an indefinite prior raises `LinAlgError`, unless unchecked."""
    a = rng.standard_normal((20, 20))
    m = a.T @ a
    w, v = np.linalg.eigh(m)
    w[np.arange(len(w)) % 2 == 1] *= -1
    m = (v * w) @ v.T

    gp = lgp.GP().addcov(m, 0)
    with pytest.raises(np.linalg.LinAlgError):
        gp.prior()

    gp._checkpositive = False
    gp.prior()


def test_priorpoints_cache():
    """Check that duplicate points in different keys have identical covariances."""
    gp = lgp.GP(lgp.ExpQuad())
    x = np.arange(20)
    gp = gp.addx(x, 0).addx(x, 1)
    prior = gp.prior()
    cov = gvar.evalcov(prior)
    util.assert_equal(cov[0, 0], cov[1, 1])
    util.assert_equal(cov[0, 0], cov[0, 1])


def test_priortransf():
    """Check that the gvar prior of a transformation matches the raw prior."""
    gp = lgp.GP(lgp.ExpQuad())
    x, y = np.arange(40).reshape(2, -1)
    gp = gp.addx(x, 0).addx(y, 1)
    gp = gp.addtransf({0: x, 1: y}, 2)
    cov1 = gp.prior(2, raw=True)
    u = gp.prior(2)
    cov2 = gvar.evalcov(u)
    util.assert_allclose(cov1, cov2, rtol=1e-15)


def test_new_element():
    """Check that `prior` raises `AttributeError` on an element of unknown type."""
    gp = lgp.GP()
    gp._elements[0] = None
    with pytest.raises(AttributeError):
        gp.prior()


def test_given_checks(rng):
    """Check the validation of the data passed to `predfromdata`."""
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = rng.standard_normal((3, 20))
    gp = gp.addx(x, 0).addx(y, 1)
    with pytest.raises(TypeError):
        gp.predfromdata(0, 1)
    with pytest.raises(TypeError):
        gp.predfromdata({0: z}, 1, givencov=0)
    with pytest.raises(KeyError):
        gp.predfromdata({2: z}, 1)
    with pytest.raises(ValueError, match=r'given\[0\] has shape'):
        gp.predfromdata({0: z[:-1]}, 1)
    with pytest.raises(TypeError):
        gp.predfromdata({0: np.empty_like(z, str)}, 1)


def test_zero_givencov(rng):
    """Check that a null error covariance is equivalent to no error covariance."""
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = rng.standard_normal((3, 20))
    gp = gp.addx(x, 0).addx(y, 1)
    cov = np.zeros(2 * x.shape)
    m1, c1 = gp.predfromdata({0: z}, 1, {(0, 0): cov}, raw=True)
    m2, c2 = gp.predfromdata({0: z}, 1, raw=True)
    util.assert_equal(m1, m2)
    util.assert_equal(c1, c2)


def test_pred_checks(rng):
    """Check the input validation of `pred` and `predfromdata`."""
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = rng.standard_normal((3, 20))
    gp = gp.addx(x, 0).addx(y, 1)
    with pytest.raises(
        ValueError, match='you must specify if `given` is data or fit result'
    ):
        gp.pred({0: z}, 1)
    with pytest.raises(ValueError, match='both keepcorr=True and raw=True'):
        gp.predfromdata({0: z}, 1, raw=True, keepcorr=True)
    with pytest.raises(ValueError, match='mean of `given` is not finite'):
        gp.predfromdata({0: np.full_like(z, np.nan)}, 1)
    with pytest.raises(ValueError, match='covariance matrix of `given` is not finite'):
        gp.predfromdata({0: z}, 1, {(0, 0): np.full(2 * x.shape, np.nan)})
    a = rng.standard_normal((20, 20))
    with pytest.raises(
        ValueError, match='covariance matrix of `given` is not symmetric'
    ):
        gp.predfromdata({0: z}, 1, {(0, 0): a})
    gp._checkfinite = False
    gp.predfromdata({0: np.full_like(z, np.nan)}, 1)


def test_pred_all(rng):
    """Check that `predfromdata` predicts all keys by default."""
    gp = lgp.GP(lgp.ExpQuad())
    x, y, z = rng.standard_normal((3, 20))
    gp = gp.addx(x, 0).addx(y, 1)
    m1, c1 = gp.predfromdata({0: z}, raw=True)
    m2, c2 = gp.predfromdata({0: z}, [0, 1], raw=True)
    util.assert_equal(m1, m2)
    util.assert_equal(c1, c2)


def test_marginal_likelihood_checks(rng):
    """Check the input validation of `marginal_likelihood`."""
    gp = lgp.GP(lgp.ExpQuad())
    x, y = rng.standard_normal((2, 20))
    gp = gp.addx(x, 0)
    z = np.full_like(x, np.nan)
    with pytest.raises(ValueError, match='mean of `given` is not finite'):
        gp.marginal_likelihood({0: z})
    m = np.full(2 * x.shape, np.nan)
    with pytest.raises(ValueError, match='covariance matrix of `given` is not finite'):
        gp.marginal_likelihood({0: y}, {(0, 0): m})
    a = rng.standard_normal(2 * x.shape)
    with pytest.raises(
        ValueError, match='covariance matrix of `given` is not symmetric'
    ):
        gp.marginal_likelihood({0: y}, {(0, 0): a})
    c = a.T @ a
    with pytest.warns(UserWarning, match='specified both explicitly and with gvars'):
        gp.marginal_likelihood({0: gvar.gvar(y, c)}, {(0, 0): c})


def test_marginal_likelihood_gvar(rng):
    """Check `marginal_likelihood` with errors as gvars or as covariance matrix."""
    gp = lgp.GP(lgp.ExpQuad())
    x, y = rng.standard_normal((2, 20))
    gp = gp.addx(x, 0)
    a = rng.standard_normal((20, 20))
    m = a.T @ a
    ml1 = gp.marginal_likelihood({0: gvar.gvar(y, m)})
    ml2 = gp.marginal_likelihood({0: y}, {(0, 0): m})
    util.assert_allclose(ml2, ml1, rtol=1e-15)


def test_singleton():
    """Check the representation of `GP.DefaultProcess` and that it can't be called."""
    dp = lgp.GP.DefaultProcess
    assert repr(dp) == 'DefaultProcess'
    with pytest.raises(NotImplementedError):
        dp()


def test_addtransf_abstract():
    """Check that the finiteness check of `addtransf` is skipped under jit."""

    def func():
        gp = lgp.GP(lgp.ExpQuad())
        gp = gp.addx(0, 0).addtransf({0: np.inf}, 1)
        return gp.prior(1, raw=True)

    with pytest.raises(ValueError, match=r'tensors\[0\] contains infs/nans'):
        func()
    assert jit(func)().item() == np.inf


def test_addlintransf_abstract():
    """Check that the linearity check of `addlintransf` is skipped under jit."""

    def func():
        gp = lgp.GP(lgp.ExpQuad())
        gp = gp.addx(0, 0).addlintransf(lambda x: x + 1, [0], 1)
        return gp.prior(1, raw=True)

    with pytest.raises(RuntimeError):
        func()
    assert jit(func)().item() == 3


def test_addcov_abstract():
    """Check that the symmetry check of `addcov` is skipped under jit."""

    def func():
        gp = lgp.GP(lgp.ExpQuad())
        gp = gp.addcov({(0, 0): 1, (1, 1): 1, (0, 1): 1, (1, 0): 0})
        return gp.prior([0, 1], raw=True)

    with pytest.raises(ValueError, match='is not the transpose of block'):
        func()
    cov = jit(func)()
    assert cov[0, 1] == 1 or cov[0, 1] == 0


def test_marginal_likelihood_abstract(rng):
    """Check which input checks of `marginal_likelihood` are skipped under jit."""

    def func():
        gp = lgp.GP(lgp.ExpQuad())
        gp = gp.addx(rng.standard_normal(10), 0)
        return gp.marginal_likelihood({0: np.full(10, np.nan)})

    with pytest.raises(ValueError, match='mean of `given` is not finite'):
        func()
    # this once did not raise under jit, but now it does, I guess due to a more
    # eager jit implementation?

    def func(cov):
        gp = lgp.GP(lgp.ExpQuad())
        gp = gp.addx(rng.standard_normal(10), 0)
        return gp.marginal_likelihood({0: rng.standard_normal(10)}, {(0, 0): cov})

    covnan = np.full((10, 10), np.nan)
    with pytest.raises(ValueError, match='covariance matrix of `given` is not finite'):
        func(covnan)
    assert np.isnan(jit(func)(covnan))

    covasym = rng.standard_normal((10, 10))
    with pytest.raises(
        ValueError, match='covariance matrix of `given` is not symmetric'
    ):
        func(covasym)
    jit(func)(covasym)


def test_addcov_decomps(rng):
    """Check that `addcov` uses the decompositions passed with `decomps`."""
    a = rng.standard_normal((10, 10))
    a = a @ a.T
    blocks = {
        (0, 0): a[:5, :5],
        (0, 1): a[:5, 5:],
        (1, 0): a[5:, :5],
        (1, 1): a[5:, 5:],
    }
    dec = lgp.GP.decompose(blocks[0, 0])
    dec1 = lgp.GP.decompose(blocks[1, 1])
    b = jnp.asarray(rng.standard_normal(5))

    def makez(**kw):
        gp = lgp.GP().addcov(blocks, **kw)
        return gp.predfromdata({0: b}, 1)

    z1 = makez()
    z2 = makez(decomps={0: dec})
    z3 = makez(decomps={0: dec1})

    util.assert_similar_gvars(z1, z2)
    with pytest.raises(AssertionError):
        util.assert_similar_gvars(z1, z3)

    def makez(**kw):
        gp = lgp.GP().addcov(blocks[0, 0], 0, **kw).addcov(blocks[1, 1], 1)
        return gp.predfromdata({0: b}, 1)

    z1 = makez()
    z2 = makez(decomps=dec)

    util.assert_similar_gvars(z1, z2)


def test_matrices(rng):
    """Check that `matrices` of a linear transformation returns its tensors."""
    shapes = [(), (10,), (3, 7)]
    for sout in shapes:
        tensors = []
        gp = lgp.GP(lgp.ExpQuad())
        for i, sin in enumerate(shapes):
            tensor = rng.standard_normal(sout + sin)
            x = rng.standard_normal(sin)
            gp = gp.addx(x, i)
            tensors.append(tensor)

            def transf(*args):
                out = 0
                for x, t, s in zip(args, tensors, shapes, strict=False):  # noqa: B023, gp is recreated along with tensors
                    out += jnp.tensordot(t, x, axes=len(s))
                return out

            gp = gp.addlintransf(transf, list(range(len(tensors))), 100 + i)

            matrices = gp._elements[100 + i].matrices(gp)
            tensors2 = [
                m.reshape(t.shape) for m, t in zip(matrices, tensors, strict=True)
            ]
            for t, t2 in zip(tensors, tensors2, strict=True):
                util.assert_equal(t2, t)


def test_transf_outer():
    """Check `addtransf` with `axes=0`, i.e., an outer product."""
    gp = lgp.GP(lgp.ExpQuad())
    gp = gp.addx(np.arange(5), 0)
    t = np.arange(5)
    gp = gp.addtransf({0: t}, 1, axes=0)
    cov = gp.prior(raw=True)
    c0 = cov[0, 0]
    c1 = np.einsum('i,jl,k', t, c0, t)
    util.assert_equal(c1, cov[1, 1])


def test_transf_checks():
    """Check that `addtransf` raises `ValueError` on an empty transformation."""
    gp = lgp.GP(lgp.ExpQuad())
    with pytest.raises(ValueError, match='empty tensors'):
        gp.addtransf({}, 2)


@pytest.mark.skip('Woodbury currently un-implemented')
def test_givencov_decomp(rng):  # noqa: PLR0915
    """Check a decomposed error covariance against a dense one in the decomposition."""

    def genpd(n, rank=None, size=()):
        if not isinstance(size, tuple):
            size = (size,)
        if rank is None:
            rank = n
        m = rng.standard_normal((*size, n, rank))
        return m @ np.swapaxes(m, -2, -1)

    def decs(gp, keys, covrank=None):
        elems = gp._elements
        shapes = [elems[key].shape for key in keys]
        given = {k: np.zeros(s) for k, s in zip(keys, shapes, strict=True)}
        size = sum(elems[key].size for key in keys)
        cov = genpd(size, covrank)
        slices = gp._slices(keys)
        givencov1 = {
            (ka, kb): cov[sla, slb].reshape(sa + sb)
            for (ka, sla, sa), (kb, slb, sb) in itertools.product(
                zip(keys, slices, shapes, strict=True), repeat=2
            )
        }
        givencov2 = gp.decompose(cov)
        dec1, _ = gp._prior_decomp(given, givencov1)
        dec2, _ = gp._prior_decomp(given, givencov2)
        classes = (_linalg.Woodbury, _linalg.Woodbury2)
        assert not isinstance(dec1, classes)
        assert isinstance(dec2, classes)
        return dec1, dec2

    # generic matrix
    a = genpd(10)
    gp = lgp.GP().addcov(a, 0)
    dec1, dec2 = decs(gp, [0])
    util.assert_close_decomps(dec2, dec1, rtol=1e-11)

    # short sandwich
    b = rng.standard_normal((len(a) // 2, len(a)))
    gp.addtransf({0: b}, 1)
    dec1, dec2 = decs(gp, [1])
    util.assert_close_decomps(dec2, dec1, rtol=1e-7)

    # tall sandwich
    c = rng.standard_normal((len(a) * 2, len(a)))
    gp = gp.addtransf({0: c}, 2)
    dec1, dec2 = decs(gp, [2])
    util.assert_close_decomps(dec2, dec1, rtol=1e-9)

    # short and tall sandwich
    dec1, dec2 = decs(gp, [1, 2])
    util.assert_close_decomps(dec2, dec1, rtol=1e-10)

    # two generic matrices
    d = genpd(20)
    gp = gp.addcov(d, 3)
    dec1, dec2 = decs(gp, [0, 3])
    util.assert_close_decomps(dec2, dec1, rtol=1e-8)

    # matrix, short and tall sandwich
    dec1, dec2 = decs(gp, [0, 1, 2])
    util.assert_close_decomps(dec2, dec1, rtol=1e-7)

    # short and tall sandwich, starting from different matrices
    e = rng.standard_normal((2 * len(d), len(d)))
    gp = gp.addtransf({3: e}, 4)
    dec1, dec2 = decs(gp, [1, 4])
    util.assert_close_decomps(dec2, dec1, rtol=1e-9)

    # sum of two matrices
    f = genpd(len(a))
    gp = gp.addcov(f, 5)
    gp = gp.addtransf({0: 1, 5: 1}, 6)
    dec1, dec2 = decs(gp, [6])
    util.assert_close_decomps(dec2, dec1, rtol=1e-11)

    # the same matrix, twice
    gp = gp.addcov({(k, q): a for k in [7, 8] for q in [7, 8]})
    dec1, dec2 = decs(gp, [7, 8])
    util.assert_close_decomps(dec2, dec1, rtol=1e-10)

    # low rank givencov
    dec1, dec2 = decs(gp, [0], len(a) // 2)
    util.assert_close_decomps(dec2, dec1, rtol=1)


def test_nochecksym_structured():
    """Check the prior of structured inputs with `checksym=False, halfmatrix=True`."""
    gp = lgp.GP(lgp.ExpQuad(), checksym=False, halfmatrix=True)
    gp = gp.addx(np.zeros(1, 'd,d'), 0)
    gp.prior(0, raw=True)


def test_nochecksym_structured_jit():
    """Check `test_nochecksym_structured` under jit."""
    jit(test_nochecksym_structured)()


def test_nochecksym_tracer():
    """Check the prior with `checksym=False, halfmatrix=True` under jit."""

    def fun():
        gp = lgp.GP(lgp.ExpQuad(), checksym=False, halfmatrix=True)
        gp = gp.addx(np.zeros(1), 0)
        return gp.prior(0, raw=True)

    jit(fun)()


def test_decompose_nd():
    """Check that `GP.decompose` treats 0d, 2d and 4d arrays of size 1 alike."""
    cov = np.array(2)
    d1 = lgp.GP.decompose(cov)
    d2 = lgp.GP.decompose(cov.reshape(1, 1))
    d3 = lgp.GP.decompose(cov.reshape(1, 1, 1, 1))
    util.assert_close_decomps(d1, d2)
    util.assert_close_decomps(d1, d3)


@pytest.mark.skip('Woodbury currently un-implemented')
def test_pred_woodbury():
    """Check `predfromdata` with the error covariance given as a decomposition."""
    gp = lgp.GP(lgp.ExpQuad())
    gp = gp.addx(0, 0)
    gp = gp.addx(1, 1)
    cov = 2
    covdec = gp.decompose(cov)
    y1 = gp.predfromdata({0: 1}, 1, {(0, 0): cov})
    y2 = gp.predfromdata({0: 1}, 1, covdec)
    util.assert_similar_gvars(y1, y2, rtol=1e-15)


def test_pred_ambiguous_error_covariance():
    """Check that `predfromdata` rejects gvar data together with an error covariance."""
    gp = lgp.GP(lgp.ExpQuad())
    gp = gp.addx(0, 0)
    gp = gp.addx(1, 1)
    with pytest.raises(
        ValueError, match='separate covariance matrix has been provided'
    ):
        gp.predfromdata({0: gvar.gvar(0, 1)}, 1, {(0, 0): 2})


def test_pred_gvars_givencov():
    """Check `predfromdata` with errors as gvars or as covariance matrix."""
    gp = lgp.GP(lgp.ExpQuad())
    gp = gp.addx(0, 0)
    gp = gp.addx(1, 1)
    mean, sdev = 1, 2
    y1 = gp.predfromdata({0: gvar.gvar(mean, sdev)}, 1)
    y2 = gp.predfromdata({0: mean}, 1, {(0, 0): sdev**2})
    # y3 = gp.predfromdata({0: mean}, 1, gp.decompose(sdev ** 2)) # woodbury, currently un-implemented
    util.assert_similar_gvars(y1, y2)
    # util.assert_similar_gvars(y1, y3)


def test_pred_fromfit_gvars_givencov():
    """Check `predfromfit` with errors as gvars or as covariance matrix."""
    gp = lgp.GP(lgp.ExpQuad())
    gp = gp.addx(0, 0)
    gp = gp.addx(1, 1)
    mean, sdev = 1, 2
    y0 = gp.predfromdata({0: gvar.gvar(mean, sdev)}, 0)
    y1 = gp.predfromfit({0: y0}, 1, keepcorr=False)
    y2 = gp.predfromfit({0: gvar.mean(y0)}, 1, {(0, 0): gvar.var(y0)}, keepcorr=False)
    # y3 = gp.predfromfit({0: gvar.mean(y0)}, 1, gp.decompose(gvar.var(y0)), keepcorr=False) # woodbury, currently un-implemented
    util.assert_similar_gvars(y1, y2)
    # util.assert_similar_gvars(y1, y3)
