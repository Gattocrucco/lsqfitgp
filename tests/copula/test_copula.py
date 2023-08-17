# lsqfitgp/tests/test_copula.py
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

""" Test the copula module """

import functools
import contextlib
import numbers

import numpy as np
import gvar
import jax
import pytest
from pytest import mark
from scipy import stats

import lsqfitgp as lgp

from .. import util

@contextlib.contextmanager
def jaxconfig(**opts):
    prev = {k: jax.config.read(k) for k in opts}
    try:
        for k, v in opts.items():
            jax.config.update(k, v)
        yield
    finally:
        for k, v in prev.items():
            jax.config.update(k, v)

class CopulaFactoryTestBase:
    """
    Base class for tests of a CopulaFactory subclass
    """

    testfor = {}
    
    def __init_subclass__(cls):
        assert cls.__name__.startswith('Test')
        attrs = dict(
            copcls=getattr(lgp.copula, cls.__name__[4:].lower()),
        )
        for k, v in attrs.items():
            if not hasattr(cls, k):
                setattr(cls, k, v)
        specialmethods = dict(
            scipy_params=staticmethod,
            rvs=classmethod,
        )
        for name, kind in specialmethods.items():
            if not isinstance(meth := getattr(cls, name), kind):
                setattr(cls, name, kind(meth))
        __class__.testfor[cls.copcls.__name__] = cls

    params = ()
    recparams = ()
    accurate_range = (-np.inf, np.inf)
    
    def scipy_params(*params):
        return params

    def cdf(self):
        distrname = self.copcls.__name__
        distr = getattr(stats, distrname)
        params = self.scipy_params(*self.params)
        cdf = distr(*params).cdf
        cdf0 = cdf(self.accurate_range[0])
        cdf1 = cdf(self.accurate_range[1])
        @functools.wraps(cdf)
        def clipped_cdf(x):
            return (np.clip(cdf(x), cdf0, cdf1) - cdf0) / (cdf1 - cdf0)
        return clipped_cdf

    def rvs(cls, *params, size=(), random_state=None):
        distrname = cls.copcls.__name__
        distr = getattr(stats, distrname)
        params = cls.scipy_params(*params)
        return distr.rvs(*params, size=size, random_state=random_state)

    @classmethod
    def recrvs(cls, level):
        
        distrname = cls.copcls.__name__
        distr = getattr(stats, distrname)
        
        def rvs(size, rng):
            if level > 0:
                params = []
                for param in cls.recparams:
                    if isinstance(param, str):
                        rvs = cls.testfor[param].recrvs(level - 1)
                        param = rvs(size, rng)
                    params.append(param)
            else:
                params = cls.params
            return cls.rvs(*params, size=size, random_state=rng)
                    
        return rvs

    @classmethod
    def trim(cls, x, corendim):
        ok = x == np.clip(x, *cls.accurate_range)
        axes = tuple(range(x.ndim - corendim, x.ndim))
        ok = np.all(ok, axis=axes)
        return x[np.nonzero(ok) + (slice(None),) * corendim]

    @classmethod
    def convert_recparams(cls, level):
        if level > 0:
            params = []
            for param in cls.recparams:
                if isinstance(param, str):
                    test = cls.testfor[param]
                    param = (test.copcls, *test.convert_recparams(level - 1))
                params.append(param)
        else:
            params = cls.params
        return params
        
    @pytest.fixture
    def name(self, request):
        return request.node.nodeid

    def test_invfcn_errorprop(self, name, rng):
        variables = self.copcls(name, *self.params)
        assert np.ndim(variables) in (0, 1)
        assert np.ndim(variables) == 0 or variables.size > 1
        x = gvar.gvar(
            rng.standard_normal(variables.shape),
            rng.gamma(10, 1/10, variables.shape),
        )
        invfcn = gvar.BufferDict.invfcn[name]
        y = invfcn(x)
        deriv = jax.jacfwd(invfcn)(gvar.mean(x))
        ymean = invfcn(gvar.mean(x))
        ycov = np.dot(deriv * gvar.var(x), deriv.T)
        util.assert_close_matrices(gvar.mean(y), ymean, rtol=1e-6)
        util.assert_close_matrices(gvar.evalcov(y).reshape(2 * y.shape), ycov, rtol=1e-6)
    
    def test_overwrite(self, name):
        gvar.BufferDict.add_distribution(name, gvar.exp)
        with pytest.raises(ValueError):
            self.copcls(name, *self.params)
        gvar.BufferDict.del_distribution(name)
        self.copcls(name, *self.params)
        self.copcls(name, *self.params)

    def test_bufferdict_gvar(self, name):
        key = f'{name}(x)'
        b = gvar.BufferDict({
            key: self.copcls(name, *self.params)
        })
        x = b['x']
        mean = gvar.mean(b[key])
        xmean = self.copcls.invfcn(mean, *self.params)
        jac = jax.jacfwd(self.copcls.invfcn)(mean, *self.params)
        xcov = np.dot(jac * gvar.var(b[key]), jac.T)
        util.assert_close_matrices(gvar.mean(x), xmean, rtol=1e-6)
        util.assert_close_matrices(gvar.evalcov(x).reshape(2 * x.shape), xcov, rtol=1e-6)

    def test_bufferdict(self, name):
        key = f'{name}(x)'
        variables = self.copcls(name, *self.params)
        b = gvar.BufferDict({
            key: np.zeros_like(variables, float)
        })
        x = b['x']
        x2 = self.copcls.invfcn(b[key], *self.params)
        np.testing.assert_allclose(x, x2, rtol=1e-6)

    def test_continuity_zero(self, name):
        """ check that invfcn is continuous in zero, since it is a common
        cutpoint to switch from ppf(cdf(·)) to isf(sf(·)) """
        eps = np.finfo(float).eps
        x1 = self.copcls.invfcn(-eps, *self.params)
        x2 = self.copcls.invfcn(eps, *self.params)
        np.testing.assert_allclose(x1, x2, atol=3 * eps, rtol=3 * eps)

    @pytest.fixture
    def nsamples(self):
        return 10000

    @pytest.fixture
    def significance(self):
        return 0.0001

    def test_correct_distribution(self, rng, nsamples, significance):
        in_shape = self.copcls.input_shape(*self.params)
        out_shape = self.copcls.output_shape(*self.params)
        samples_norm = rng.standard_normal((nsamples,) + in_shape)
        samples = self.copcls.invfcn(samples_norm, *self.params)
        assert samples.shape == (nsamples,) + out_shape
        if out_shape or in_shape:
            refsamples = self.recrvs(0)(nsamples, rng)
            self.compare_samples(samples, refsamples, rng, significance)
        else:
            test = stats.ks_1samp(samples, self.cdf())
            assert test.pvalue >= significance

    @mark.parametrize('level', [0, 1, 2])
    def test_recursive(self, name, level, rng, nsamples, significance):
        variables = self.copcls(name, *self.convert_recparams(level))
        samples_norm = rng.standard_normal((nsamples,) + variables.shape)
        bd = gvar.BufferDict({f'{name}(x)': samples_norm})
        samples = bd['x']
        refsamples = self.recrvs(level)(nsamples, rng)
        self.compare_samples(samples, refsamples, rng, significance)

    def assert_notnan(self, x):
        assert np.all(~np.isnan(x)), np.sum(np.isnan(x))

    def compare_samples(self, samples, refsamples, rng, significance):
        self.assert_notnan(samples)
        self.assert_notnan(refsamples)
        assert refsamples.shape == samples.shape
        out_shape = samples.shape[1:]
        samples = self.trim(samples, len(out_shape))
        refsamples = self.trim(refsamples, len(out_shape))
        
        for i in np.ndindex(*out_shape):
            i = (...,) + i
            test = stats.ks_2samp(samples[i], refsamples[i])
            assert test.pvalue >= significance
        
        direction = rng.standard_normal(out_shape)
        samples_1d = np.tensordot(samples, direction, direction.ndim)
        refsamples_1d = np.tensordot(refsamples, direction, direction.ndim)
        test = stats.ks_2samp(samples_1d, refsamples_1d)
        assert test.pvalue >= significance

class TestBeta(CopulaFactoryTestBase):
    params = 1.2, 2.3
    recparams = 'invgamma', 'halfcauchy'

class TestDirichlet(CopulaFactoryTestBase):
    params = 1.2, [1, 4, 3]
    recparams = 'gamma', 5
    accurate_range = 1e-15, np.inf
        # TODO remove after using loggamma
    
    def scipy_params(alpha, n):
        alpha = np.asarray(alpha)
        if isinstance(n, numbers.Integral):
            n = np.ones(n)
        n = np.asarray(n)
        return alpha[..., None] * n / n.sum(axis=-1, keepdims=True),

    def rvs(cls, alpha, n, size=(), random_state=None):
        alpha, = cls.scipy_params(alpha, n)
        rng = np.random.default_rng(random_state)
        shape = np.broadcast_shapes(alpha.shape[:-1], size) + alpha.shape[-1:]
        alpha = np.broadcast_to(alpha, shape)
        return cls.dirichlet_rvs(alpha, rng)

    @staticmethod
    @functools.partial(np.vectorize, excluded=(1, 2), signature='(n)->(n)')
    def dirichlet_rvs(alpha, rng):
        """ neither numpy's nor scipy's rvs support broadcasting on alpha """
        return rng.dirichlet(alpha)

class TestGamma(CopulaFactoryTestBase):
    params = 1.2, 2.3
    recparams = 'invgamma', 'halfnorm'
    scipy_params = lambda alpha, beta: (alpha, 0, 1 / beta)

class TestHalfCauchy(CopulaFactoryTestBase):
    params = 0.7,
    recparams = 'invgamma',
    scipy_params = lambda gamma: (0, gamma)

class TestHalfNorm(CopulaFactoryTestBase):
    params = 1.3,
    recparams = 'invgamma',
    scipy_params = lambda sigma: (0, sigma)

class TestInvGamma(CopulaFactoryTestBase):
    params = 1.2, 2.3
    recparams = 'invgamma', 'halfnorm'
    scipy_params = lambda alpha, beta: (alpha, 0, beta)

class TestUniform(CopulaFactoryTestBase):
    params = -0.5, 2
    recparams = -1, 'uniform'
    scipy_params = lambda a, b: (a, b - a)

def test_invgamma_divergence():
    y = lgp.copula.invgamma.invfcn(10., 1, 1)
    assert np.isfinite(y)

@mark.parametrize('distr', ['gamma', 'invgamma'])
@mark.parametrize('x64', [False, True])
def test_gamma_zero(distr, x64):
    
    test = CopulaFactoryTestBase.testfor[distr]

    # check there's no over/underflow
    if distr == 'gamma':
        y = test.copcls.invfcn(100, *test.params)
        assert y < np.inf
    else:
        y = test.copcls.invfcn(-100, *test.params)
        assert y > 0
    
    # check continuity at asymptotic series switchpoint
    with jaxconfig(jax_enable_x64=x64):
        dtype = [np.float32, np.float64][x64]
        rtol = [2e-4, 1e-5][x64]
        boundary = test.copcls._boundary(dtype(0.))
        y1 = test.copcls.invfcn(boundary * (1 + np.finfo(dtype).eps), *test.params)
        y2 = test.copcls.invfcn(boundary * (1 - np.finfo(dtype).eps), *test.params)
        assert y1 > 0 and y2 > 0
        np.testing.assert_allclose(y1, y2, atol=0, rtol=rtol)
    
    # TODO improve the accuracy
