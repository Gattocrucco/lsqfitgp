# lsqfitgp/tests/test_copulas.py
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

""" Test the predefined distributions """

import functools
import contextlib
import string

import numpy as np
import gvar
import jax
from jax import numpy as jnp
import pytest
from pytest import mark
from scipy import stats, special

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

class DistrTestBase:
    """
    Base class for tests of a Distr subclass
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
                    param = test.copcls(*test.convert_recparams(level - 1))
                params.append(param)
        else:
            params = cls.params
        return params
        
    @pytest.fixture
    def name(self, request):
        return request.node.nodeid

    def test_invfcn_errorprop(self, name, rng):
        variables = self.copcls(*self.params, name=name)
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
    
    def test_partial_invfcn_gvar_vectorized(self, name, rng):

        distr = self.copcls(*self.params)

        shape = (13,)
        x = gvar.gvar(
            rng.standard_normal(shape + distr.in_shape),
            rng.gamma(10, 1/10, shape + distr.in_shape),
        )
        
        y = distr.partial_invfcn(x)
        assert y.shape == shape + distr.shape
        
        ymean = distr.partial_invfcn(gvar.mean(x))
        assert ymean.shape == shape + distr.shape
        deriv = jax.vmap(jax.jacfwd(distr.partial_invfcn))(gvar.mean(x))
        assert deriv.shape == shape + distr.shape + distr.in_shape
        
        ii = string.ascii_lowercase[:len(distr.in_shape)]
        io1 = string.ascii_uppercase[:len(distr.shape)]
        io2 = string.ascii_uppercase[len(distr.shape):2 * len(distr.shape)]
        ycov = np.einsum(f'...{io1}{ii}, ...{ii}, ...{io2}{ii} -> ...{io1}{io2}', deriv, gvar.var(x), deriv)

        for i in np.ndindex(*shape):
            util.assert_close_matrices(gvar.mean(y[i]), ymean[i], rtol=1e-6)
            util.assert_close_matrices(gvar.evalcov(y[i]).reshape(2 * y[i].shape), ycov[i], rtol=1e-6)

    def test_overwrite(self, name):
        gvar.BufferDict.add_distribution(name, gvar.exp)
        with pytest.raises(ValueError):
            self.copcls(*self.params, name=name)
        gvar.BufferDict.del_distribution(name)
        self.copcls(*self.params, name=name)
        self.copcls(*self.params, name=name)

    def test_bufferdict_gvar(self, name):
        key = f'{name}(x)'
        b = gvar.BufferDict({
            key: self.copcls(*self.params, name=name)
        })
        x = b['x']
        mean = gvar.mean(b[key])
        xmean = self.copcls.invfcn(mean, *self.array_params)
        jac = jax.jacfwd(self.copcls.invfcn)(mean, *self.array_params)
        xcov = np.dot(jac * gvar.var(b[key]), jac.T)
        util.assert_close_matrices(gvar.mean(x), xmean, rtol=1e-6)
        util.assert_close_matrices(gvar.evalcov(x).reshape(2 * x.shape), xcov, rtol=1e-6)

    def test_bufferdict(self, name):
        key = f'{name}(x)'
        variables = self.copcls(*self.params, name=name)
        b = gvar.BufferDict({
            key: np.zeros_like(variables, float)
        })
        x = b['x']
        x2 = self.copcls.invfcn(b[key], *self.array_params)
        util.assert_allclose(x, x2, rtol=1e-6)

    def test_continuity_zero(self, name):
        """ check that invfcn is continuous in zero, since it is a common
        cutpoint to switch from ppf(cdf(·)) to isf(sf(·)) """
        eps = np.finfo(float).eps
        x1 = self.copcls.invfcn(-eps, *self.array_params)
        x2 = self.copcls.invfcn(eps, *self.array_params)
        util.assert_allclose(x1, x2, atol=4 * eps, rtol=3 * eps)

    @pytest.fixture
    def nsamples(self):
        return 10000

    @pytest.fixture
    def significance(self):
        return 0.0001

    @functools.cached_property
    def array_params(self):
        return tuple(map(np.asarray, self.params))

    def test_correct_distribution(self, rng, nsamples, significance):
        sig = self.copcls.signature.eval(None, *self.array_params)
        in_shape = sig.in_shapes[0]
        out_shape, = sig.out_shapes
        samples_norm = rng.standard_normal((nsamples,) + in_shape)
        samples = self.copcls.invfcn(samples_norm, *self.array_params)
        assert samples.shape == (nsamples,) + out_shape
        if out_shape or in_shape:
            refsamples = self.recrvs(0)(nsamples, rng)
            self.compare_samples(samples, refsamples, rng, significance)
        else:
            test = stats.ks_1samp(samples, self.cdf())
            assert test.pvalue >= significance

    @mark.parametrize('level', [0, 1, 2])
    def test_recursive(self, name, level, rng, nsamples, significance):
        variables = self.copcls(*self.convert_recparams(level), name=name)
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

class TestBeta(DistrTestBase):
    params = 1.2, 2.3
    recparams = 'invgamma', 'halfcauchy'

class TestDirichlet(DistrTestBase):
    params = 1.2, [1, 4, 3]
    recparams = 'gamma', [1, 1, 1, 1, 1]
    
    def scipy_params(alpha, n):
        alpha = np.asarray(alpha)
        n = np.asarray(n)
        return alpha[..., None] * n / n.sum(axis=-1, keepdims=True),

    def rvs(cls, alpha, n, size=(), random_state=None):
        alpha, = cls.scipy_params(alpha, n)
        rng = np.random.default_rng(random_state)
        shape = np.broadcast_shapes(alpha.shape[:-1], size) + alpha.shape[-1:]
        alpha = np.broadcast_to(alpha, shape)
        return cls.dirichlet_rvs(alpha, rng)

    @staticmethod
    def dirichlet_rvs(alpha, rng):
        lny = TestLogGamma.rvs(alpha, random_state=rng)
        norm = special.logsumexp(lny, axis=-1, keepdims=True)
        return np.exp(lny - norm)

        # numpy.random.Generator.dirichlet is inaccurate for small alpha at x <
        # eps, scipy piggybacks on numpy, see
        # https://github.com/numpy/numpy/issues/24475

class TestGamma(DistrTestBase):
    params = 1.2, 2.3
    recparams = 'invgamma', 'halfnorm'
    scipy_params = lambda alpha, beta: (alpha, 0, 1 / beta)

class TestHalfCauchy(DistrTestBase):
    params = 0.7,
    recparams = 'invgamma',
    scipy_params = lambda gamma: (0, gamma)

class TestHalfNorm(DistrTestBase):
    params = 1.3,
    recparams = 'invgamma',
    scipy_params = lambda sigma: (0, sigma)

class TestInvGamma(DistrTestBase):
    params = 1.2, 2.3
    recparams = 'invgamma', 'halfnorm'
    scipy_params = lambda alpha, beta: (alpha, 0, beta)

class TestLogGamma(DistrTestBase):
    params = 1.2,
    recparams = 'invgamma',

    def rvs(cls, c, size=(), random_state=None):
        # old scipy versions do not handle small c, so I copied the code from
        # a recent version
        shape = getattr(c, 'shape', ())
        size = np.broadcast_shapes(shape, size)
        c = np.broadcast_to(c, size)
        random_state = np.random.default_rng(random_state)
        return (
            np.log(random_state.gamma(c + 1, size=size))
            + np.log(random_state.uniform(size=size)) / c
        )
        # return stats.loggamma.rvs(c, size=size, random_state=random_state)

class TestUniform(DistrTestBase):
    params = -0.5, 2
    recparams = -1, 'uniform'
    scipy_params = lambda a, b: (a, b - a)

def test_invgamma_divergence():
    y = lgp.copula.invgamma.invfcn(10., 1, 1)
    assert np.isfinite(y)

@mark.parametrize('distr', ['gamma', 'invgamma', 'loggamma'])
@mark.parametrize('x64', [False, True])
def test_gamma_asymp(distr, x64):
    
    test = DistrTestBase.testfor[distr]

    # check there's no over/underflow
    if distr == 'gamma':
        y = test.copcls.invfcn(100, *test.params)
        assert y < np.inf
    elif distr == 'invgamma':
        y = test.copcls.invfcn(-100, *test.params)
        assert y > 0
    
    # check continuity at asymptotic series switchpoint
    with jaxconfig(jax_enable_x64=x64):
        dtype = [np.float32, np.float64][x64]
        rtol = [2e-4, 1e-5][x64]
        boundary = test.copcls._boundary(dtype(0.))
        y1 = test.copcls.invfcn(boundary * (1 + np.finfo(dtype).eps), *test.params)
        y2 = test.copcls.invfcn(boundary * (1 - np.finfo(dtype).eps), *test.params)
        assert 0 < y1 < np.inf and 0 < y2 < np.inf
        assert y1 != y2
        np.testing.assert_allclose(y1, y2, atol=0, rtol=rtol)
    
def test_staticdescr_repr():
    
    x = lgp.copula.beta(1, 2)
    assert repr(x._staticdescr) == 'beta(1, 2)'

    x = lgp.copula.beta(1, 2, shape=(3, 4))
    assert repr(x._staticdescr) == 'beta(1, 2, shape=(3, 4))'

    x = lgp.copula.beta(1, lgp.copula.uniform(0.5, 1))
    assert repr(x._staticdescr) == 'beta(1, uniform(0.5, 1))'

    x = lgp.copula.beta([1, 2, 3], 1)
    assert repr(x._staticdescr) == 'beta([1, 2, 3], 1, shape=3)'

    x = lgp.copula.beta([[1, 2], [3, 4]], [1, 2])
    assert repr(x._staticdescr) == 'beta([[1, 2], [3, 4]], [1, 2], shape=(2, 2))'

    x = lgp.copula.dirichlet(lgp.copula.invgamma(1, 1, shape=4), [1, 1, 1])
    assert repr(x._staticdescr) == 'dirichlet(invgamma(1, 1, shape=4), [1, 1, 1], shape=(4, 3))'

    x = lgp.copula.beta(np.array([1, 2.]), 3)
    assert repr(x._staticdescr) == 'beta([1.0, 2.0], 3, shape=2)'

    x = lgp.copula.beta(jnp.array([1, 2.]), 3)
    assert repr(x._staticdescr) == 'beta([1.0, 2.0], 3, shape=2)'

def test_repr():

    x = lgp.copula.beta(1, 2)
    assert repr(x) == 'beta(1, 2)'

    x = lgp.copula.beta(1, 2, shape=(3, 4))
    assert repr(x) == 'beta(1, 2, shape=(3, 4))'

    x = lgp.copula.beta(1, lgp.copula.uniform(0.5, 1))
    assert repr(x) == 'beta(1, uniform(0.5, 1))'

    x = lgp.copula.beta([1, 2, 3], 1)
    assert repr(x) == 'beta([1, 2, 3], 1, shape=3)'

    x = lgp.copula.beta([[1, 2], [3, 4]], [1, 2])
    assert repr(x) == 'beta([[1, 2], [3, 4]], [1, 2], shape=(2, 2))'

    x = lgp.copula.dirichlet(lgp.copula.invgamma(1, 1, shape=4), [1, 1, 1])
    assert repr(x) == 'dirichlet(invgamma(1, 1, shape=4), [1, 1, 1], shape=(4, 3))'

    x = lgp.copula.beta(np.array([1, 2.]), 3)
    assert repr(x) == 'beta(Array[2], 3, shape=2)'

    x = lgp.copula.beta(jnp.ones((2, 3)), 3)
    assert repr(x) == 'beta(Array[2,3], 3, shape=(2, 3))'

def test_repr_recursive():

    x = lgp.copula.beta(1, 2)
    y = lgp.copula.uniform(x, x)
    z = lgp.copula.beta(y, x)
    assert repr(z) == 'beta(uniform(beta(1, 2), <0.0>), <0.0>)'

def test_shared_basic(rng):
    """ test that a shared variable is not duplicated """
    
    x = lgp.copula.invgamma(1, 1)
    y = lgp.copula.halfnorm(x)
    z = lgp.copula.halfcauchy(x)
    q = lgp.copula.uniform(y, z)

    @functools.partial(jnp.vectorize, signature='(4)->()')
    def q_invfcn(n):
        x = lgp.copula.invgamma.invfcn(n[0], 1, 1)
        y = lgp.copula.halfnorm.invfcn(n[1], x)
        z = lgp.copula.halfcauchy.invfcn(n[2], x)
        return lgp.copula.uniform.invfcn(n[3], y, z)

    samples = rng.standard_normal((10000, 4))
    s1 = q.partial_invfcn(samples)
    s2 = q_invfcn(samples)
    util.assert_allclose(s1, s2)

def test_shared_degeneracy(rng):
    """ test that a shared variable is not duplicated, by checking the
    degeneracy in the model """

    x = lgp.copula.loggamma(1)
    y = lgp.copula.uniform(x, x)
    samples = rng.standard_normal((10000, 2))
    s1 = x.partial_invfcn(samples[:, 0])
    s2 = y.partial_invfcn(samples)
    util.assert_allclose(s1, s2)

def test_shared_hierarchy(rng):
    """ test that a shared variable is not duplicated, with complex hierachy """

    x = lgp.copula.invgamma(1, 1)
    y = lgp.copula.halfnorm(x)
    z = lgp.copula.halfcauchy(x)
    q = lgp.copula.uniform(y, z)
    r = lgp.copula.beta(q, x)

    @functools.partial(jnp.vectorize, signature='(5)->()')
    def r_invfcn(n):
        x = lgp.copula.invgamma.invfcn(n[0], 1, 1)
        y = lgp.copula.halfnorm.invfcn(n[1], x)
        z = lgp.copula.halfcauchy.invfcn(n[2], x)
        q = lgp.copula.uniform.invfcn(n[3], y, z)
        return lgp.copula.beta.invfcn(n[4], q, x)

    samples = rng.standard_normal((10000, 5))
    s1 = r.partial_invfcn(samples)
    s2 = r_invfcn(samples)
    util.assert_allclose(s1, s2)

def test_shared_shapes(rng):
    """ test that a shared variable is not duplicated, with complex hierachy
    and shapes """

    a = lgp.copula.invgamma(2, 2, shape=2)     # 2
    x = lgp.copula.invgamma(1, 1, shape=3)     # 3
    y = lgp.copula.halfnorm(x)                 # 3
    z = lgp.copula.halfcauchy(x)               # 3
    q = lgp.copula.uniform(y, z, shape=(2, 3)) # 6
    r = lgp.copula.beta(q, x)                  # 6
    s = lgp.copula.dirichlet(a, r)             # 6

    assert a.in_shape == (2,)
    assert x.in_shape == (3,)
    assert y.in_shape == (3 + 3,)
    assert z.in_shape == (3 + 3,)
    assert q.in_shape == (3 + 3 + 3 + 6,)
    assert r.in_shape == (3 + 3 + 3 + 6 + 6,)
    assert s.in_shape == (2 + 3 + 3 + 3 + 6 + 6 + 6,)

    @functools.partial(jnp.vectorize, signature='(29)->(2,3)')
    def s_invfcn(n):
        a = lgp.copula.invgamma.invfcn(n[0:2], 2, 2)
        x = lgp.copula.invgamma.invfcn(n[2:5], 1, 1)
        y = lgp.copula.halfnorm.invfcn(n[5:8], x)
        z = lgp.copula.halfcauchy.invfcn(n[8:11], x)
        q = lgp.copula.uniform.invfcn(n[11:17].reshape(2, 3), y, z)
        r = lgp.copula.beta.invfcn(n[17:23].reshape(2, 3), q, x)
        return lgp.copula.dirichlet.invfcn(n[23:29].reshape(2, 3), a, r)

    shape = (100,)
    samples = rng.standard_normal(shape + s.in_shape)
    s1 = s_invfcn(samples)
    assert s1.shape == shape + s.shape
    s2 = s.partial_invfcn(samples)
    assert s2.shape == shape + s.shape
    util.assert_allclose(s1, s2)

def test_wrong_nargs():
    with pytest.raises(TypeError):
        lgp.copula.beta(1)
    with pytest.raises(TypeError):
        lgp.copula.beta(1, 2, 3)

def test_staticdescr():
    x1 = lgp.copula.beta(1, 2)
    x2 = x1.__class__(*x1.params)
    assert x1._staticdescr == x2._staticdescr
    
    y1 = lgp.copula.beta(x1, x1)
    y2 = lgp.copula.beta(x2, x2)
    assert y1._staticdescr == y2._staticdescr

    y1 = lgp.copula.beta(x1, x2)
    y2 = lgp.copula.beta(x1, x1)
    assert y1._staticdescr != y2._staticdescr
