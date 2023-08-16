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

import numpy as np
import gvar
import jax
import pytest
from pytest import mark
from scipy import stats

import lsqfitgp as lgp

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

class CopulaBaseTest:
    """
    Base class for tests of a CopulaFactory subclass
    """
    
    @pytest.fixture
    def name(self, request):
        return request.node.nodeid

    testfor = {}
    
    def __init_subclass__(cls, *, params, scipy_params=None, recparams=None):
        assert cls.__name__.startswith('Test')
        if scipy_params is None:
            scipy_params = lambda *p: p
        attrs = dict(
            copcls=getattr(lgp.copula, cls.__name__[4:].lower()),
            params=params,
            scipy_params=staticmethod(scipy_params),
            recparams=recparams,
        )
        for k, v in attrs.items():
            if not hasattr(cls, k):
                setattr(cls, k, v)
        __class__.testfor[cls.copcls.__name__] = cls

    def cdf(self):
        distrname = self.copcls.__name__
        distr = getattr(stats, distrname)
        params = self.scipy_params(*self.params)
        return distr(*params).cdf

    @classmethod
    def recrvs(cls, level):
        
        distrname = cls.copcls.__name__
        distr = getattr(stats, distrname)
        
        def rvs(size):
            if level > 0:
                params = []
                for param in cls.recparams:
                    if isinstance(param, str):
                        rvs = cls.testfor[param].recrvs(level - 1)
                        param = rvs(size)
                    params.append(param)
            else:
                params = cls.params 
            params = cls.scipy_params(*params)
            return distr.rvs(*params, size=size)
        
        return rvs

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
        
    def test_invfcn_errorprop(self, name):
        self.copcls(name, *self.params)
        invfcn = gvar.BufferDict.invfcn[name]
        x = gvar.gvar(0.1, 1.2)
        y = invfcn(x)
        deriv = jax.grad(invfcn)(x.mean)
        ymean = invfcn(x.mean)
        ysdev = x.sdev * deriv
        np.testing.assert_allclose(y.mean, ymean, rtol=1e-6)
        np.testing.assert_allclose(y.sdev, ysdev, rtol=1e-6)
    
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
        xmean = self.copcls.invfcn(b[key].mean, *self.params)
        xsdev = jax.grad(self.copcls.invfcn)(b[key].mean, *self.params)
        np.testing.assert_allclose(x.mean, xmean, rtol=1e-6)
        np.testing.assert_allclose(x.sdev, xsdev, rtol=1e-6)

    def test_bufferdict(self, name):
        key = f'{name}(x)'
        self.copcls(name, *self.params)
        b = gvar.BufferDict({
            key: 0
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

    def test_correct_distribution(self, rng):
        samples = rng.standard_normal(10000)
        samples = self.copcls.invfcn(samples, *self.params)
        test = stats.ks_1samp(samples, self.cdf())
        assert test.pvalue >= 0.001

    @mark.parametrize('level', [0, 1, 2])
    def test_recursive(self, name, level, rng):
        variables = self.copcls(name, *self.convert_recparams(level))
        samples = rng.standard_normal((10000,) + variables.shape)
        bd = gvar.BufferDict({f'{name}(x)': samples})
        samples = bd['x']
        refsamples = self.recrvs(level)(samples.size)
        test = stats.ks_2samp(samples, refsamples)
        assert test.pvalue >= 0.001

class TestBeta(CopulaBaseTest,
    params=(1.2, 2.3),
    recparams=('invgamma', 'halfcauchy'),
): pass

class TestHalfCauchy(CopulaBaseTest,
    params=(0.7,),
    scipy_params=(lambda gamma: (0, gamma)),
    recparams=('invgamma',),
): pass

class TestHalfNorm(CopulaBaseTest,
    params=(1.3,),
    scipy_params=(lambda sigma: (0, sigma)),
    recparams=('invgamma',)
): pass

class TestInvGamma(CopulaBaseTest,
    params=(1.2, 2.3),
    scipy_params=(lambda alpha, beta: (alpha, 0, beta)),
    recparams=('invgamma', 'halfnorm')
): pass

class TestUniform(CopulaBaseTest,
    params=(-0.5, 2),
    scipy_params=(lambda a, b: (a, b - a)),
    recparams=(-1, 'uniform'),
): pass

def test_invgamma_divergence():
    y = lgp.copula.invgamma.invfcn(10., 1, 1)
    assert np.isfinite(y)

def test_invgamma_zero():
    params = (1.2, 2.3)
    y = lgp.copula.invgamma.invfcn(-100, *params)
    assert y > 0
    with jaxconfig(jax_enable_x64=False):
        y1 = lgp.copula.invgamma.invfcn(-12 * (1 + np.finfo(np.float32).eps), *params)
        y2 = lgp.copula.invgamma.invfcn(-12 * (1 - np.finfo(np.float32).eps), *params)
        assert y1 > 0 and y2 > 0
        np.testing.assert_allclose(y1, y2, atol=0, rtol=2e-4)
    with jaxconfig(jax_enable_x64=True):
        y1 = lgp.copula.invgamma.invfcn(-37 * (1 + np.finfo(np.float64).eps), *params)
        y2 = lgp.copula.invgamma.invfcn(-37 * (1 - np.finfo(np.float64).eps), *params)
        assert y1 > 0 and y2 > 0
        np.testing.assert_allclose(y1, y2, atol=0, rtol=1e-5)

    # TODO improve the accuracy
