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

import numpy as np
import gvar
import jax
import pytest

import lsqfitgp as lgp

class JaxConfig:
    
    def __init__(self, **opts):
        self.opts = opts
    
    def __enter__(self):
        self.prev = {k: jax.config.read(k) for k in self.opts}
        for k, v in self.opts.items():
            jax.config.update(k, v)
    
    def __exit__(self, *_):
        for k, v in self.prev.items():
            jax.config.update(k, v)

class CopulaBaseTest:
    
    name = 'lgp.copula.estdistr'
    
    def __init_subclass__(cls, params):
        assert cls.__name__.startswith('Test')
        cls.copcls = getattr(lgp.copula, cls.__name__[4:].lower())
        cls.params = params
    
    # don't make this a staticmethod because it wouldn't be callable at class
    # definition time in old python versions
    def clean(meth):
        @functools.wraps(meth)
        def newmeth(self, *args, **kw):
            if gvar.BufferDict.has_distribution(self.name):
                gvar.BufferDict.del_distribution(self.name)
            return meth(self, *args, **kw)
        return newmeth
    
    @clean
    def test_invfcn(self):
        self.copcls(self.name, *self.params)
        invfcn = gvar.BufferDict.invfcn[self.name]
        x = gvar.gvar(0.1, 1.2)
        y = invfcn(x)
        deriv = jax.grad(invfcn)(x.mean)
        ymean = invfcn(x.mean)
        ysdev = x.sdev * deriv
        np.testing.assert_allclose(y.mean, ymean, rtol=1e-6)
        np.testing.assert_allclose(y.sdev, ysdev, rtol=1e-6)
    
    @clean
    def test_overwrite(self):
        gvar.BufferDict.add_distribution(self.name, gvar.exp)
        with pytest.raises(ValueError):
            self.copcls(self.name, *self.params)
        gvar.BufferDict.del_distribution(self.name)
        self.copcls(self.name, *self.params)
        self.copcls(self.name, *self.params)
        with pytest.raises(ValueError):
            self.copcls(self.name, *(np.nan,) * len(self.params))
    
    @clean
    def test_bufferdict_gvar(self):
        key = f'{self.name}(x)'
        b = gvar.BufferDict({
            key: self.copcls(self.name, *self.params)
        })
        x = b['x']
        xmean = self.copcls.invfcn(b[key].mean, *self.params)
        xsdev = jax.grad(self.copcls.invfcn)(b[key].mean, *self.params)
        np.testing.assert_allclose(x.mean, xmean, rtol=1e-6)
        np.testing.assert_allclose(x.sdev, xsdev, rtol=1e-6)

    @clean
    def test_bufferdict(self):
        key = f'{self.name}(x)'
        self.copcls(self.name, *self.params)
        b = gvar.BufferDict({
            key: 0
        })
        x = b['x']
        x2 = self.copcls.invfcn(b[key], *self.params)
        np.testing.assert_allclose(x, x2, rtol=1e-6)

class TestInvGamma(CopulaBaseTest, params=(1.2, 2.3)): pass
class TestBeta(CopulaBaseTest, params=(1.2, 2.3)): pass

def test_invgamma_divergence():
    y = lgp.copula.invgamma.invfcn(10., 1, 1)
    assert np.isfinite(y)

def test_invgamma_zero():
    params = (1.2, 2.3)
    y = lgp.copula.invgamma.invfcn(-100, *params)
    assert y > 0
    with JaxConfig(jax_enable_x64=False):
        y1 = lgp.copula.invgamma.invfcn(-12 * (1 + np.finfo(np.float32).eps), *params)
        y2 = lgp.copula.invgamma.invfcn(-12 * (1 - np.finfo(np.float32).eps), *params)
        assert y1 > 0 and y2 > 0
        np.testing.assert_allclose(y1, y2, atol=0, rtol=2e-4)
    with JaxConfig(jax_enable_x64=True):
        y1 = lgp.copula.invgamma.invfcn(-37 * (1 + np.finfo(np.float64).eps), *params)
        y2 = lgp.copula.invgamma.invfcn(-37 * (1 - np.finfo(np.float64).eps), *params)
        assert y1 > 0 and y2 > 0
        np.testing.assert_allclose(y1, y2, atol=0, rtol=1e-5)

    # TODO improve the accuracy
