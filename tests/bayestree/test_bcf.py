# lsqfitgp/tests/test_bcf.py
#
# Copyright (c) 2024, Giacomo Petrillo
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

""" test lgp.bayestree.bcf """

import numpy as np
import jax
from jax import random
from jax import numpy as jnp
import statsmodels.api as sm
import pytest

import lsqfitgp as lgp

from .. import util

def gen_X(key, p, n):
    return random.uniform(key, (p, n), float, -2, 2)

T = 2 # base period

def ps(X): # treatment probability
    minps = 0.05
    ps = 0.5 + 0.5 * jnp.sum(jnp.cos(2 * jnp.pi / (T / 2) * X), axis=0)
    return jnp.clip(ps, minps, 1 - minps)

def gen_z(key, X):
    return random.bernoulli(key, ps(X))

def f(X, z): # outcome mean
    mu = jnp.sum(jnp.cos(2 * jnp.pi / T * X), axis=0)
    tau = jnp.sum(jnp.sin(2 * jnp.pi / T * X), axis=0)
    return mu + z * tau

def gen_y(key, X, z):
    sigma = 0.1
    return f(X, z) + sigma * random.normal(key, z.shape)

def estimate_ps(X, z):
    """ use a GLM """
    z = np.array(z)
    X = np.concatenate([X, np.ones((1, z.size))]).T
    model = sm.GLM(z, X, family=sm.families.Binomial())
    result = model.fit()
    return result.predict()

@pytest.fixture
def n():
    return 101

@pytest.fixture
def p():
    return 11

@pytest.fixture
def X(n, p, key):
    key = random.fold_in(key, 0xd9b0963d)
    return gen_X(key, p, n)

@pytest.fixture
def z(X, key):
    key = random.fold_in(key, 0x1a7c4e8d)
    return gen_z(key, X)

@pytest.fixture
def y(z, X, key):
    key = random.fold_in(key, 0x1391bc96)
    return gen_y(key, X, z)

@pytest.fixture
def pihat(X, z):
    return estimate_ps(X, z)

@pytest.fixture(params=[1, 2, 3])
def kw(request, X):
    variant = request.param
    
    if variant == 1:
        return dict()

    if variant == 2:
        def gpaux(hp, gp):
            kernel = lgp.ExpQuad(scale=hp['scale'], dim='aux')
            return gp.defproc('aux', kernel)
        return dict(
            x_tau=X.T ** 2,
            include_pi='both',
            marginalize_mean=False,
            gpaux=gpaux,
            x_aux=jnp.abs(X.T),
            otherhp=lgp.copula.makedict(dict(scale=lgp.copula.invgamma(1, 1))),
        )

    if variant == 3:
        return dict(include_pi='tau')

def getkw(kw, key):
    return kw.get(key, lgp.bayestree.bcf.__init__.__kwdefaults__[key])

def test_scale_shift(y, z, X, pihat, key, kw):
    
    kw.update(z=z, x_mu=X.T, pihat=pihat, transf='standardize')
    bcf1 = lgp.bayestree.bcf(y=y, **kw)

    offset = 0.4703189
    scale = 0.5294714
    tilde_y = offset + y * scale
    bcf2 = lgp.bayestree.bcf(y=tilde_y, **kw)

    seed = random.bits(key)
    rng1 = np.random.default_rng(seed.item())
    rng2 = np.random.default_rng(seed.item())
    predkw = dict(transformed=False, samples=1, error=True)
    y1, = bcf1.pred(**predkw, rng=rng1)
    y2, = bcf2.pred(**predkw, rng=rng2)
    util.assert_allclose(y2, offset + y1 * scale, rtol=1e-7, atol=1e-7)

    eta1 = bcf1.from_data(y)
    eta2 = bcf2.from_data(tilde_y)
    util.assert_allclose(eta2, eta1, rtol=1e-13)

    if getkw(kw, 'marginalize_mean'):
        assert bcf1.m == 0
    else:
        assert hasattr(bcf1.m, 'sdev')

def test_to_from_data(y, z, X, pihat, kw, key):
    kw.update(
        y=y,
        z=z,
        x_mu=X.T,
        pihat=pihat,
        transf=['standardize', 'yeojohnson'],
    )
    bcf = lgp.bayestree.bcf(**kw)

    eta = bcf.from_data(y)
    y2 = bcf.to_data(eta)
    util.assert_allclose(y, y2, rtol=1e-15, atol=1e-15)

    seed = random.bits(key)
    rng1 = np.random.default_rng(seed.item())
    rng2 = np.random.default_rng(seed.item())
    eta = bcf.from_data(y, hp='sample', rng=rng1)
    y2 = bcf.to_data(eta, hp='sample', rng=rng2)
    util.assert_allclose(y, y2, rtol=1e-15, atol=1e-15)

def test_yeojohnson():
    """ check the Yeo-Johnson transformation """
    testinput = np.linspace(-2, 2, 100)
    lamda = 1.5
    mod = lgp.bayestree._bcf
    np.testing.assert_allclose(
        mod.yeojohnson_inverse(mod.yeojohnson(testinput, lamda), lamda),
        testinput, atol=0, rtol=1e-14)

# TODO
# - tests of basic statistical ability
#     - true ATE/SATT vs. infer it with bcf
#     - predictions on a test set
#     What margins should I use? I could look at the result, decide it makes sense,
#     then set margins based on that to look for regressions.
# - process index spec for gpaux
#     How do I access it? => Define a noop proclintransf that inspects x
# - x_tau changes the result
# - a test with no transf
