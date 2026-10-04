# lsqfitgp/tests/bayestree/test_bcf.py
#
# Copyright (c) 2024, 2026, Giacomo Petrillo
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

"""Test `lgp.bayestree.bcf`."""

import numpy as np
import pytest
import statsmodels.api as sm
from jax import numpy as jnp
from jax import random

import lsqfitgp as lgp
from tests import util


def gen_X(key, p, n):
    """Generate random covariates uniform in [-2, 2] with shape (p, n)."""
    return random.uniform(key, (p, n), float, -2, 2)


T = 2  # base period


def ps(X):  # treatment probability
    """Return the true propensity score, clipped to [0.05, 0.95]."""
    minps = 0.05
    ps = 0.5 + 0.5 * jnp.sum(jnp.cos(2 * jnp.pi / (T / 2) * X), axis=0)
    return jnp.clip(ps, minps, 1 - minps)


def gen_z(key, X):
    """Generate random treatment assignments."""
    return random.bernoulli(key, ps(X))


def f(X, z):  # outcome mean
    """Return the true outcome mean."""
    mu = jnp.sum(jnp.cos(2 * jnp.pi / T * X), axis=0)
    tau = jnp.sum(jnp.sin(2 * jnp.pi / T * X), axis=0)
    return mu + z * tau


def gen_y(key, X, z):
    """Generate outcomes as `f(X, z)` plus Gaussian noise."""
    sigma = 0.1
    return f(X, z) + sigma * random.normal(key, z.shape)


def estimate_ps(X, z):
    """Estimate the propensity score using a GLM."""
    z = np.array(z)
    X = np.concatenate([X, np.ones((1, z.size))]).T
    model = sm.GLM(z, X, family=sm.families.Binomial())
    result = model.fit()
    return result.predict()


@pytest.fixture
def n():
    """Return the number of units."""
    return 101


@pytest.fixture
def p():
    """Return the number of covariates."""
    return 11


@pytest.fixture
def X(n, p, key):
    """Return random covariates."""
    key = random.fold_in(key, 0xD9B0963D)
    return gen_X(key, p, n)


@pytest.fixture
def z(X, key):
    """Return random treatment assignments."""
    key = random.fold_in(key, 0x1A7C4E8D)
    return gen_z(key, X)


@pytest.fixture
def y(z, X, key):
    """Return random outcomes."""
    key = random.fold_in(key, 0x1391BC96)
    return gen_y(key, X, z)


@pytest.fixture
def pihat(X, z):
    """Return the propensity score estimated with a GLM."""
    return estimate_ps(X, z)


@pytest.fixture(params=[1, 2, 3])
def kw(request, X):
    """Return keyword arguments for `bcf`, in three variants."""
    variant = request.param

    if variant == 1:
        return dict()

    elif variant == 2:

        def gpaux(hp, gp):
            kernel = lgp.ExpQuad(scale=hp['scale'], dim='aux')
            return gp.defproc('aux', kernel)

        return dict(
            x_tau=X.T**2,
            include_pi='both',
            marginalize_mean=False,
            gpaux=gpaux,
            x_aux=jnp.abs(X.T),
            otherhp=lgp.copula.makedict(dict(scale=lgp.copula.invgamma(1, 1))),
        )

    else:
        assert variant == 3
        return dict(include_pi='tau')


def getkw(kw, key):
    """Return the value of a `bcf` argument in `kw`, or its default."""
    return kw.get(key, lgp.bayestree.bcf.__init__.__kwdefaults__[key])


def test_scale_shift(y, z, X, pihat, key, kw):
    """Check that `bcf` is equivariant to an affine transformation of the outcome."""
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
    (y1,) = bcf1.pred(**predkw, rng=rng1)
    (y2,) = bcf2.pred(**predkw, rng=rng2)
    util.assert_allclose(y2, offset + y1 * scale, rtol=1e-7, atol=1e-7)

    eta1 = bcf1.from_data(y)
    eta2 = bcf2.from_data(tilde_y)
    util.assert_allclose(eta2, eta1, rtol=1e-13)

    if getkw(kw, 'marginalize_mean'):
        assert bcf1.m == 0
    else:
        assert hasattr(bcf1.m, 'sdev')


def test_to_from_data(y, z, X, pihat, kw, key):
    """Check that `to_data` inverts `from_data`, also with sampled hyperparameters."""
    kw.update(y=y, z=z, x_mu=X.T, pihat=pihat, transf=['standardize', 'yeojohnson'])
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


def test_transf_list():
    """Check that each transformation in a list uses its own hyperparameter."""
    y = np.linspace(-1, 1, 5)
    from_data, to_data, _, hypers = lgp.bayestree.bcf._get_transf(
        None, transf=['yeojohnson', 'yeojohnson'], y=y, weights=None
    )
    assert len(hypers) == 2
    hp = {'transf0_lambda_yj': 0.5, 'transf1_lambda_yj': 1.5}
    mod = lgp.bayestree._bcf
    eta = mod.yeojohnson(mod.yeojohnson(y, 0.5), 1.5)
    util.assert_allclose(from_data(hp, y), eta, rtol=1e-15)
    util.assert_allclose(to_data(hp, eta), y, rtol=1e-14, atol=1e-15)


def test_yeojohnson():
    """Check the Yeo-Johnson transformation."""
    testinput = np.linspace(-2, 2, 100)
    lamda = 1.5
    mod = lgp.bayestree._bcf
    np.testing.assert_allclose(
        mod.yeojohnson_inverse(mod.yeojohnson(testinput, lamda), lamda),
        testinput,
        atol=0,
        rtol=1e-14,
    )
