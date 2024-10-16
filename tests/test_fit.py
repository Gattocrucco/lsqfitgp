# lsqfitgp/tests/test_fit.py
#
# Copyright (c) 2020, 2022, 2023, 2024, Giacomo Petrillo
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

import numpy as np
from jax import numpy as jnp
import gvar
from scipy import stats
import pytest
from pytest import mark

import lsqfitgp as lgp
from . import util

FITKW = dict()

def flat(g):
    """convert dictionary or array to 1D array"""
    if hasattr(g, 'buf'):
        return g.buf
    elif hasattr(g, 'keys'):
        return gvar.BufferDict(g).buf
    else:
        return np.reshape(g, -1)

def quad(A, v):
    """compute v.T @ A^-1 @ v"""
    w, u = np.linalg.eigh(A)
    utv = u.T @ v
    eps = len(A) * 1e-12 * np.max(w)
    return (utv.T / np.maximum(w, eps)) @ utv
    # TODO maybe low-rank is better?

def chisq_test(g, alpha):
    """chisquare test on g being 0"""
    g = flat(g)
    mean = gvar.mean(g)
    cov = gvar.evalcov(g)
    q = quad(cov, mean)
    n = len(mean)
    assert stats.chi2(n).sf(q) > alpha / 2
    assert stats.chi2(n).cdf(q) > alpha / 2

def check_fit(hyperprior, gpfactory, alpha=1e-5):
    """do a fit with empbayes_fit and check the fitted hyperparameters
    are compatible with the ones used to generate the data"""
    
    # generate hyperparameters
    truehp = gvar.sample(hyperprior)
    
    # generate data
    gp = gpfactory(truehp)
    data = gvar.sample(gp.prior())
        
    # run fit
    fit = lgp.empbayes_fit(hyperprior, gpfactory, data, raises=False, **FITKW)
    
    # check fit result against hyperparameters
    chisq_test(fit.p - truehp, alpha)

@mark.xfail(reason='I guess Laplace approximation bad for this model. Seen passing.')
def test_period():
    hp = {
        'log(scale)': gvar.log(gvar.gvar(1, 0.1))
    }
    x = np.linspace(0, 6, 10)
    def gpfactory(hp):
        return lgp.GP(lgp.Periodic(scale=hp['scale'])).addx(x, 'x')
    check_fit(hp, gpfactory)

def test_scale():
    hp = {
        'log(scale)': gvar.log(gvar.gvar(3, 0.2))
    }
    x = np.linspace(0, 2 * np.pi * 5, 20)
    def gpfactory(hp):
        return lgp.GP(lgp.ExpQuad(scale=hp['scale'])).addx(x, 'x')
    check_fit(hp, gpfactory, alpha=1e-7)

def test_sdev():
    hp = {
        'log(sdev)': gvar.log(gvar.gvar(1, 1))
    }
    x = np.linspace(0, 5, 10)
    def gpfactory(hp):
        return lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2).addx(x, 'x')
    check_fit(hp, gpfactory, alpha=1e-8)
    
    # TODO once I've seen the chi2 check fail with sf(q) = 1e-8. Is this
    # a minimization problem or the posterior distribution of log(sdev) which
    # has a very heavy tail? The distribution of sdev**2 should be a scaled
    # inverse chisquared, so very heavy tailed indeed

def test_scale_sdev():
    hp = {
        'log(scale)': gvar.log(gvar.gvar(3, 0.2)),
        'log(sdev)': gvar.log(gvar.gvar(1, 1)),
    }
    x = np.linspace(0, 2 * np.pi * 5, 20)
    def gpfactory(hp):
        kernel = hp['sdev'] ** 2 * lgp.ExpQuad(scale=hp['scale'])
        return lgp.GP(kernel).addx(x, 'x')
    check_fit(hp, gpfactory, alpha=1e-7)

def test_flat_scalar():
    hp = gvar.BufferDict({
        'log(sdev)': gvar.log(gvar.gvar(1, 1)),
    })
    x = np.linspace(0, 5, 10)
    
    def gpfactory1(hp):
        return lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2).addx(x, 'x')
    def gpfactory2(hp):
        return lgp.GP(lgp.ExpQuad() * jnp.exp(hp[0]) ** 2).addx(x, 'x')
    def gpfactory3(hp):
        return lgp.GP(lgp.ExpQuad() * jnp.exp(hp) ** 2).addx(x, 'x')
    
    truehp = gvar.sample(hp)
    truegp = gpfactory1(truehp)
    trueprior = truegp.prior()
    data = gvar.sample(trueprior)
    
    kw = dict(raises=False, **FITKW)
    fit1 = lgp.empbayes_fit(hp, gpfactory1, data, **kw)
    fit2 = lgp.empbayes_fit(hp.buf, gpfactory2, data, **kw)
    fit3 = lgp.empbayes_fit(hp.buf[0], gpfactory3, data, **kw)
    
    util.assert_similar_gvars(fit1.p.buf[0], fit2.p[0], fit3.p)

def test_method():
    
    hp = gvar.BufferDict({
        'log(sdev)': gvar.log(gvar.gvar(1, 1))
    })
    x = np.linspace(0, 5, 10)
    def gpfactory(hp):
        return lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2).addx(x, 'x')
    truehp = gvar.sample(hp)
    truegp = gpfactory(truehp)
    trueprior = truegp.prior()
    data_fixed = gvar.sample(trueprior)
    def data_variable(hp):
        return {k: v + hp['log(sdev)'] for k, v in data_fixed.items()}
    
    for data in [data_fixed, data_variable]:
        fits = []
        kws = [
            dict(method='nograd', minkw=dict(options=dict(xatol=1e-6))),
            dict(method='gradient'),
            dict(method='fisher'),
            dict(method='fisher', minkw=dict(method='trust-constr')),
        ]
        if not sys.version.startswith('3.8'):
            kws.append(dict(method='gradient', minkw=dict(method='l-bfgs-b')))

        for kw in kws:
            kwargs = dict(data=data)
            kwargs.update(kw)
            kwargs.setdefault('minkw', {}).update(x0=truehp.buf)
            fit = lgp.empbayes_fit(hp, gpfactory, **kwargs, **FITKW)
            fits.append(fit)
        p = fits[0].minresult.x
        for fit in fits[1:]:
            util.assert_allclose(fit.minresult.x, p, atol=1e-5)

def test_checks():
    with pytest.raises(KeyError):
        lgp.empbayes_fit(gvar.gvar(0, 1), lambda: None, lambda: None, method='cippa', **FITKW)
    with pytest.raises(RuntimeError) as err:
        def makegp(x):
            return lgp.GP(lgp.ExpQuad()).addx(x, 'x')
        lgp.empbayes_fit(gvar.gvar(0, 1), makegp, {'x': 0.}, minkw=dict(options=dict(maxiter=0)), **FITKW)
    assert 'minimization failed: ' in str(err.value)

def test_int_data():
    def makegp(x):
        return lgp.GP(lgp.ExpQuad()).addx(x, 'x')
    lgp.empbayes_fit(gvar.gvar(0, 1), makegp, {'x': 0}, **FITKW)

def test_data_formats():
    """ check that presenting data in different formats does not change the
    result """
    
    hp = gvar.BufferDict({
        'log(sdev)': gvar.log(gvar.gvar(1, 1))
    })
    x = np.linspace(0, 5, 10)
    def gpfactory(hp):
        return lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2).addx(x, 'x')
    truehp = gvar.sample(hp)
    truegp = gpfactory(truehp)
    trueprior = truegp.prior()
    
    def makeerr(bd, err):
        return gvar.BufferDict(bd, buf=np.full_like(bd.buf, err))
    
    data_noerr = gvar.sample(trueprior)
    error = makeerr(data_noerr, 0.1)
    zeroerror = makeerr(data_noerr, 0)
    zerocov = gvar.evalcov(gvar.gvar(data_noerr, zeroerror))
    data_err = gvar.make_fake_data(gvar.gvar(data_noerr, error))
    
    datas = [
        [
            data_noerr,
            gvar.gvar(data_noerr),
            (data_noerr,),
            (data_noerr, zerocov),
            lambda _: data_noerr,
            lambda _: gvar.gvar(data_noerr),
            lambda _: (data_noerr,),
            lambda _: (data_noerr, zerocov),
        ],
        [
            data_err,
            (data_err,),
            (gvar.mean(data_err), gvar.evalcov(data_err)),
            lambda _: data_err,
            lambda _: (data_err,),
            lambda _: (gvar.mean(data_err), gvar.evalcov(data_err)),
        ],
    ]
    
    for datasets in datas:
        fits = []
        for data in datasets:
            fit = lgp.empbayes_fit(hp, gpfactory, data, **FITKW)
            fits.append(fit)

        p = fits[0].minresult.x
        for fit in fits[1:]:
            util.assert_allclose(fit.minresult.x, p, atol=1e-6)

def test_loss_zero():
    """ check that adding a zero loss function does not change the result """
    
    hp = gvar.BufferDict({
        'log(sdev)': gvar.log(gvar.gvar(1, 1))
    })
    x = np.linspace(0, 5, 10)
    def gpfactory(hp):
        return lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2).addx(x, 'x')
    
    truehp = gvar.sample(hp)
    truegp = gpfactory(truehp)
    trueprior = truegp.prior()
    data = gvar.sample(trueprior)

    common_args = dict(
        hyperprior=hp,
        gpfactory=gpfactory,
        data=data,
        minkw=dict(method='bfgs'),
            # for backward compatibility when I change the default to l-bfgs-b
    )
    varying_args = [
        dict(),
        dict(additional_loss=lambda _: 0.),
    ]

    fits = []
    for kw in varying_args:
        fits.append(lgp.empbayes_fit(**common_args, **kw, **FITKW))

    f0 = fits[0].minresult.fun
    for fit in fits[1:]:
        util.assert_allclose(fit.minresult.fun, f0)

def test_loss_offset():
    """ check that adding a constant loss function changes the function value
    but not the location of the minimum """
    
    hp = gvar.BufferDict({
        'log(sdev)': gvar.log(gvar.gvar(1, 1))
    })
    x = np.linspace(0, 5, 10)
    def gpfactory(hp):
        return lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2).addx(x, 'x')
    
    truehp = gvar.sample(hp)
    truegp = gpfactory(truehp)
    trueprior = truegp.prior()
    data = gvar.sample(trueprior)

    common_args = dict(
        hyperprior=hp,
        gpfactory=gpfactory,
        data=data,
        minkw=dict(method='bfgs'),
            # for backward compatibility when I change the default to l-bfgs-b
    )
    
    offset = 100.
    fit0 = lgp.empbayes_fit(**common_args, **FITKW)
    fit1 = lgp.empbayes_fit(**common_args, additional_loss=lambda _: offset, **FITKW)

    util.assert_allclose(fit0.minresult.fun + offset, fit1.minresult.fun, rtol=1e-13)
    util.assert_allclose(fit0.minresult.x, fit1.minresult.x, rtol=1e-12)

def test_loss_shrinkage():
    """ check that a loss with minimum in a different position moves the
    result towards there """
    
    hp = gvar.BufferDict({
        'log(sdev)': gvar.log(gvar.gvar(1, 1))
    })
    x = np.linspace(0, 5, 10)
    def gpfactory(hp):
        return lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2).addx(x, 'x')
    
    truehp = gvar.sample(hp)
    truegp = gpfactory(truehp)
    trueprior = truegp.prior()
    data = gvar.sample(trueprior)

    common_args = dict(
        hyperprior=hp,
        gpfactory=gpfactory,
        data=data,
        minkw=dict(method='bfgs'),
            # for backward compatibility when I change the default to l-bfgs-b
    )

    fit0 = lgp.empbayes_fit(**common_args, **FITKW)
    loc = (fit0.pmean['log(sdev)'] + 10).item()
        # .item() is for a shape bug, presumed due to old gvar-old numpy interaction
    def loss(hp):
        return (hp['log(sdev)'] - loc) ** 2
    fit1 = lgp.empbayes_fit(**common_args, additional_loss=loss, **FITKW)

    assert fit1.minresult.fun > fit0.minresult.fun
    dist = lambda x, y: np.linalg.norm(x - y)
    assert dist(fit1.minresult.x, loc) < dist(fit0.minresult.x, loc)

def test_loss_fisher():
    """ check that using fisher with a user loss raises """
    
    hp = gvar.BufferDict({
        'log(sdev)': gvar.log(gvar.gvar(1, 1))
    })
    x = np.linspace(0, 5, 10)
    def gpfactory(hp):
        return lgp.GP(lgp.ExpQuad() * hp['sdev'] ** 2).addx(x, 'x')
    
    truehp = gvar.sample(hp)
    truegp = gpfactory(truehp)
    trueprior = truegp.prior()
    data = gvar.sample(trueprior)

    with pytest.raises(NotImplementedError):
        fit = lgp.empbayes_fit(hp, gpfactory, data, method='fisher', additional_loss=lambda _: 0., **FITKW)
