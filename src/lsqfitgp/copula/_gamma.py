# lsqfitgp/copula/_gamma.py
#
# Copyright (c) 2023, 2024, Giacomo Petrillo
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

"""
JAX-compatible implementation of the gamma and related distributions
"""

import functools

from scipy import special
import jax
from jax.scipy import special as jspecial
from jax import numpy as jnp
import numpy

from .. import _jaxext

def _castto(func, type):
    @functools.wraps(func)
    def newfunc(*args, **kw):
        return func(*args, **kw).astype(type)
    return newfunc

@jax.custom_jvp
def gammainccinv(a, y):
    a = jnp.asarray(a)
    y = jnp.asarray(y)
    dtype = _jaxext.float_type(a.dtype, y.dtype)
    ufunc = _castto(special.gammainccinv, dtype)
    return _jaxext.pure_callback_ufunc(ufunc, dtype, a, y)

dQ_da = _jaxext.elementwise_grad(jspecial.gammaincc, 0)
dQ_dx = _jaxext.elementwise_grad(jspecial.gammaincc, 1)

@gammainccinv.defjvp
def gammainccinv_jvp(primals, tangents):
    a, y = primals
    at, yt = tangents

    x = gammainccinv(a, y)

    dQ_dx_a_x = dQ_dx(a, x)
    dQinv_dy_a_y = 1 / dQ_dx_a_x
    xt = dQinv_dy_a_y * yt

    if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating): # modern jax would be: getattr(at, 'dtype', jnp.float64) != jax.float0
        dQ_da_a_x = dQ_da(a, x)
        dQinv_da_a_y = -dQinv_dy_a_y * dQ_da_a_x
        xt += dQinv_da_a_y * at
    
    return x, xt

@jax.custom_jvp
def gammaincinv(a, y):
    a = jnp.asarray(a)
    y = jnp.asarray(y)
    dtype = _jaxext.float_type(a.dtype, y.dtype)
    ufunc = _castto(special.gammaincinv, dtype)
    return _jaxext.pure_callback_ufunc(ufunc, dtype, a, y)

dP_da = _jaxext.elementwise_grad(jspecial.gammainc, 0)
dP_dx = _jaxext.elementwise_grad(jspecial.gammainc, 1)

@gammaincinv.defjvp
def gammaincinv_jvp(primals, tangents):
    a, y = primals
    at, yt = tangents
    
    x = gammaincinv(a, y)

    dP_dx_a_x = dP_dx(a, x)
    dPinv_dy_a_y = 1 / dP_dx_a_x
    xt = dPinv_dy_a_y * yt   
    
    if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating): # modern jax would be: getattr(at, 'dtype', jnp.float64) != jax.float0
        dP_da_a_x = dP_da(a, x)
        dPinv_da_a_y = -dPinv_dy_a_y * dP_da_a_x
        xt += dPinv_da_a_y * at
    
    return x, xt

def _gammaisf_normcdf_large_neg_x(x, a):
    logphi = lambda x: -1/2 * jnp.log(2 * jnp.pi) - 1/2 * jnp.square(x) - jnp.log(-x)
    logq = logphi(x)
    loggammaa = jspecial.gammaln(a)
    f = lambda y: (a - 1) * jnp.log(y) - y - loggammaa - logq
    f1 = lambda y: (a - 1) / y - 1
    y0 = -logq
    y1 = y0 - ((a - 1) * jnp.log(y0) - loggammaa) / ((a - 1) / y0 - 1)
    return y1

    # TODO Improve the accuracy. I tried adding one Newton step more, but it
    # does not improve the accuracy. I probably have to add terms to the
    # approximations of Phi and Q. I could try first special.erfcx for Phi.

    # x -> -∞,  q -> 0+,  y -> ∞
    # q = Φ(x) ≈ -1/√2π exp(-x²/2)/x
    # q = Q(a, y) ≈ y^(a-1) e^-y / Γ(a)
    # gamma.isf(q, a) = Q⁻¹(a, q)
    # log q = -1/2 log 2π - x²/2 - log(-x)              (1)
    # log q = (a - 1) log y - y - log Γ(a)              (2)
    # f(y) = (a - 1) log y - y - log Γ(a) - log(q)
    #      = 0     by (2)
    # f'(y) = (a - 1) / y - 1
    # y_0 = -log q    by considering y -> ∞
    # y_1 = y_0 - f(y_0) / f'(y_0)    Newton step

def _loggammaisf_normcdf_large_neg_x(x, a):
    logphi = lambda x: -1/2 * jnp.log(2 * jnp.pi) - 1/2 * jnp.square(x) - jnp.log(-x)
    logq = logphi(x)
    loggammaa = jspecial.gammaln(a)
    g = lambda logy: (a - 1) * logy - jnp.exp(logy) - loggammaa - logq
    g1 = lambda logy: (a - 1) - jnp.exp(logy)
    logy0 = jnp.log(-logq)
    logy1 = logy0 - ((a - 1) * logy0 - loggammaa) / ((a - 1) + logq)
    return logy1

class gamma:
    
    @staticmethod
    def ppf(q, a):
        return gammaincinv(a, q)

    @staticmethod
    def isf(q, a):
        return gammainccinv(a, q)

class invgamma:
    
    @staticmethod
    def ppf(q, a):
        return 1 / gammainccinv(a, q)

    @staticmethod
    def isf(q, a):
        return 1 / gammaincinv(a, q)

    @staticmethod
    def logpdf(x, a):
        return -(a + 1) * jnp.log(x) - 1 / x - jspecial.gammaln(a)

    @staticmethod
    def cdf(x, a):
        return jspecial.gammaincc(a, 1 / x)

class loggamma:

    @staticmethod
    def ppf(q, c):
        # scipy code:
        # g = sc.gammaincinv(c, q)
        # return _lazywhere(g < _XMIN, (g, q, c),
        #                   lambda g, q, c: (np.log(q) + sc.gammaln(c+1))/c,
        #                   f2=lambda g, q, c: np.log(g))
        g = gammaincinv(c, q)
        return jnp.where(g < jnp.finfo(g.dtype).tiny,
            (jnp.log(q) + jspecial.gammaln(c + 1)) / c,
            jnp.log(g),
        )

    @staticmethod
    def isf(q, c):
        # scipy code:
        # g = sc.gammainccinv(c, q)
        # return _lazywhere(g < _XMIN, (g, q, c),
        #                   lambda g, q, c: (np.log1p(-q) + sc.gammaln(c+1))/c,
        #                   f2=lambda g, q, c: np.log(g))
        g = gammainccinv(c, q)
        return jnp.where(g < jnp.finfo(g.dtype).tiny,
            (jnp.log1p(-q) + jspecial.gammaln(c + 1)) / c,
            jnp.log(g),
        )
