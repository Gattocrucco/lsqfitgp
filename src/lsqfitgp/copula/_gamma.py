# lsqfitgp/copula/_gamma.py
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

"""
JAX-compatible implementation of the gamma and inverse gamma distributions
"""

import functools

from scipy import special
import jax
from jax.scipy import special as jspecial
from jax import numpy as jnp
import numpy

from .. import _patch_jax

def _castto(func, type):
    @functools.wraps(func)
    def newfunc(*args, **kw):
        return func(*args, **kw).astype(type)
    return newfunc

@jax.custom_jvp
def gammainccinv(a, y):
    a = jnp.asarray(a)
    y = jnp.asarray(y)
    class ResultDummy:
        dtype = jnp.result_type(a.dtype, y.dtype)
        shape = jnp.broadcast_shapes(a.shape, y.shape)
    ufunc = _castto(special.gammainccinv, ResultDummy.dtype)
    return jax.pure_callback(ufunc, ResultDummy, a, y, vectorized=True)

dQ_da = _patch_jax.elementwise_grad(jspecial.gammaincc, 0)
dQ_dx = _patch_jax.elementwise_grad(jspecial.gammaincc, 1)

@gammainccinv.defjvp
def gammainccinv_jvp(primals, tangents):
    a, y = primals
    at, yt = tangents
    x = gammainccinv(a, y)
    a = jnp.asarray(a)
    a = a.astype(_patch_jax.float_type(a))
    # convert a to float to avoid dQ_da complaining even when we are not
    # actually deriving w.r.t. a
    dQ_da_a_x = dQ_da(a, x)
    dQ_dx_a_x = dQ_dx(a, x)
    dQinv_dy_a_y = 1 / dQ_dx_a_x
    dQinv_da_a_y = -dQinv_dy_a_y * dQ_da_a_x
    return x, dQinv_da_a_y * at + dQinv_dy_a_y * yt

@jax.custom_jvp
def gammaincinv(a, y):
    a = jnp.asarray(a)
    y = jnp.asarray(y)
    class ResultDummy:
        dtype = jnp.result_type(a.dtype, y.dtype)
        shape = jnp.broadcast_shapes(a.shape, y.shape)
    ufunc = _castto(special.gammaincinv, ResultDummy.dtype)
    return jax.pure_callback(ufunc, ResultDummy, a, y, vectorized=True)

dP_da = _patch_jax.elementwise_grad(jspecial.gammainc, 0)
dP_dx = _patch_jax.elementwise_grad(jspecial.gammainc, 1)

@gammaincinv.defjvp
def gammaincinv_jvp(primals, tangents):
    a, y = primals
    at, yt = tangents
    x = gammaincinv(a, y)
    a = jnp.asarray(a)
    a = a.astype(_patch_jax.float_type(a))
    # convert a to float to avoid dP_da complaining even when we are not
    # actually deriving w.r.t. a
    dP_da_a_x = dP_da(a, x)
    dP_dx_a_x = dP_dx(a, x)
    dPinv_dy_a_y = 1 / dP_dx_a_x
    dPinv_da_a_y = -dPinv_dy_a_y * dP_da_a_x
    return x, dPinv_da_a_y * at + dPinv_dy_a_y * yt

def _invgammappf_normcdf_large_neg_x_1(x):
    return 1 / (1/2 * jnp.log(2 * jnp.pi) + 1/2 * jnp.square(x) + jnp.log(-x))
    # Φ(x) ≈ -1/√2π exp(-x²/2)/x
    # Q(a, x) ≈ x^(a-1) e^-x / Γ(a)
    # invgamma.ppf(x, a) = 1 / Q⁻¹(a, x)

def _invgammappf_normcdf_large_neg_x(x, a):
    x0 = 1/2 * jnp.log(2 * jnp.pi) + 1/2 * jnp.square(x) + jnp.log(-x)
    x1 = x0 - (-(a - 1) * jnp.log(x0) + jspecial.gammaln(a)) / (1 - (a - 1) / x0)
    return 1 / x1
    # compared to _1, this adds one newton step for Q⁻¹(a, x)

class gamma:
    
    @staticmethod
    def ppf(q, a, scale=1):
        return scale * gammaincinv(a, q)

    @staticmethod
    def isf(q, a, scale=1):
        return scale * gammainccinv(a, q)

class invgamma:
    
    @staticmethod
    def ppf(q, a, scale=1):
        return scale / gammainccinv(a, q)

    @staticmethod
    def isf(q, a, scale=1):
        return scale / gammaincinv(a, q)

    @staticmethod
    def logpdf(x, a, scale=1):
        x = x / scale
        return -jnp.log(scale) - (a + 1) * jnp.log(x) - 1 / x - jspecial.gammaln(a)

    @staticmethod
    def cdf(x, a, scale=1):
        return jspecial.gammaincc(a, scale / x)
