# lsqfitgp/copula/_copulas.py
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

import functools
import collections
import numbers

from jax.scipy import special as jspecial
import jax
from jax import numpy as jnp

from .. import _patch_jax
from .. import _array
from . import _beta, _gamma
from ._base import CopulaFactory

def _normcdf(x):
    x = jnp.asarray(x)
    x = x.astype(_patch_jax.float_type(x))
    return jspecial.ndtr(x)

    # TODO jax.scipy.stats.norm.sf is implemented as 1 - cdf(x) instead
    # of cdf(-x), defeating the purpose of numerical accuracy, open an
    # issue and PR to fix it

class beta(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    
    def __new__(cls, name, alpha, beta, **kw):
        return super().__new__(cls, name, alpha, beta, **kw)
    
    @staticmethod
    def invfcn(x, alpha, beta):
        return _beta.beta.ppf(_normcdf(x), a=alpha, b=beta)

class dirichlet(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Dirichlet_distribution
    """
    
    def __new__(cls, name, alpha, n, **kw):
        return super().__new__(cls, name, alpha, n, **kw)
    
    @classmethod
    def invfcn(cls, x, alpha, n):
        alpha = jnp.asarray(alpha)
        if isinstance(n, numbers.Integral):
            n = jnp.ones(n)
        else:
            n = jnp.asarray(n)
        alpha = alpha[..., None] * n / n.sum(axis=-1, keepdims=True)
        lny = loggamma.invfcn(x, alpha)
        norm = jspecial.logsumexp(lny, axis=-1, keepdims=True)
        return jnp.exp(lny - norm)

    @classmethod
    def _invfcn_tiny_alpha(cls, x, alpha):
        q = _normcdf(x)
        lnq = jnp.log(q)
        lny = lnq / alpha
        lnnorm = jspecial.logsumexp(lny, axis=-1, keepdims=True)
        return jnp.exp(lny - lnnorm)

        # For a -> 0:
        #
        # gamma.cdf(x, a) = P(a, x)
        #                 = gamma(a, x) / Gamma(a)
        #                 = int_0^x dt e^-t t^(a - 1) / (1 / a)
        #                 = a [t^a / a]_0^x
        #                 = a x^a / a
        #                 = x^a
        #
        # gamma.ppf(q, a) = P^-1(a, q)
        #                 = q^1/a

    @staticmethod
    def input_shape(alpha, n):
        if isinstance(n, numbers.Integral):
            return (n,)
        else:
            return _array.asarray(n).shape[-1:]

    @classmethod
    def output_shape(cls, alpha, n):
        return cls.input_shape(alpha, n)

class gamma(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    def __new__(cls, name, alpha, beta, **kw):
        return super().__new__(cls, name, alpha, beta, **kw)

    @staticmethod
    def _boundary(x):
        return {
            jnp.dtype(jnp.float32): 12,
            jnp.dtype(jnp.float64): 37,
        }[x.dtype]

    @classmethod
    def invfcn(cls, x, alpha, beta):
        x = jnp.asarray(x)
        x = x.astype(_patch_jax.float_type(x))
        boundary = cls._boundary(x)
        return _piecewise_multiarg(
            [x < 0, x < boundary, x >= boundary],
            [
                lambda x, a: _gamma.gamma.ppf(_normcdf(x), a),
                lambda x, a: _gamma.gamma.isf(_normcdf(-x), a),
                lambda x, a: _gamma._gammaisf_normcdf_large_neg_x(-x, a),
            ],
            x, alpha,
        ) / beta

class loggamma(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Gamma_distribution, `scipy.stats.loggamma`
    """
    
    def __new__(cls, name, c, **kw):
        return super().__new__(cls, name, c, **kw)

    @staticmethod
    def _boundary(x):
        return gamma._boundary(x)

    @classmethod
    def invfcn(cls, x, c):
        x = jnp.asarray(x)
        x = x.astype(_patch_jax.float_type(x))
        boundary = cls._boundary(x)
        return _piecewise_multiarg(
            [x < 0, x < boundary, x >= boundary],
            [
                lambda x, c: _gamma.loggamma.ppf(_normcdf(x), c),
                lambda x, c: _gamma.loggamma.isf(_normcdf(-x), c),
                lambda x, c: _gamma._loggammaisf_normcdf_large_neg_x(-x, c),
            ],
            x, c,
        )

class invgamma(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """
    
    def __new__(cls, name, alpha, beta, **kw):
        return super().__new__(cls, name, alpha, beta, **kw)

    @staticmethod
    def _boundary(x):
        return -gamma._boundary(x)

    @classmethod
    def invfcn(cls, x, alpha, beta):
        x = jnp.asarray(x)
        x = x.astype(_patch_jax.float_type(x))
        boundary = cls._boundary(x)
        return beta * _piecewise_multiarg(
            [x < boundary, x < 0, x >= 0],
            [
                lambda x, a: 1 / _gamma._gammaisf_normcdf_large_neg_x(x, a),
                lambda x, a: _gamma.invgamma.ppf(_normcdf(x), a),
                lambda x, a: _gamma.invgamma.isf(_normcdf(-x), a),
            ],
            x, alpha,
        )

def _piecewise_multiarg(conds, functions, *operands):
    conds = jnp.stack(conds, axis=-1)
    index = jnp.argmax(conds, axis=-1)
    return _vectorized_switch(index, functions, *operands)

@functools.partial(jnp.vectorize, excluded=(1,))
def _vectorized_switch(index, branches, *operands):
    return jax.lax.switch(index, branches, *operands)

class halfcauchy(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Cauchy_distribution, `scipy.stats.halfcauchy`
    """
    
    def __new__(cls, name, gamma, **kw):
        return super().__new__(cls, name, gamma, **kw)

    @staticmethod
    def _ppf(p):
        return jnp.tan(jnp.pi * p / 2)
    
    @staticmethod
    def _isf(p):
        return 1 / jnp.tan(jnp.pi * p / 2)
    
    @classmethod
    def invfcn(cls, x, gamma):
        return gamma * jnp.where(x < 0,
            cls._ppf(_normcdf(x)),
            cls._isf(_normcdf(-x)),
        )

class halfnorm(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Half-normal_distribution
    """
    
    def __new__(cls, name, sigma, **kw):
        return super().__new__(cls, name, sigma, **kw)

    @staticmethod
    def _ppf(p):
        # F(x) = 2 Φ(x) - 1
        # -->  F⁻¹(p) = Φ⁻¹((1 + p) / 2)
        return jspecial.ndtri((1 + p) / 2)

    @staticmethod
    def _isf(p):
        # Φ(-x) = 1 - Φ(x)
        # -->  Φ⁻¹(1 - p) = -Φ⁻¹(p)
        # S(x) = 1 - F(x)
        # -->  S⁻¹(p) = F⁻¹(1 - p)
        #             = Φ⁻¹((2 - p) / 2)
        #             = Φ⁻¹(1 - p / 2)
        #             = -Φ⁻¹(p / 2)
        return -jspecial.ndtri(p / 2)

    @classmethod
    def invfcn(cls, x, sigma):
        return sigma * jnp.where(x < 0,
            cls._ppf(_normcdf(x)),
            cls._isf(_normcdf(-x)),
        )

class uniform(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Continuous_uniform_distribution
    """
    
    def __new__(cls, name, a, b, **kw):
        return super().__new__(cls, name, a, b, **kw)
    
    @staticmethod
    def invfcn(x, a, b):
        return a + (b - a) * _normcdf(x)
