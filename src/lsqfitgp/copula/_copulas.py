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

""" predefined distributions """

import functools
import collections

from jax.scipy import special as jspecial
import jax
from jax import numpy as jnp

from .. import _jaxext
from .. import _array
from . import _beta, _gamma
from . import _distr

def _normcdf(x):
    x = jnp.asarray(x)
    x = x.astype(_jaxext.float_type(x))
    return jspecial.ndtr(x)

    # In jax < 0.?.?, jax.scipy.stats.norm.sf is implemented as 1 - cdf(x)
    # instead of cdf(-x), defeating the purpose of numerical accuracy. Use
    # _normcdf(-x) instead. See https://github.com/google/jax/issues/17199

class beta(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    
    @staticmethod
    def invfcn(x, alpha, beta):
        return _beta.beta.ppf(_normcdf(x), a=alpha, b=beta)

class dirichlet(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Dirichlet_distribution
    """

    signature = '(n),(n)->(n)'
    
    @classmethod
    def invfcn(cls, x, alpha):
        lny = loggamma.invfcn(x, alpha)
        norm = jspecial.logsumexp(lny, axis=-1, keepdims=True)
        return jnp.exp(lny - norm)

    # @classmethod
    # def _invfcn_tiny_alpha(cls, x, alpha):
    #     q = _normcdf(x)
    #     lnq = jnp.log(q)
    #     lny = lnq / alpha
    #     lnnorm = jspecial.logsumexp(lny, axis=-1, keepdims=True)
    #     return jnp.exp(lny - lnnorm)

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

class gamma(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    @staticmethod
    def _boundary(x):
        return {
            jnp.dtype(jnp.float32): 12,
            jnp.dtype(jnp.float64): 37,
        }[x.dtype]

    @classmethod
    def invfcn(cls, x, alpha, beta):
        x = jnp.asarray(x)
        x = x.astype(_jaxext.float_type(x))
        boundary = cls._boundary(x)
        return _piecewise_multiarg(
            [x < 0, x < boundary, x >= boundary],
                # TODO the x < 0 case is probably never considered because
                # piecewise evaluates from the right and x < boundary is
                # satisfied too. Why are the tests not uncovering the
                # inaccuracy? First find whether it's accurate the same or if
                # the tests are lacking, then correct the conditionals.
            [
                lambda x, a: _gamma.gamma.ppf(_normcdf(x), a),
                lambda x, a: _gamma.gamma.isf(_normcdf(-x), a),
                lambda x, a: _gamma._gammaisf_normcdf_large_neg_x(-x, a),
            ],
            x, alpha,
        ) / beta

class loggamma(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Gamma_distribution, `scipy.stats.loggamma`

    This is the distribution of the logarithm of a Gamma variable. The naming
    convention is the opposite of lognorm, which is the distribution of the
    exponential of a Normal variable.
    """
    
    @staticmethod
    def _boundary(x):
        return gamma._boundary(x)

    @classmethod
    def invfcn(cls, x, alpha):
        x = jnp.asarray(x)
        x = x.astype(_jaxext.float_type(x))
        boundary = cls._boundary(x)
        return _piecewise_multiarg(
            [x < 0, x < boundary, x >= boundary],
            [
                lambda x, alpha: _gamma.loggamma.ppf(_normcdf(x), alpha),
                lambda x, alpha: _gamma.loggamma.isf(_normcdf(-x), alpha),
                lambda x, alpha: _gamma._loggammaisf_normcdf_large_neg_x(-x, alpha),
            ],
            x, alpha,
        )

    # TODO scipy.stats.gamma has inaccurate logsf instead of using loggamma.sf,
    # open an issue

class invgamma(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """
    
    @staticmethod
    def _boundary(x):
        return -gamma._boundary(x)

    @classmethod
    def invfcn(cls, x, alpha, beta):
        x = jnp.asarray(x)
        x = x.astype(_jaxext.float_type(x))
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

class halfcauchy(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Cauchy_distribution, `scipy.stats.halfcauchy`
    """
    
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

class halfnorm(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Half-normal_distribution
    """
    
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

class uniform(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Continuous_uniform_distribution
    """
    
    @staticmethod
    def invfcn(x, a, b):
        return a + (b - a) * _normcdf(x)

class lognorm(_distr.Distr):
    """
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    
    @staticmethod
    def invfcn(x, mu, sigma):
        return jnp.exp(mu + sigma * x)
