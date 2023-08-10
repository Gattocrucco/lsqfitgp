# lsqfitgp/copula/__init__.py
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

""" Gaussian copulas for gvar """

import abc
import functools

import gvar
from jax.scipy import special as jspecial
import numpy
import jax
from jax import numpy as jnp

from .. import _patch_gvar
from .. import _patch_jax
from . import _beta, _invgamma

def normcdf(x):
    x = jnp.asarray(x)
    x = x.astype(_patch_jax.float_type(x))
    return jspecial.ndtr(x)

class CopulaFactory(metaclass=abc.ABCMeta):
    """
    Abstract base class for copula factories.

    Class to represent transformations in `gvar.BufferDict`, i.e., Gaussian
    copulas.

    Parameters
    ----------
    name : str
        The unique name assigned to the transformation using
        `gvar.BufferDict.add_distribution`.
    *params : scalars
        The parameters of the distribution.

    Examples
    --------
    >>> copula = gvar.BufferDict({
    ...     'transf(x)': lgp.copula.beta('transf', 1, 1),
    ...     'transf_2(y)': lgp.copula.beta('transf_2', 3, 5),
    ... })
    >>> copula['x']
    0.50(40)
    >>> copula['y']
    0.36(18)

    See also
    --------
    gvar.BufferDict.uniform
    """
    
    def __init_subclass__(cls, **_):
        cls.params = {}
    
    @staticmethod
    @abc.abstractmethod
    def invfcn(x, *params):
        """
        :math:`F^{-1}(\\Phi(x))`, i.e., maps a Normal variable to a variable
        with the desired marginal distribution. Jax-traceable.
        """
        pass

    def __new__(cls, name, *params):
        if gvar.BufferDict.has_distribution(name):
            invfcn = gvar.BufferDict.invfcn[name]
            if getattr(invfcn, '_CopulaFactory_subclass_', None) is not cls:
                raise ValueError(f'distribution {name} already defined')
            if cls.params[name] != params:
                raise ValueError(f'Attempt to overwrite existing {cls.__name__} distribution with name {name}')
                # cls.params is not updated by
                # gvar.BufferDict.del_distribution, but it is not a problem
        else:
            invfcn = lambda x: cls.invfcn(x, *params)
            invfcn = _patch_gvar.add_gvar_support(invfcn)
            invfcn._CopulaFactory_subclass_ = cls
            gvar.BufferDict.add_distribution(name, invfcn)
            cls.params[name] = params
        return gvar.gvar(0, 1)

    @classmethod
    def _assert(cls, cond, message='invalid parameters'):
        """ method to raise an exception for invalid distribution parameters """
        if not cond:
            raise ValueError(f'{cls.__name__} distribution: {message}')
    
class beta(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    
    def __new__(cls, name, alpha, beta):
        cls._assert(alpha > 0 and beta > 0)
        return super().__new__(cls, name, alpha, beta)
    
    @staticmethod
    def invfcn(x, alpha, beta):
        return _beta.beta.ppf(normcdf(x), a=alpha, b=beta)

class invgamma(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """
    
    def __new__(cls, name, alpha, beta):
        cls._assert(alpha > 0 and beta > 0)
        return super().__new__(cls, name, alpha, beta)

    @staticmethod
    def _invgamma_ppf_norm_cdf_large_negative_x_1(x):
        return 1 / (1/2 * jnp.log(2 * jnp.pi) + 1/2 * jnp.square(x) + jnp.log(-x))
        # Φ(x) ≈ -1/√2π exp(-x²/2)/x
        # Q(a, x) ≈ x^(a-1) e^-x / Γ(a)
        # invgamma.ppf(x, a) = 1 / Q⁻¹(a, x)
    
    @staticmethod
    def _invgamma_ppf_norm_cdf_large_negative_x_2(x, a):
        x0 = 1/2 * jnp.log(2 * jnp.pi) + 1/2 * jnp.square(x) + jnp.log(-x)
        x1 = x0 - (-(a - 1) * jnp.log(x0) + jspecial.gammaln(a)) / (1 - (a - 1) / x0)
        return 1 / x1
        # compared to _1, this adds one newton step for Q⁻¹(a, x)

    @classmethod
    def invfcn(cls, x, alpha, beta):
        x = jnp.asarray(x)
        x = x.astype(_patch_jax.float_type(x))
        boundary = 37 if x.dtype == jnp.float64 else 12
        return beta * jnp.piecewise(x, 
            [x < -boundary, x > 0],
            [
                lambda x: cls._invgamma_ppf_norm_cdf_large_negative_x_2(x, a=alpha),
                lambda x: _invgamma.invgamma.isf(normcdf(-x), a=alpha),
                lambda x: _invgamma.invgamma.ppf(normcdf(x), a=alpha),
            ],
        )

class uniform(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Continuous_uniform_distribution
    """
    
    def __new__(cls, name, a, b):
        cls._assert(a <= b)
        return super().__new__(cls, name, a, b)
    
    @staticmethod
    def invfcn(x, a, b):
        return a + (b - a) * normcdf(x)

class halfcauchy(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Cauchy_distribution

    `scipy.stats.halfcauchy`
    """
    
    def __new__(cls, name, gamma):
        cls._assert(gamma > 0)
        return super().__new__(cls, name, gamma)

    @staticmethod
    def _ppf(p):
        return jnp.tan(jnp.pi * p / 2)
    
    @staticmethod
    def _isf(p):
        return 1 / jnp.tan(jnp.pi * p / 2)
    
    @classmethod
    def invfcn(cls, x, gamma):
        return gamma * jnp.where(x < 0,
            cls._ppf(normcdf(x)),
            cls._isf(normcdf(-x)),
        )

# TODO jax.scipy.stats.norm.sf is implemented as 1 - cdf(x) instead
# of cdf(-x), defeating the purpose of numerical accuracy, open an
# issue and PR to fix it

class halfnorm(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Half-normal_distribution
    """
    
    def __new__(cls, name, sigma):
        cls._assert(sigma > 0)
        return super().__new__(cls, name, sigma)

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
            cls._ppf(normcdf(x)),
            cls._isf(normcdf(-x)),
        )
