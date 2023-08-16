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

# TODO add a function copula that somehow allows to define dictionary without
# repeating twice each distribution name, and is also compatible with any normal
# usage => takes in a dict, scans it, when key is string, value is tuple, and
# first element of tuple is callable, then call to get value, rest is args,
# convert the result to BufferDict. Exaple:
#
# lgp.copula.copula({
#     'HC(a)': (lgp.copula.halfcauchy, 1.0),
# })
#
# or maybe I could make it even more automatic, the end user probably does not
# even want to deal with the transformation names or copula objects:
#
# lgp.copula.copula({
#     'a': ('halfcauchy', 1.0),
# }, prefix='ciao_')
#
# -> BufferDict({
#     'ciao_halfcauchy_1.0(a)': lgp.copula.halfcauchy('a', 1.0),
# })
#
# The default prefix would be something like '__copula_'.

import abc
import functools
import collections

import gvar
from jax.scipy import special as jspecial
import numpy
import jax
from jax import numpy as jnp

from .. import _patch_gvar
from .. import _patch_jax
from .. import _array
from . import _beta, _invgamma

def normcdf(x):
    x = jnp.asarray(x)
    x = x.astype(_patch_jax.float_type(x))
    return jspecial.ndtr(x)

# TODO jax.scipy.stats.norm.sf is implemented as 1 - cdf(x) instead
# of cdf(-x), defeating the purpose of numerical accuracy, open an
# issue and PR to fix it

# BufferDict's function mechanism works with functions which modify an array at
# once, possibly changing the shape. I can use that to implement nontrivial
# multivariate distributions. Maybe I should ask Lepage to state this behavior
# in the documentation.

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
    
    @classmethod
    @abc.abstractmethod
    def invfcn(cls, x, *params):
        r"""
        Maps a (multivariate) Normal variable to a variable with the desired
        marginal distribution. This function must be an ufunc, jax traceable,
        and differentiable one time in forward mode. :math:`F^{-1}(\Phi(x))`.
        """
        pass

    @classmethod
    def _partial_invfcn(cls, invfcn, params):
        """
        Partially apply `invfcn` to `params` and return a function that maps
        `nvariables` Normal variables to the desired marginal distribution.
        """

        # expand copulas in parameters
        invfcn, nvariables = cls._partial_invfcn_rec(invfcn, params)

        # add gvar support, manage conversion to array
        @_patch_gvar.add_gvar_support # TODO this was not supposed to work on gufuncs!!
        @functools.wraps(invfcn)
        def partial_invfcn(x):
            x = _array.asarray(x)
            if nvariables > 1:
                assert x.ndim and x.shape[-1] == nvariables
            else:
                x = x[..., None]
            y, length = invfcn(x, 0)
            assert length == nvariables
            assert y.shape == x.shape[:-1]
            return y

        # mark copula class
        partial_invfcn._CopulaFactory_subclass_ = cls

        # return transformation and size of dict field
        return partial_invfcn, nvariables

    @classmethod
    def _partial_invfcn_rec(cls, invfcn, params, nvariables=0):

        # process copulas in parameters
        partialparams = []
        for param in params:
            if isinstance(param, (tuple, list)) and param and issubclass(param[0], __class__):
                param, nvariables = cls._partial_invfcn_rec(param[0].invfcn, param[1:], nvariables)
            partialparams.append(param)
        partialparams = tuple(partialparams)
        
        # wrap function to close on parameter values, call ancestor copulas,
        # and keep track of array usage
        @functools.wraps(invfcn)
        def newinvfcn(x, i):
            params = []
            for param in partialparams:
                if callable(param):
                    param, i = param(x, i)
                params.append(param)
            return invfcn(x[..., i], *params), i + 1
        
        return newinvfcn, nvariables + 1

    _Params = collections.namedtuple('Params', ['params', 'nvariables'])

    def __new__(cls, name, *params):
        if gvar.BufferDict.has_distribution(name):
            invfcn = gvar.BufferDict.invfcn[name]
            if getattr(invfcn, '_CopulaFactory_subclass_', None) is not cls:
                raise ValueError(f'distribution {name} already defined')
            if (existing := cls.params[name]).params != params:
                raise ValueError(f'Attempt to overwrite existing {cls.__name__} distribution with name {name}')
                # cls.params is not updated by
                # gvar.BufferDict.del_distribution, but it is not a problem
            nvariables = existing.nvariables
        else:
            invfcn, nvariables = cls._partial_invfcn(cls.invfcn, params)
            gvar.BufferDict.add_distribution(name, invfcn)
            cls.params[name] = cls._Params(params, nvariables)
        return gvar.gvar(numpy.zeros(nvariables), numpy.ones(nvariables)).squeeze()

class beta(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    
    def __new__(cls, name, alpha, beta):
        return super().__new__(cls, name, alpha, beta)
    
    @staticmethod
    def invfcn(x, alpha, beta):
        return _beta.beta.ppf(normcdf(x), a=alpha, b=beta)

class halfcauchy(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Cauchy_distribution, `scipy.stats.halfcauchy`
    """
    
    def __new__(cls, name, gamma):
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

class halfnorm(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Half-normal_distribution
    """
    
    def __new__(cls, name, sigma):
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

class invgamma(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """
    
    def __new__(cls, name, alpha, beta):
        return super().__new__(cls, name, alpha, beta)

    @staticmethod
    def _ppf_normcdf_large_neg_x_1(x):
        return 1 / (1/2 * jnp.log(2 * jnp.pi) + 1/2 * jnp.square(x) + jnp.log(-x))
        # Φ(x) ≈ -1/√2π exp(-x²/2)/x
        # Q(a, x) ≈ x^(a-1) e^-x / Γ(a)
        # invgamma.ppf(x, a) = 1 / Q⁻¹(a, x)
    
    @staticmethod
    def _ppf_normcdf_large_neg_x(x, a):
        x0 = 1/2 * jnp.log(2 * jnp.pi) + 1/2 * jnp.square(x) + jnp.log(-x)
        x1 = x0 - (-(a - 1) * jnp.log(x0) + jspecial.gammaln(a)) / (1 - (a - 1) / x0)
        return 1 / x1
        # compared to _1, this adds one newton step for Q⁻¹(a, x)

    @classmethod
    def invfcn(cls, x, alpha, beta):
        x = jnp.asarray(x)
        x = x.astype(_patch_jax.float_type(x))
        boundary = 37 if x.dtype == jnp.float64 else 12
        return beta * _piecewise_multiarg(
            [x < -boundary, x < 0, x >= 0],
            [
                lambda x, a: cls._ppf_normcdf_large_neg_x(x, a=a),
                lambda x, a: _invgamma.invgamma.ppf(normcdf(x), a=a),
                lambda x, a: _invgamma.invgamma.isf(normcdf(-x), a=a),
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

class uniform(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Continuous_uniform_distribution
    """
    
    def __new__(cls, name, a, b):
        return super().__new__(cls, name, a, b)
    
    @staticmethod
    def invfcn(x, a, b):
        return a + (b - a) * normcdf(x)
