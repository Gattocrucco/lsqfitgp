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

# TODO document the conditional copula functionality.

import abc
import functools
import collections
import numbers

import gvar
from jax.scipy import special as jspecial
import numpy
import jax
from jax import numpy as jnp

from .. import _patch_gvar
from .. import _patch_jax
from .. import _array
from . import _beta, _gamma

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

        Normal to desired distribution transformation.

        Maps a (multivariate) Normal variable to a variable with the desired
        marginal distribution. In symbols: :math:`y = F^{-1}(\Phi(x))`. This
        function is a generalized ufunc, jax traceable, and differentiable one
        time.

        Parameters
        ----------
        x : (*broadcast_shape, *in_shape) array or scalar
            The input Normal variable. If multivariate, the variable shape must
            be in the last axes. The calculation is broadcasted over other
            leading axes. Use `input_shape` to determine the required variable
            shape.
        params :
            The parameters of the distribution. Typically, numerical parameters
            may be arrays and are broadcasted with `x`.

        Returns
        -------
        y : (*broadcast_shape, *out_shape) array or scalar
            The output variable with the desired marginal distribution.

        """
        pass

    @classmethod
    def input_shape(cls, *params):
        """
        Return the input shape expected by `invfcn`.

        May depend on the type of the parameters, but not on their values. This
        shape applies only to the class member `invfcn`, not to the one intended
        for `gvar.BufferDict` that is created by the class constructor, which
        always expects either a scalar or vector tail shape.

        Parameters
        ----------
        params :
            The same arguments that would be passed to `invfcn`.

        Returns
        -------
        shape : tuple of int
            The required shape of the last axes of the input array `x` to
            `invfcn`.
        """
        return ()

    @classmethod
    def output_shape(cls, *params):
        return ()

    @classmethod
    def _partial_invfcn(cls, params):
        """
        Partially apply `invfcn` to `params` and return a function that maps
        `nvariables` Normal variables to the desired marginal distribution.
        """

        # expand copulas in parameters
        invfcn, nvariables, out_shape = cls._partial_invfcn_rec(params)

        # determine signature
        out_sig = ','.join(str(l) for l in out_shape)
        if nvariables == 1:
            signature = f'()->({out_sig})'
        else:
            signature = f'({nvariables})->({out_sig})'

        # add gvar support, manage conversion to array
        @functools.partial(_patch_gvar.gvar_gufunc, signature=signature)
        @functools.wraps(cls.invfcn)
        def partial_invfcn(x):
            x = _array.asarray(x)
            if nvariables > 1:
                actual_nvar = x.shape[-1] if x.ndim else 1
                if actual_nvar != nvariables:
                    raise ValueError(
                        f'{cls.__name__} copula expected {nvariables} Normal '
                        f'input variables, found {actual_nvar}'
                    )
            else:
                x = x[..., None]
            y, length = invfcn(x, 0)
            assert length == nvariables
            assert y.shape == x.shape[:-1] + out_shape
            return y

        # mark copula class
        partial_invfcn._CopulaFactory_subclass_ = cls

        # return transformation and size of dict field
        return partial_invfcn, nvariables

    @classmethod
    def _partial_invfcn_rec(cls, params, nvariables=0):

        # process copulas in parameters, converting them to their invfcn
        partialparams = []
        shapeparams = []
        for param in params:
            is_copula = (
                isinstance(param, (tuple, list))
                and param
                and isinstance(param[0], type)
                and issubclass(param[0], __class__)
            )
            if is_copula:
                invfcn, nvariables, out_shape = param[0]._partial_invfcn_rec(
                    param[1:],
                    nvariables,
                )
                partialparams.append(invfcn)
                shapeparams.append(numpy.broadcast_to(numpy.empty(()), out_shape))
            else:
                partialparams.append(param)
                shapeparams.append(param)
        partialparams = tuple(partialparams)

        # statically determine input/output tail shapes
        expected_in_shape = cls.input_shape(*shapeparams)
        expected_in_size = numpy.prod(expected_in_shape, dtype=int)
        expected_out_shape = cls.output_shape(*shapeparams)
        
        # wrap function to close on parameter values, call ancestor copulas,
        # and keep track of array usage
        @functools.wraps(cls.invfcn)
        def newinvfcn(x, i):

            params = []
            for param in partialparams:
                if callable(param):
                    param, i = param(x, i)
                params.append(param)

            in_shape = cls.input_shape(*params)
            assert in_shape == expected_in_shape
            out_shape = cls.output_shape(*params)
            assert out_shape == expected_out_shape

            in_size = numpy.prod(in_shape, dtype=int)
            x = x[..., i:i + in_size]
            broadcast_shape = x.shape[:-1]
            x = x.reshape(broadcast_shape + in_shape)

            y = cls.invfcn(x, *params)
            assert y.shape == broadcast_shape + out_shape
            
            return y, i + in_size

        return newinvfcn, nvariables + expected_in_size, expected_out_shape

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
            invfcn, nvariables = cls._partial_invfcn(params)
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

class dirichlet(CopulaFactory):
    """
    https://en.wikipedia.org/wiki/Dirichlet_distribution
    """
    
    def __new__(cls, name, alpha, n):
        return super().__new__(cls, name, alpha, n)
    
    @staticmethod
    def invfcn(x, alpha, n):
        alpha = jnp.asarray(alpha)
        if isinstance(n, numbers.Integral):
            n = jnp.ones(n)
        else:
            n = jnp.asarray(n)
        n = n / n.sum(axis=-1, keepdims=True)
        alpha = alpha[..., None] * n
        y = jnp.where(x < 0,
            _gamma.gamma.ppf(normcdf(x), a=alpha),
            _gamma.gamma.isf(normcdf(-x), a=alpha),
        )
        return y / y.sum(axis=-1, keepdims=True)

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
    
    def __new__(cls, name, alpha, beta):
        return super().__new__(cls, name, alpha, beta)

    @staticmethod
    def invfcn(x, alpha, beta):
        return jnp.where(x < 0,
            _gamma.gamma.ppf(normcdf(x), a=alpha),
            _gamma.gamma.isf(normcdf(-x), a=alpha),
        ) / beta

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

    @classmethod
    def invfcn(cls, x, alpha, beta):
        x = jnp.asarray(x)
        x = x.astype(_patch_jax.float_type(x))
        boundary = 37 if x.dtype == jnp.float64 else 12
        return beta * _piecewise_multiarg(
            [x < -boundary, x < 0, x >= 0],
            [
                lambda x, a: _gamma._invgammappf_normcdf_large_neg_x(x, a=a),
                lambda x, a: _gamma.invgamma.ppf(normcdf(x), a=a),
                lambda x, a: _gamma.invgamma.isf(normcdf(-x), a=a),
            ],
            x, alpha,
        )
        # gamma does not have the corresponding special case for large positive
        # x. Is it because I have not encountered the problem yet, or because it
        # is less likely to happen?

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
