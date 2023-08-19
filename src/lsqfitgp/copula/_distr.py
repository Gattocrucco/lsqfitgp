# lsqfitgp/copula/_distr.py
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

""" defines Distr """

import abc
import functools
import collections
import numbers
import inspect

import gvar
import numpy
import jax
from jax import numpy as jnp

from .. import _patch_gvar
from .. import _array
from . import _signature

class Distr(metaclass=abc.ABCMeta):
    r"""

    Abstract base class to represent distributions.

    A `Distr` object represents a probability distribution, and provides a
    transformation function from a (multivariate) Normal variable to the target
    random variable, in particular for use with `gvar.BufferDict`.

    Parameters
    ----------
    *params : tuple of scalar, array or Distr
        The parameters of the distribution. If the parameters have leading axes
        other than those required, the distribution is independently repeated
        over those axes. If a parameter is an instance of `Distr` itself, it
        is a random parameter and its distribution is accounted for.
    name : str, optional
        If specified, the distribution is defined for usage with
        `gvar.BufferDict` using `gvar.BufferDict.add_distribution`, and the
        constructor returns an array of gvars with the appropriate shape for
        convenience. See `add_distribution`.
    shape : int or tuple of int
        The shape of the array of variables to be represented. If the variable
        is multivariate, this shape adds as leading axes in the array. Default
        scalar. This shape broadcasts with the non-core shapes of the
        parameters.

    Returns
    -------
    If `name` is None (default):

    distr : Distr
        An object representing the distribution.

    Else:

    gvars : array of gvars
        An array of primary gvars that can be set as value in a
        `gvar.BufferDict` under a key that uses the just defined name.

    Attributes
    ----------
    params : tuple
        The parameters as passed to the constructor.
    in_shape : tuple of int
        The core shape of the input array to `partial_invfcn`.
    shape : tuple of int
        The core shape of the output array of `partial_invfcn`.
    dtype : dtype
        The dtype of the output array of `partial_invfcn`.
    distrshape : tuple of int
        The sub-core shape of the output array of `partial_invfcn` that
        represents the atomic shape of the distribution.
    signature : Signature
        An object representing the signature of `invfcn`. This is a class
        attribute.

    Methods
    -------
    partial_invfcn :
        Transformation function from a (multivariate) Normal variable to the
        target random variable. Differs from the class method `invfcn` in that
        1) the core input shape is flat or scalar, 2) the parameters are baked
        in, including transformation of the random parameters.
    add_distribution :
        Define the distribution for usage with `gvar.BufferDict`.
    gvars :
        Return an array of gvars with the appropriate shape for usage with
        `gvar.BufferDict`.
    invfcn : classmethod
        Transformation function from a (multivariate) Normal variable to the
        target random variable.

    Examples
    --------

    Use directly with `gvar.BufferDict` by setting `name`:

    >>> copula = gvar.BufferDict({
    ...     'A(x)': lgp.copula.beta(1, 1, name='A'),
    ...     'B(y)': lgp.copula.beta(3, 5, name='B'),
    ... })
    >>> copula['x']
    0.50(40)
    >>> copula['y']
    0.36(18)

    Corresponding "unrolled" usage:

    >>> A = lgp.copula.beta(1, 1)
    >>> B = lgp.copula.beta(3, 5)
    >>> A.add_distribution('A')
    >>> B.add_distribution('B')
    >>> copula = gvar.BufferDict({
    ...     'A(x)': A.gvars(),
    ...     'B(y)': B.gvars(),
    ... })

    Notice that, although the name used for `add_distribution` must be globally
    unique, for convenience it is permitted to redefine the same distribution
    family with the same parameters, even from another `Distr` instance.

    Define a distribution with a random parameter:

    >>> X = lgp.copula.halfnorm(lgp.copula.invgamma(1, 1))

    Now `X` represents the model

    .. math::
        \sigma &\sim \mathrm{InvGamma}(1, 1), \\
        X \mid \sigma &\sim \mathrm{HalfNorm}(\sigma).

    Repeated usage of `Distr` instances for random parameters will share
    those parameters in the distributions. The following code:

    >>> sigma = lgp.copula.invgamma(1, 1)
    >>> X = lgp.copula.halfnorm(sigma)
    >>> Y = lgp.copula.halfcauchy(sigma)

    Corresponds to the model

    .. math::
        \sigma &\sim \mathrm{InvGamma}(1, 1), \\
        X \mid \sigma &\sim \mathrm{HalfNorm}(\sigma), \\
        Y \mid \sigma &\sim \mathrm{HalfCauchy}(\sigma),

    with the same parameter :math:`\sigma` shared between the two distributions.
    However, if the distributions are now put into a `gvar.BufferDict`, e.g.,
    with

    >>> sigmaXY = lgp.copula.makedict({'sigma': sigma, 'X': X, 'Y': Y})

    then this relationship breaks down; the model represented by the dictionary
    `sigmaXY` is

    .. math::
        \sigma, \sigma_X, \sigma_Y &\sim \mathrm{InvGamma}(1, 1), \\
        X \mid \sigma &\sim \mathrm{HalfNorm}(\sigma_X), \\
        Y \mid \sigma &\sim \mathrm{HalfCauchy}(\sigma_Y),

    with separate, independent parameters :math:`\sigma,\sigma_X,\sigma_Y`,
    because each dictionary entry is evaluated separately.
    
    To apply arbitrary transformations, use manually `invfcn`:

    >>> @functools.partial(lgp.gvar_gufunc, signature='(n)->(n)')
    >>> @functools.partial(jnp.vectorize, signature='(n)->(n)')
    >>> def model_invfcn(normal_params):
    ...     sigma2 = lgp.copula.invgamma.invfcn(normal_params[0], 1, 1)
    ...     sigma = jnp.sqrt(sigma2)
    ...     X = lgp.copula.halfnorm.invfcn(normal_params[1], sigma)
    ...     Y = lgp.copula.halfcauchy.invfcn(normal_params[2], sigma)
    ...     return jnp.stack([sigma, X, Y])
    
    Now the function `model_invfcn` represents the model

    .. math::
        \sigma^2 &\sim \mathrm{InvGamma}(1, 1), \\
        X \mid \sigma &\sim \mathrm{HalfNorm}(\sigma), \\
        Y \mid \sigma &\sim \mathrm{HalfCauchy}(\sigma).

    The `jax.numpy.vectorize` decorator makes `model_invfcn` support
    broadcasting on additional input axes, while `gvar_gufunc` makes it accept
    gvars as input.

    See also
    --------
    gvar.BufferDict.uniform

    Notes
    -----
    Concrete subclasses must define `invfcn`, and define the class attribute
    `signature` to the numpy signature string of `invfcn`, unless `invfcn` is
    an ufunc. `invfcn` must be vectorized.

    """
    
    @classmethod
    @abc.abstractmethod
    def invfcn(cls, x, *params):
        r"""

        Normal to desired distribution transformation.

        Maps a (multivariate) Normal variable to a variable with the desired
        marginal distribution. In symbols: :math:`y = F^{-1}(\Phi(x))`. This
        function is a generalized ufunc, jax traceable, vmappable one time, and
        differentiable one time. The signature is accessible through the
        class attribute `signature`.

        Parameters
        ----------
        x : array_like
            The input Normal variable.
        *params : array_like
            The parameters of the distribution.

        Returns
        -------
        y : array_like
            The output variable with the desired marginal distribution.

        """
        pass

    def _eval_shapes(self, shape):

        # convert shape to tuple
        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        else:
            shape = tuple(shape)

        # make sure parameters have a shape
        array_params = [
            p if hasattr(p, 'shape') else jnp.asarray(p)
            for p in self.params
        ]

        # parse signature of cls.invfcn
        sig = self.signature.eval(None, *array_params)
        x = jax.ShapeDtypeStruct(shape + sig.core_in_shapes[0], 'd')
        sig = self.signature.eval(x, *array_params)
        self._in_shape_1 = sig.in_shapes[0]
        self.distrshape, = sig.core_out_shapes
        self.shape, = sig.out_shapes
        
        self._compute_in_shape()

    def _compute_in_shape(self):
        in_size = numpy.prod(self._in_shape_1, dtype=int)
        cache = set()
        for p in self.params:
            if isinstance(p, __class__):
                in_size += p._compute_in_size(cache)
        if in_size == 1:
            self.in_shape = ()
        else:
            self.in_shape = in_size,
        self._ancestor_count = len(cache)

    def _compute_in_size(self, cache):
        if self in cache:
            return 0
        cache.add(self)
        in_size = numpy.prod(self._in_shape_1, dtype=int)
        for p in self.params:
            if isinstance(p, __class__):
                in_size += p._compute_in_size(cache)
        return in_size

    @functools.cached_property
    def _partial_invfcn_internal(self):
        
        @functools.wraps(self.invfcn)
        def _partial_invfcn_internal(x, i, cache):
            assert x.ndim == 1

            # loop over parameters
            concrete_params = []
            for p in self.params:
                
                # parameter is not a copula, convert to array
                if not isinstance(p, __class__):
                    p = jnp.asarray(p)
                
                # parameter is an already evaluated copula
                elif p in cache:
                    p = cache[p]
                
                # parameter is an unseen copula, evaluate and cache the result
                else:
                    y, i, cache = p._partial_invfcn_internal(x, i, cache)
                    cache[p] = y
                    p = y
                
                concrete_params.append(p)

            # evaluate inverse transformation
            in_size = numpy.prod(self._in_shape_1, dtype=int)
            assert i + in_size <= x.size
            last = x[i:i + in_size].reshape(self._in_shape_1)
            y = self.invfcn(last, *concrete_params)
            assert y.shape == self.shape
            assert y.dtype == self.dtype
            return y, i + in_size, cache

        return _partial_invfcn_internal

    @functools.cached_property
    def _partial_invfcn(self):

        # determine signature
        shapestr = lambda shape: ','.join(map(str, shape))
        signature = f'({shapestr(self.in_shape)})->({shapestr(self.shape)})'

        # wrap to support gvars
        @functools.partial(_patch_gvar.gvar_gufunc, signature=signature)
        # jax.jit?
        @functools.partial(jnp.vectorize, signature=signature)
        @functools.wraps(self._partial_invfcn_internal)
        def _partial_invfcn(x):
            assert x.shape == self.in_shape
            if not self.in_shape:
                x = x[None]
            y, i, cache = self._partial_invfcn_internal(x, 0, {})
            assert i == x.size
            assert len(cache) == self._ancestor_count
            return y

        return _partial_invfcn

    def partial_invfcn(self, x):
        """
            
        Map independent Normal variables to the desired distribution.

        This function is a generalized ufunc. It is jax traceable and
        differentiable one time. It supports arrays of gvars as input.

        Parameters
        ----------
        x : array
            An array of values representing draws of i.i.d. Normal variates.

        Returns
        -------
        y : array
            An array of values representing draws of the desired distribution.

        """

        return self._partial_invfcn(x)

    def __init_subclass__(cls, **_):
        cls._named = {}
        if not hasattr(cls, 'signature'):
            sig = inspect.signature(cls.invfcn)
            cls.signature = ','.join(['()'] * len(sig.parameters)) + '->()'
        if not isinstance(cls.signature, _signature.Signature):
            cls.signature = _signature.Signature(cls.signature)
        cls.signature.check_nargs(cls.invfcn)
        if not hasattr(cls, 'dtype'):
            cls.dtype = jax.dtypes.canonicalize_dtype(jnp.float64)

    def __new__(cls, *params, name=None, shape=()):

        self = super().__new__(cls)
        self.params = params
        self._eval_shapes(shape)

        if name is None:
            return self
        else:
            self.add_distribution(name)
            return self.gvars()
    
    class _Descr(collections.namedtuple('Distr', 'family shape params')):
        """ static representation of a Distr object """

        def _repr(self, conv):
            args = list(map(conv, self.params))
            if len(self.shape) == 1:
                args += [f'shape={self.shape[0]}']
            elif self.shape:
                args += [f'shape={self.shape}']
            arglist = ', '.join(args)
            return f'{self.family.__name__}({arglist})'

        __repr__ = lambda self: self._repr(repr)
        __str__ = lambda self: self._repr(str)

    def _descr(self, param_converter):

        params = []
        for p in self.params:
            if isinstance(p, __class__):
                p = p._descr(param_converter)
            else:
                p = param_converter(p)
            params.append(p)

        return self._Descr(self.__class__, self.shape, tuple(params))

    @functools.cached_property
    def _staticdescr(self):
        """ static description of self, can be compared """
        return self._descr(lambda p: numpy.asarray(p).tolist())

    @functools.cached_property
    def _repr(self):
        """ description of self for representation """
        def param_converter(p):
            if hasattr(p, 'shape'):
                return f'array{p.shape}'
            else:
                return p
        return self._descr(param_converter)

    def __repr__(self):
        return str(self._repr)

    def _is_same_family(self, invfcn):
        return getattr(invfcn, '__self__', None).__class__ is self.__class__

    def add_distribution(self, name):
        """

        Define the distribution for usage with `gvar.BufferDict`.

        Parameters
        ----------
        name : str
            The name to use for the distribution. It must be globally unique,
            and it should not contain parentheses. To redefine a distribution
            with the same name, use `gvar.BufferDict.del_distribution` first.
            However, it is allowed to reuse the name if the distribution family
            and parameters are identical to those used for the existing
            definition.

        See also
        --------
        gvar.BufferDict.add_distribution, gvar.BufferDict.del_distribution

        """

        if gvar.BufferDict.has_distribution(name):
            invfcn = gvar.BufferDict.invfcn[name]
            if not self._is_same_family(invfcn):
                raise ValueError(f'distribution {name} already defined')
            existing = self._named[name]
            if existing != self._staticdescr:
                raise ValueError('Attempt to overwrite existing'
                    f' {self.__class__.__name__} distribution with name {name}')
                # cls._named is not updated by
                # gvar.BufferDict.del_distribution, but it is not a problem
        
        else:
            gvar.BufferDict.add_distribution(name, self.partial_invfcn)
            self._named[name] = self._staticdescr
    
    def gvars(self):
        """

        Return an array of gvars intended as value in a `gvar.BufferDict`.

        Returns
        -------
        gvars : array of gvars
            An array of i.i.d. standard Normal primary gvars with shape
            `in_shape`.

        """

        return gvar.gvar(numpy.zeros(self.in_shape), numpy.ones(self.in_shape))

# TODO
# - make Distr instances dispatching array-likes that perform the operations
#   by creating a new instance with a custom invfcn (this requires always
#   getting invfcn from self!) from a generic subclass that just applies the
#   operation to the output of the operand invfcns using jax.numpy.
