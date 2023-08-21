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

""" define Distr """

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
from .. import _signature
from . import _base

class Distr(_base.DistrBase):
    r"""

    Abstract base class to represent probability distributions.

    A `Distr` object represents a probability distribution of a variable in
    :math:`\mathbb R^n`, and provides a transformation function from a
    (multivariate) Normal variable to the target random variable.

    The main functionality is defined in `DistrBase`. The additional attributes
    and methods `params`, `signature`, and `invfcn` are not intended for common
    usage.

    Parameters
    ----------
    *params : tuple of scalar, array or Distr
        The parameters of the distribution. If the parameters have leading axes
        other than those required, the distribution is repeated i.i.d.
        over those axes. If a parameter is an instance of `Distr` itself, it
        is a random parameter and its distribution is accounted for.
    shape : int or tuple of int
        The shape of the array of i.i.d. variables to be represented, scalar by
        default. If the variable is multivariate, this shape adds as leading
        axes in the array. This shape broadcasts with the non-core shapes of the
        parameters.
    name : str, optional
        If specified, the distribution is defined for usage with
        `gvar.BufferDict` using `gvar.BufferDict.add_distribution`, and for
        convenience the constructor returns an array of gvars with the
        appropriate shape instead of the `Distr` object. See `add_distribution`.

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
    signature : Signature
        An object representing the signature of `invfcn`. This is a class
        attribute.

    Methods
    -------
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

    To generate automatically sensible names and avoid repeating them twice, use
    `makedict`:

    >>> lgp.copula.makedict({
    ...     'x': lgp.copula.beta(1, 1),
    ...     'y': lgp.copula.beta(3, 5),
    ... })
    BufferDict({'__copula_beta{1, 1}(x)': 0.0(1.0), '__copula_beta{3, 5}(y)': 0.0(1.0)})

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
    However, if the distributions are now put into a `gvar.BufferDict`, with
    
    >>> sigma.add_distribution('distr_sigma')
    >>> X.add_distribution('distr_X')
    >>> Y.add_distribution('distr_Y')
    >>> bd = gvar.BufferDict({
    ...     'distr_sigma(sigma)': sigma.gvars(),
    ...     'distr_X(X)': X.gvars(),
    ...     'distr_Y(Y)': Y.gvars(),
    ... })

    then this relationship breaks down; the model represented by the dictionary
    `bd` is

    .. math::
        \sigma &\sim \mathrm{InvGamma}(1, 1), \\
        X \mid \sigma_X &\sim \mathrm{HalfNorm}(\sigma_X), \quad
            & \sigma_X &\sim \mathrm{InvGamma}(1, 1), \\
        Y \mid \sigma_Y &\sim \mathrm{HalfCauchy}(\sigma_Y), \quad
            & \sigma_Y &\sim \mathrm{InvGamma}(1, 1),

    with separate, independent parameters :math:`\sigma,\sigma_X,\sigma_Y`,
    because each dictionary entry is evaluated separately. Indeed, trying to do
    this with `makedict` will raise an error:

    >>> bd = lgp.copula.makedict({'sigma': sigma, 'X': X, 'Y': Y})
    ValueError: cross-key occurrences of object(s):
    invgamma with id 10952820880: <sigma>, <X.0>, <Y.0>

    To use all the distributions at once while preserving the relationships,
    put them into a container of choice and wrap it as a `Copula` object:

    >>> sigmaXY = lgp.copula.Copula({'sigma': sigma, 'X': X, 'Y': Y})

    The `Copula` provides a `partial_invfcn` function to map Normal variables
    to a structure, with the same layout as the input one, of desired variates.
    The whole `Copula` can be used in `gvar.BufferDict`:

    >>> bd = lgp.copula.makedict({'sigmaXY': sigmaXY})
    >>> bd
    BufferDict({"__copula_{'sigma': invgamma{1, 1}, 'X': halfnorm{_Path{path=[{DictKey{key='sigma'},}]}}, 'Y': halfcauchy{_Path{path=[{DictKey{key='sigma'},}]}}}(sigmaXY)": array([0.0(1.0), 0.0(1.0), 0.0(1.0)], dtype=object)})
    >>> bd['sigmaXY']
    {'sigma': 1.4(1.7), 'X': 1.0(1.4), 'Y': 1.4(2.5)}
    >>> gvar.corr(bd['sigmaXY']['X'], bd['sigmaXY']['Y'])
    0.5260263326687865

    Although the actual dictionary value is an array, getting the unwrapped key
    reproduces the original structure.
    
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
    DistrBase, Copula, gvar.BufferDict.uniform

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

        # check number of parameters
        if self.signature.nin != 1 + len(self.params):
            raise TypeError(f'{self.__class__.__name__} distribution has '
                f'{self.signature.nin - 1} parameters, but {len(self.params)} '
                'parameters were passed to the constructor')

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
        if (out := super()._compute_in_size(cache)) is not None:
            return out
        in_size = numpy.prod(self._in_shape_1, dtype=int)
        for p in self.params:
            if isinstance(p, __class__):
                in_size += p._compute_in_size(cache)
        return in_size

    def _partial_invfcn_internal(self, x, i, cache):
        if (out := super()._partial_invfcn_internal(x, i, cache)) is not None:
            return out

        concrete_params = []
        for p in self.params:
            
            if isinstance(p, __class__):
                p, i = p._partial_invfcn_internal(x, i, cache)
            else:
                p = jnp.asarray(p)
            
            concrete_params.append(p)

        in_size = numpy.prod(self._in_shape_1, dtype=int)
        assert i + in_size <= x.size
        last = x[i:i + in_size].reshape(self._in_shape_1)
        
        y = self.invfcn(last, *concrete_params)
        assert y.shape == self.shape
        assert y.dtype == self.dtype
        
        cache[self] = y
        return y, i + in_size

    @functools.cached_property
    def _partial_invfcn(self):

        # determine signature
        shapestr = lambda shape: ','.join(map(str, shape))
        signature = f'({shapestr(self.in_shape)})->({shapestr(self.shape)})'

        # wrap to support gvars
        @functools.partial(_patch_gvar.gvar_gufunc, signature=signature)
        # @jax.jit
        @functools.partial(jnp.vectorize, signature=signature)
        def _partial_invfcn(x):
            assert x.shape == self.in_shape
            if not self.in_shape:
                x = x[None]
            cache = {}
            y, i = self._partial_invfcn_internal(x, 0, cache)
            assert i == x.size
            assert len(cache) == 1 + self._ancestor_count
            return y

        return _partial_invfcn

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not hasattr(cls, 'signature'):
            sig = inspect.signature(cls.invfcn)
            cls.signature = ','.join(['()'] * len(sig.parameters)) + '->()'
        if not isinstance(cls.signature, _signature.Signature):
            cls.signature = _signature.Signature(cls.signature)
        cls.signature.check_nargs(cls.invfcn)
        if getattr(cls, 'dtype', NotImplemented) is NotImplemented:
            cls.dtype = jax.dtypes.canonicalize_dtype(jnp.float64)

    def __new__(cls, *params, shape=(), name=None):

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

        def __repr__(self):
            args = list(map(repr, self.params))
            if len(self.shape) == 1:
                args += [f'shape={self.shape[0]}']
            elif self.shape:
                args += [f'shape={self.shape}']
            arglist = ', '.join(args)
            return f'{self.family.__name__}({arglist})'

    def _compute_staticdescr(self, path, cache):
        if (obj := super()._compute_staticdescr(path, cache)) is not None:
            return obj
        
        params = []
        for i, p in enumerate(self.params):
            if isinstance(p, __class__):
                p = p._compute_staticdescr(path + [i], cache)
            else:
                p = numpy.asarray(p).tolist()
            params.append(p)

        return self._Descr(self.__class__, self.shape, tuple(params))

    def _shapestr(self, shape):
        if shape:
            return (str(shape)
                .replace(',)', ')')
                .replace('(' , '[')
                .replace(')' , ']')
                .replace(' ', '')
            )
        else:
            return ''
        
    def __repr__(self, path='', cache=None):

        if isinstance(cache := super().__repr__(path, cache), str):
            return cache
        
        args = []
        for i, p in enumerate(self.params):
            
            if isinstance(p, __class__):
                p = p.__repr__('.'.join((path, str(i))).lstrip('.'), cache)
            elif hasattr(p, 'shape'):
                p = f'Array{self._shapestr(p.shape)}'
            else:
                p = repr(p)
            args.append(p)

        if len(self.shape) == 1:
            args += [f'shape={self.shape[0]}']
        elif self.shape:
            args += [f'shape={self.shape}']

        return f'{self.__class__.__name__}({", ".join(args)})'

# TODO
# - make Distr instances dispatching array-likes that perform the operations
#   by creating a new instance from a generic Op subclass that just applies the
#   operation to the output of the operand invfcns using jax.numpy.
