# lsqfitgp/copula/_base.py
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
#     'ciao_halfcauchy_1.0(a)': lgp.copula.halfcauchy(1.0, name='a'),
# })
#
# The default prefix would be something like '__copula_'.
#
# I need to remove any parentheses in the name.

# TODO document the conditional copula functionality.

import abc
import functools
import collections
import numbers

import gvar
import numpy
import jax
from jax import numpy as jnp

from .. import _patch_gvar
from .. import _array

class Distr(metaclass=abc.ABCMeta):
    """
    Abstract base class to represent distributions.

    A `Distr` object represents a probability distribution, and provides a
    transformation function from a (multivariate) Normal variable to the target
    random variable, in particular for use with `gvar.BufferDict`.

    Parameters
    ----------
    *params : scalars
        The parameters of the distribution.
    name : str, optional
        If specified, the distribution is defined for usage with
        `gvar.BufferDict` using `gvar.BufferDict.add_distribution`, and the
        constructor returns an array of gvars with the appropriate shape for
        convenience.
    shape : int or tuple of int
        The shape of the array of variables to be represented. If the variable
        is multivariate, this shape adds as leading axes in the array. Default
        scalar. This shape broadcasts with those of the parameters.

    Returns
    -------
    If `name` is None (default):

    distr : Distr
        An object representing the distribution.

    Else:

    gvars : array of gvars
        An array of primary gvars that can be set as value in a
        `gvar.BufferDict` under a key that uses the just defined name.

    Examples
    --------
    >>> copula = gvar.BufferDict({
    ...     'A(x)': lgp.copula.beta(1, 1, name='A'),
    ...     'B(y)': lgp.copula.beta(3, 5, name='B'),
    ... })
    >>> copula['x']
    0.50(40)
    >>> copula['y']
    0.36(18)

    See also
    --------
    gvar.BufferDict.uniform
    """
    
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
        x : (*head_shape, *in_shape) array or scalar
            The input Normal variable. If multivariate, the variable shape must
            be in the core shape `in_shape`. The calculation is broadcasted over
            other leading axes. Use `input_shape` to determine the required
            variable shape.
        params : scalar or arrays
            The parameters of the distribution. They are broadcasted with `x`.

        Returns
        -------
        y : (*head_shape, *out_shape) array or scalar
            The output variable with the desired marginal distribution. Use
            `output_shape` to determine `out_shape` before calling `invfcn`.

        """
        pass

    @classmethod
    def input_shape(cls, *params):
        """
        Return the input core shape expected by `invfcn`.

        May depend on the type and shape of the parameters, but not on their
        values. This shape applies only to the class member `invfcn`, not to the
        one intended for `gvar.BufferDict` that is created by the class
        constructor, which always expects either a scalar or vector core shape.

        Parameters
        ----------
        params :
            The same arguments that would be passed to `invfcn` or to the
            class constructor.

        Returns
        -------
        shape : tuple of int
            The required core shape of the input array `x` to `invfcn`.
        """
        return ()

    @classmethod
    def output_shape(cls, *params):
        """
        Return the output core shape of `invfcn`.

        May depend on the type and shape of the parameters, but not on their
        values. Contrary to `input_shape`, this function applies both to
        `invfcn` and to the function used by `gvar.BufferDict`.

        Parameters
        ----------
        params :
            The same arguments that would be passed to `invfcn` or to the
            class constructor.

        Returns
        -------
        shape : tuple of int
            The core shape of the output array `y` of `invfcn`.
        """
        return ()

    @classmethod
    def _eval_shapes(cls, params, shape):

        # convert parameters to array dummies
        dummies = []
        for p in params:
            if not hasattr(p, 'shape') or not hasattr(p, 'dtype'):
                p = jnp.asarray(p)
            dummies.append(jax.ShapeDtypeStruct(p.shape, p.dtype))
        
        # determine signature of cls.invfcn
        in_shape_0 = cls.input_shape(*dummies)
        out_shape_0 = cls.output_shape(*dummies)
        
        # make a dummy for the input to cls.invfcn
        x_dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
        x_dummy = jax.ShapeDtypeStruct(shape + in_shape_0, x_dtype)
        
        # use jax to get shape with parameter broadcasting
        out_dummy = jax.eval_shape(cls.invfcn, x_dummy, *dummies)
        broadcast_ndim = len(out_dummy.shape) - len(out_shape_0)
        assert broadcast_ndim >= 0
        assert out_dummy.shape[broadcast_ndim:] == out_shape_0

        # determine shapes to use with cls.invfcn
        in_shape_1 = out_dummy.shape[:broadcast_ndim] + in_shape_0
        out_shape_1 = out_dummy.shape
        out_dtype = out_dummy.dtype

        return in_shape_1, out_shape_1, out_shape_0, out_dtype

    @classmethod
    def _compute_in_shape_2(cls, params, in_shape_1):
        in_size = 0
        for p in params:
            if isinstance(p, __class__):
                in_size += numpy.prod(p.in_shape, dtype=int)
        in_size += numpy.prod(in_shape_1, dtype=int)
        if in_size == 1:
            return ()
        else:
            return in_size,

    @classmethod
    def _partial_invfcn(cls, params, in_shape_2, in_shape_1, out_shape_1, out_dtype):
        
        # determine signature
        shapestr = lambda shape: ', '.join(map(str, shape))
        signature = f'({shapestr(in_shape_2)})->({shapestr(out_shape_1)})'

        # wrap to support input array of gvars and unpack input into parameters'
        # distributions and our
        @functools.partial(_patch_gvar.gvar_gufunc, signature=signature)
        @functools.wraps(cls.invfcn)
        def partial_invfcn(x):

            # check input
            x = jnp.asarray(x)
            assert x.ndim >= len(in_shape_2)
            assert x.shape[x.ndim - len(in_shape_2):] == in_shape_2
            if not in_shape_2:
                x = x[..., None]
            in_size = numpy.prod(in_shape_2, dtype=int)

            # loop over parameters
            i = 0
            concrete_params = []
            for p in params:
                
                # parameter is a copula, apply it
                if isinstance(p, __class__):
                    p_in_size = numpy.prod(p.in_shape, dtype=int)
                    assert i + p_in_size <= in_size
                    p_x = x[..., i:i + p_in_size]
                    p_x = p_x.reshape(p_x.shape[:-1] + p.in_shape)
                    p = p.invfcn(p_x)
                    i += p_in_size
                
                concrete_params.append(p)

            # evaluate inverse transformation
            last = x[..., i:]
            last = last.reshape(last.shape[:-1] + in_shape_1)
            out = cls.invfcn(last, *concrete_params)
            assert out.shape == x.shape[:-1] + out_shape_1
            assert out.dtype == out_dtype
            return out

        return partial_invfcn

    def __new__(cls, *params, name=None, shape=()):

        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        else:
            shape = tuple(shape)

        in_shape_1, out_shape_1, out_shape_0, out_dtype = cls._eval_shapes(params, shape)
        in_shape_2 = cls._compute_in_shape_2(params, in_shape_1)
        invfcn = cls._partial_invfcn(params, in_shape_2, in_shape_1, out_shape_1, out_dtype)

        self = super().__new__(cls)
        self.invfcn = invfcn
        self.params = params
        self.in_shape = in_shape_2
        self.coreshape = out_shape_0
        self.shape = out_shape_1
        self.dtype = out_dtype

        if name is None:
            return self
        else:
            self.name(name)
            return self.gvars()
    
    class _Descr(collections.namedtuple('Distr', 'family shape params')):

        def __repr__(self):
            args = [
                f'shape={self.shape}',
            ] + list(map(repr, self.params))
            arglist = ', '.join(args)
            return f'{self.family.__name__}({arglist})'

    @functools.cached_property
    def _descr(self):
        
        staticparams = []
        for p in self.params:
            if isinstance(p, __class__):
                p = p._descr
            else:
                p = numpy.asarray(p).tolist()
            staticparams.append(p)

        return self._Descr(self.__class__, self.shape, tuple(staticparams))

    @classmethod
    def _earmark(cls, obj):
        obj._Distr_subclass_ = cls

    @classmethod
    def _is_earmarked(cls, obj):
        return getattr(obj, '_Distr_subclass_', None) is cls

    def __init_subclass__(cls, **_):
        cls._named = {}
    
    def name(self, name):

        if gvar.BufferDict.has_distribution(name):
            invfcn = gvar.BufferDict.invfcn[name]
            if not self._is_earmarked(invfcn):
                raise ValueError(f'distribution {name} already defined')
            existing = self._named[name]
            if existing != self._descr:
                raise ValueError('Attempt to overwrite existing'
                    f' {self.__class__.__name__} distribution with name {name}')
                # cls._named is not updated by
                # gvar.BufferDict.del_distribution, but it is not a problem
        
        else:
            self._earmark(self.invfcn)
            gvar.BufferDict.add_distribution(name, self.invfcn)
            self._named[name] = self._descr
    
    def gvars(self):
        return gvar.gvar(numpy.zeros(self.in_shape), numpy.ones(self.in_shape))
