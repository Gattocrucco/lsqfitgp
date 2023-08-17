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
#     'ciao_halfcauchy_1.0(a)': lgp.copula.halfcauchy('a', 1.0),
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

from .. import _patch_gvar
from .. import _array

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

    def __new__(cls, name, *params, shape=()):
        
        if gvar.BufferDict.has_distribution(name):
            invfcn = gvar.BufferDict.invfcn[name]
            if getattr(invfcn, '_CopulaFactory_subclass_', None) is not cls:
                raise ValueError(f'distribution {name} already defined')
            if (existing := cls.params[name]).params != params:
                raise ValueError(f'Attempt to overwrite existing {cls.__name__}'
                    f' distribution with name {name}')
                # cls.params is not updated by
                # gvar.BufferDict.del_distribution, but it is not a problem
            nvariables = existing.nvariables
        
        else:
            invfcn, nvariables = cls._partial_invfcn(params)
            gvar.BufferDict.add_distribution(name, invfcn)
            cls.params[name] = cls._Params(params, nvariables)
        
        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        if nvariables > 1:
            shape = shape + (nvariables,)
        return gvar.gvar(numpy.zeros(shape), numpy.ones(shape))

        # TODO I could attach metadata to the output ndarray here, such that
        # recursive copulas can be built by calling __new__ instead of the
        # custom tuple format. I can accumulate the gvars directly. This would
        # allow supporting custom initialization code, and also the shape
        # parameter. It's also more natural for the user. To attach metadata I
        # need to subclass ndarray, I think it would be safe since I won't add
        # anything but a metadata dict.
        # https://numpy.org/doc/stable/user/basics.subclassing.html#basics-subclassing
        # Subclass ndarray, define __array_finalize__(self, obj) to copy
        # metadata from obj if it's the same class, use ndarray.view to cast to
        # the new class, set the metadata on the new array
