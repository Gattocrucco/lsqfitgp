# lsqfitgp/_Kernel/_util.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

import numbers
import operator

import numpy
from jax import numpy as jnp

from .. import _array

def is_numerical_scalar(x):
    return (
        isinstance(x, numbers.Number) or
        (isinstance(x, (numpy.ndarray, jnp.ndarray)) and x.ndim == 0)
    )
    # do not use jnp.isscalar because it returns False for strongly
    # typed 0-dim arrays; do not use jnp.ndim(•) == 0 because it accepts
    # non-numerical types

# TODO reimplement with tree_reduce, closuring ndim to recognize shaped fields
def _reduce_recurse_dtype(fun, args, reductor, npreductor, jnpreductor):
    x = args[0]
    if x.dtype.names is None:
        return fun(*args)
    else:
        acc = None
        for name in x.dtype.names:
            recargs = tuple(arg[name] for arg in args)
            result = _reduce_recurse_dtype(fun, recargs, reductor, npreductor, jnpreductor)
            
            dtype = x.dtype[name]
            if dtype.ndim:
                axis = tuple(range(-dtype.ndim, 0))
                red = jnpreductor if isinstance(result, jnp.ndarray) else npreductor
                result = red(result, axis=axis)
            
            if acc is None:
                acc = result
            else:
                acc = reductor(acc, result)
        
        assert acc.shape == _array.broadcast(*args).shape
        return acc

def sum_recurse_dtype(fun, *args):
    return _reduce_recurse_dtype(fun, args, operator.add, numpy.sum, jnp.sum)

def prod_recurse_dtype(fun, *args):
    return _reduce_recurse_dtype(fun, args, operator.mul, numpy.prod, jnp.prod)

# TODO reimplement with tree_map
def transf_recurse_dtype(transf, x, *args):
    if x.dtype.names is None:
        return transf(x, *args)
    else:
        x = _array.StructuredArray(x)
        for name in x.dtype.names:
            newargs = tuple(y[name] for y in args)
            x = x.at[name].set(transf_recurse_dtype(transf, x[name], *newargs))
        return x
