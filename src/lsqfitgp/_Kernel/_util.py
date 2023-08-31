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
import jax
from jax import numpy as jnp
from jax import tree_util

from .. import _array

def is_numerical_scalar(x):
    return (
        isinstance(x, numbers.Number) or
        (isinstance(x, (numpy.ndarray, jnp.ndarray)) and x.ndim == 0)
    )
    # do not use jnp.isscalar because it returns False for strongly
    # typed 0-dim arrays; do not use jnp.ndim(•) == 0 because it accepts
    # non-numerical types

def is_nonnegative_integer_scalar(x):
    if isinstance(x, numbers.Integral) and x >= 0:
        # python scalars and numpy scalars
        return True
    if isinstance(x, numpy.ndarray) and x.ndim == 0 and numpy.issubdtype(x.dtype, numpy.integer) and x.item() >= 0:
        # 0-dim numpy arrays
        return True
    if isinstance(x, jnp.ndarray) and x.ndim == 0 and jnp.issubdtype(x.dtype, jnp.integer):
        try:
            # concrete jax arrays
            return x.item() >= 0
        except jax.errors.ConcretizationTypeError:
            # jax tracers
            return jnp.issubdtype(x.dtype, jnp.unsignedinteger)
    return False

def is_scalar_cond_trueontracer(x, cond):
    if isinstance(x, numbers.Number) and cond(x):
        # python scalars and numpy scalars
        return True
    if isinstance(x, numpy.ndarray) and x.ndim == 0 and numpy.issubdtype(x.dtype, numpy.number) and cond(x.item()):
        # 0-dim numpy arrays
        return True
    if isinstance(x, jnp.ndarray) and x.ndim == 0 and jnp.issubdtype(x.dtype, jnp.number):
        try:
            # concrete jax arrays
            return cond(x.item())
        except jax.errors.ConcretizationTypeError:
            # jax tracers
            return True
    return False

def is_nonnegative_scalar_trueontracer(x):
    return is_scalar_cond_trueontracer(x, lambda x: x >= 0)

# TODO reimplement with tree_reduce, closuring ndim to recognize shaped fields
def _reduce_recurse_dtype(fun, args, reductor, npreductor, jnpreductor, **kw):
    x = args[0]
    if x.dtype.names is None:
        return fun(*args, **kw)
    else:
        acc = None
        for name in x.dtype.names:
            recargs = tuple(arg[name] for arg in args)
            result = _reduce_recurse_dtype(fun, recargs, reductor, npreductor, jnpreductor, **kw)
            
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

def sum_recurse_dtype(fun, *args, **kw):
    return _reduce_recurse_dtype(fun, args, operator.add, numpy.sum, jnp.sum, **kw)

def prod_recurse_dtype(fun, *args, **kw):
    return _reduce_recurse_dtype(fun, args, operator.mul, numpy.prod, jnp.prod, **kw)

def ufunc_recurse_dtype(ufunc, x, *args):
    """ apply an ufunc to all the leaf fields """
    
    allargs = (x, *args)
    expected_shape = jnp.broadcast_shapes(*(x.shape for x in allargs))

    if x.dtype.names is None:
        out = ufunc(*allargs)
    else:
        args = map(_array.StructuredArray, allargs)
        out = tree_util.tree_map(ufunc, *args)
    
    assert out.shape == expected_shape
    return out
