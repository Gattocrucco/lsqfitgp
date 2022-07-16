# lsqfitgp/_array.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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

import numpy
from numpy.lib import recfunctions
from jax import numpy as jnp
from jax import tree_util

__all__ = [
    'StructuredArray',
    'broadcast',
    'broadcast_to',
    'broadcast_arrays',
    'asarray',
]

def _readonlyview(x):
    if isinstance(x, numpy.ndarray):
        x = x.view()
        x.flags.writeable = False
        # jax arrays and StructuredArrays are already readonly
    return x

def _wrapifstructured(x):
    if x.dtype.names is None:
        return x
    else:
        return StructuredArray(x)

@tree_util.register_pytree_node_class
class StructuredArray:
    """
    JAX-friendly imitation of a numpy structured array.
    
    It behaves like a read-only numpy structured array, with the exception that
    you can set a whole field/subfield.
    
    Examples
    --------
    >>> a = numpy.empty(3, dtype=[('f', float), ('g', float)])
    >>> a = StructuredArray(a)
    >>> a['f'] = numpy.arange(3) # this is allowed
    >>> a[1] = (0.3, 0.4) # this raises an error
        
    Parameters
    ----------
    array : numpy array, StructuredArray
        A structured array. An array qualifies as structured if
        ``array.dtype.names is not None``.
    
    Notes
    -----
    The StructuredArray is a readonly view on the input array. When you
    change the content of a field of the StructuredArray, however, the
    reference to the original array for that field is lost.
    
    """
    
    @classmethod
    def _array(cls, s, t, d, readonly=False):
        """
        Create a new StructuredArray with shape `s`, dtype `t`, using `d` as
        `_dict` member (no copy).
        """
        # if s is None:
        #     # infer the shape from the arrays in the dictionary
        #     f0 = t.names[0]
        #     a0 = d[f0]
        #     subshape = t[0].shape
        #     s = a0.shape[:len(a0.shape) - len(subshape)]
        out = super().__new__(cls)
        out.dtype = t
        out.shape = s
        out._dict = d
        out._readonly = readonly
        return out
    
    @property
    def size(self):
        return numpy.prod(self.shape, dtype=int)
    
    @property
    def ndim(self):
        return len(self.shape)
    
    def __new__(cls, array):
        assert isinstance(array, (numpy.ndarray, cls))
        assert array.dtype.names is not None
        d = {
            name: _readonlyview(_wrapifstructured(array[name]))
            for name in array.dtype.names
        }
        return cls._array(array.shape, array.dtype, d)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._dict[key]
        elif isinstance(key, list) and key and all(isinstance(k, str) for k in key):
            d = {
                name: self._dict[name]
                for name in key
            }
            return self._array(self.shape, self.dtype[key], d)
        else:
            d = {
                name: x[
                    (key if isinstance(key, tuple) else (key,))
                    + (slice(None),) * len(self.dtype[name].shape)
                ]
                for name, x in self._dict.items()
            }
            shape = numpy.empty(self.shape, [])[key].shape
            return self._array(shape, self.dtype, d, readonly=True)
    
    def __setitem__(self, key, val):
        if self._readonly:
            msg = 'This StructuredArray is read-only, maybe you did '
            msg += '"array[index][label] = ..." instead of '
            msg += '"array[label] = numpy.array([...])"?'
            raise ValueError(msg)
        assert key in self.dtype.names
        assert isinstance(val, (numpy.ndarray, jnp.ndarray, StructuredArray))
        prev = self._dict[key]
        # TODO support casting and broadcasting
        assert prev.dtype == val.dtype
        assert prev.shape == val.shape
        self._dict[key] = _readonlyview(_wrapifstructured(val))
    
    def reshape(self, *shape):
        """
        Reshape the array without changing its contents. See numpy.ndarray.reshape.
        """
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        d = {
            name: x.reshape(shape + self.dtype[name].shape)
            for name, x in self._dict.items()
        }
        shape = numpy.empty(self.shape, []).reshape(shape).shape
        return self._array(shape, self.dtype, d)
    
    def broadcast_to(self, shape, **kw):
        """
        Return a view of the array broadcasted to another shape. See
        numpy.broadcast_to.
        """
        numpy.broadcast_shapes(self.shape, shape) # raises if not broadcastable
        d = {
            name: broadcast_to(x, shape + self.dtype[name].shape, **kw)
            for name, x in self._dict.items()
        }
        return self._array(shape, self.dtype, d)
    
    def tree_flatten(self):
        """JAX PyTree encoder. See `jax.tree_util.tree_flatten`."""
        children = tuple(self._dict.values())
        aux_data = dict(
            keys = tuple(self._dict.keys()),
            dtype = self.dtype,
            shape = self.shape,
            _readonly = self._readonly,
        )
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """JAX PyTree decoder. See `jax.tree_util.tree_unflatten`."""
        obj = super().__new__(cls)
        for attr, val in aux_data.items():
            if attr != 'keys':
                setattr(obj, attr, val)
        obj._dict = dict(zip(aux_data['keys'], children))
        return obj
    
    def __repr__(self):
        # code from gvar https://github.com/gplepage/gvar
        # bufferdict.pyx:BufferDict:__str__
        out = 'StructuredArray({'

        listrepr = [(repr(k), repr(v)) for k, v in self._dict.items()]
        newlinemode = any('\n' in rv for _, rv in listrepr)
        
        for rk, rv in listrepr:
            if not newlinemode:
                out += '{}: {}, '.format(rk, rv)
            elif '\n' in rv:
                rv = rv.replace('\n', '\n    ')
                out += '\n    {}:\n    {},'.format(rk, rv)
            else:
                out += '\n    {}: {},'.format(rk, rv)
                
        if out.endswith(', '):
            out = out[:-2]
        elif newlinemode:
            out += '\n'
        out += '})'
        
        return out
    
    def __array__(self):
        array = numpy.empty(self.shape, self.dtype)
        self._copy_into_array(array)
        return array
    
    def _copy_into_array(self, dest):
        assert self.dtype == dest.dtype
        assert self.shape == dest.shape
        for name, src in self._dict.items():
            if isinstance(src, StructuredArray):
                src._copy_into_array(dest[name])
            else:
                dest[name][...] = src

    def __array_function__(self, func, types, args, kwargs):
        if func not in self._handled_functions:
            return NotImplemented
        return self._handled_functions[func](*args, **kwargs)
    
    _handled_functions = {}
    
    @classmethod
    def _implements(cls, np_function):
        "Register an __array_function__ implementation"
        def decorator(func):
            cls._handled_functions[np_function] = func
            return func
        return decorator
    
    def _fill_unstructured(self, out, idx=None, shape=None):
        if idx is None:
            idx = 0
        if shape is None:
            shape = self.shape
        for name, src in self._dict.items():
            if isinstance(src, StructuredArray):
                if src.dtype.names and self.dtype[name].subdtype:
                    for i in numpy.ndindex(src.shape[len(shape):]):
                        out, idx = src[(...,) + i]._fill_unstructured(out, idx, shape)
                    # TODO this loop is inefficient, use strides
                else:
                    out, idx = src._fill_unstructured(out, idx, shape)
            else:
                n = numpy.prod(src.shape[len(shape):], dtype=int)
                src = src.reshape(shape + (n,))
                key = (..., slice(idx, idx + n))
                if hasattr(out, 'at'):
                    out = out.at[key].set(src)
                else:
                    out[key] = src
                idx += n
        return out, idx

@StructuredArray._implements(numpy.broadcast_to)
def broadcast_to(x, shape, **kw):
    """
    Version of numpy.broadcast_to that works with StructuredArray and JAX
    arrays.
    """
    if isinstance(x, StructuredArray):
        return x.broadcast_to(shape, **kw)
    elif isinstance(x, jnp.ndarray):
        return jnp.broadcast_to(x, shape, **kw)
    else:
        return numpy.broadcast_to(x, shape, **kw)

@StructuredArray._implements(numpy.broadcast_arrays)
def broadcast_arrays(*arrays, **kw):
    """
    Version of numpy.broadcast_arrays that works with StructuredArray and JAX
    arrays.
    """
    shapes = [a.shape for a in arrays]
    shape = numpy.broadcast_shapes(*shapes)
    return [broadcast_to(a, shape, **kw) for a in arrays]
    # numpy.broadcast_arrays returns a list, not a tuple

class broadcast:
    """
    Version of numpy.broadcast that works with StructuredArray.
    """
    
    # not handled by __array_function__
    
    def __init__(self, *arrays):
        self.shape = numpy.broadcast_shapes(*(a.shape for a in arrays))

def asarray(x):
    """
    Version of numpy.asarray that works with StructuredArray and JAX arrays.
    If x is not a numpy array, returns a JAX array if possible.
    """
    # not handled by __array_function__
    if isinstance(x, (StructuredArray, jnp.ndarray, numpy.ndarray)):
        return x
    try:
        return jnp.asarray(x)
    except (TypeError, ValueError):
        return numpy.asarray(x)

@StructuredArray._implements(recfunctions.structured_to_unstructured)
def _structured_to_unstructured(arr, dtype=None, casting='unsafe'):
    mockup = numpy.empty(0, arr.dtype)
    dummy = recfunctions.structured_to_unstructured(mockup, dtype=dtype, casting=casting)
    args = (arr.shape + dummy.shape[-1:], dummy.dtype)
    try:
        out = jnp.empty(*args)
    except TypeError:
        out = numpy.empty(*args)
        # TODO can I make out column-major w.r.t. only the last column?
    out, length = arr._fill_unstructured(out)
    assert length == dummy.shape[-1]
    return out
