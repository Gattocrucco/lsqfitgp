# lsqfitgp/_array.py
#
# Copyright (c) 2020, 2022, 2023, 2024, Giacomo Petrillo
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

import textwrap
import math

import numpy
from numpy.lib import recfunctions
import jax
from jax import numpy as jnp
from jax import tree_util

# TODO use register_pytree_with_keys
@tree_util.register_pytree_node_class
class StructuredArray:
    """
    JAX-friendly imitation of a numpy structured array.
    
    It behaves like a read-only numpy structured array, and you can create
    a copy with a modified field with a jax-like syntax.
    
    Examples
    --------
    >>> a = numpy.empty(3, dtype=[('f', float), ('g', float)])
    >>> a = StructuredArray(a)
    >>> a = a.at['f'].set(numpy.arange(3))
    ... # is equivalent to a['f'] = numpy.arange(3)
        
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
    def _readonlyview_wrapifstructured(cls, x):
        if x.dtype.names is not None:
            x = cls(x)
        if isinstance(x, numpy.ndarray):
            x = x.view()
            x.flags.writeable = False
            # jax arrays and StructuredArrays are already readonly
        return x

    @classmethod
    def _array(cls, s, t, d, *, check=True):
        """
        Create a new StructuredArray.

        All methods and functions that create a new StructuredArray object
        should use this method.

        Parameters
        ----------
        s : tuple or None
            The shape. If None, it is determined automatically from the arrays.
        t : dtype or None
            The dtype of the array. If None, it is determined automatically
            (before the shape).
        d : dict str -> array
            The _dict of the array. The arrays, if structured, must be already
            StructuredArrays. The order of the keys must match the order of the
            fields.
        check : bool
            If True (default), check the passed values are consistent.

        Return
        ------
        out : StructuredArray
            A new StructuredArray object.
        """

        if t is None:
            # infer the data type from the arrays in the dictionary
            ndim = min((x.ndim for x in d.values()), default=None)
            t = numpy.dtype([
                (name, x.dtype, x.shape[ndim:])
                for name, x in d.items()
            ])
            # TODO infer the least common head shape instead of counting dims

        # remove offset info since this is actually a columnar format
        t = recfunctions.repack_fields(t, align=False, recurse=True)

        if s is None:
            # infer the shape from the arrays in the dictionary
            assert d, 'can not infer array shape with no fields'
            f = t.names[0]
            a = d[f]
            s = a.shape[:a.ndim - t[0].ndim]
        
        if check:
            assert len(t) == len(d)
            assert t.names == tuple(d.keys())
            assert all(
                x.dtype == t[f].base and x.ndim >= t[f].ndim
                for f, x in d.items()
            )
            shapes = [
                x.shape[:x.ndim - t[f].ndim]
                for f, x in d.items()
            ]
            assert all(s == s1 for s1 in shapes)
        
        out = super().__new__(cls)
        out.shape = s
        out.dtype = t
        out._dict = d
        
        return out
    
    def __new__(cls, array):
        if isinstance(array, cls):
            return array
        d = {
            name: cls._readonlyview_wrapifstructured(array[name])
            for name in array.dtype.names
        }
        return cls._array(array.shape, array.dtype, d)
        
    @classmethod
    def from_dataframe(cls, df):
        """
        Make a StructuredArray from a DataFrame. Data is not copied if not
        necessary.
        """
        d = {
            col: cls._readonlyview_wrapifstructured(df[col].to_numpy())
            for col in df.columns
        }
        return cls._array(None, None, d)
        # TODO support polars structured dtypes
        # TODO polars has a parameter Series.to_numpy(zero_copy_only: bool),
        # default False. Maybe make it accessible through kw or options.

    @classmethod
    def from_dict(cls, mapping):
        """
        Make a StructuredArray from a dictionary of arrays. Data is not copied.
        """
        d = {
            name: cls._readonlyview_wrapifstructured(value)
            for name, value in mapping.items()
        }
        return cls._array(None, None, d)
    
    @property
    def size(self):
        return math.prod(self.shape)
    
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._dict.values())

    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.swapaxes(self.ndim - 2, self.ndim - 1)

        # TODO this is mT, not T! Make a unit test and correct it.

    def swapaxes(self, i, j):
        shape = jax.eval_shape(lambda: jnp.empty(self.shape).swapaxes(i, j)).shape
        d = {k: v.swapaxes(i, j) for k, v in self._dict.items()}
        return self._array(shape, self.dtype, d)

        # TODO: doesn't this break when the indices are negative and there is
        # an array field? Test it.

    def __len__(self):
        if self.shape:
            return self.shape[0]
        else:
            raise TypeError('len() of unsized object')
    
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
                    + (slice(None),) * self.dtype[name].ndim
                ]
                for name, x in self._dict.items()
            }
            shape = jax.eval_shape(lambda: jnp.empty(self.shape)[key]).shape
            return self._array(shape, self.dtype, d)
    
    @property
    def at(self):
        return self._Getter(self)
    
    class _Getter:
        
        def __init__(self, array):
            self.array = array
        
        def __getitem__(self, key):
            if key not in self.array.dtype.names:
                raise KeyError(key)
            return self.Setter(self.array, key)
        
        class Setter:
            
            def __init__(self, array, key, parent=None):
                self.array = array
                self.key = key
                self.parent = parent

            def __getitem__(self, subkey):
                if subkey not in self.array.dtype[self.key].names:
                    raise KeyError(subkey)
                return self.__class__(self.array[self.key], subkey, self)
            
            def set(self, val):
                assert isinstance(val, (numpy.ndarray, jnp.ndarray, StructuredArray))
                prev = self.array._dict[self.key]
                # TODO support casting and broadcasting
                assert prev.dtype == val.dtype
                assert prev.shape == val.shape
                d = dict(self.array._dict)
                d[self.key] = self.array._readonlyview_wrapifstructured(val)
                out = self.array._array(self.array.shape, self.array.dtype, d)
                if self.parent:
                    return self.parent.set(out)
                else:
                    return out
    
    def reshape(self, *shape):
        """
        Reshape the array without changing its contents. See
        numpy.ndarray.reshape.
        """
        if len(shape) == 1 and hasattr(shape[0], '__len__'):
            shape = shape[0]
        shape = tuple(shape)
        d = {
            name: x.reshape(shape + self.dtype[name].shape)
            for name, x in self._dict.items()
        }
        shape = numpy.empty(self.shape, []).reshape(shape).shape
        return self._array(shape, self.dtype, d)

    def squeeze(self, axis=None):
        """
        Remove axes of length 1. See numpy.ndarray.squeeze.
        """
        if axis is None:
            axis = tuple(i for i, size in enumerate(self.shape) if size == 1)
        if not hasattr(axis, '__len__'):
            axis = (axis,)
        assert all(self.shape[i] == 1 for i in axis)
        newshape = [size for i, size in enumerate(self.shape) if i not in axis]
        return self.reshape(newshape)

    def astype(self, dtype):
        if dtype != self.dtype:
            raise NotImplementedError
        return self
    
    def broadcast_to(self, shape, **kw):
        """
        Return a view of the array broadcasted to another shape. See
        numpy.broadcast_to.
        """
        # raises if not broadcastable
        numpy.broadcast_to(numpy.empty(self.shape, []), shape, **kw)
        d = {
            name: broadcast_to(x, shape + self.dtype[name].shape, **kw)
            for name, x in self._dict.items()
        }
        return self._array(shape, self.dtype, d)
    
    # TODO implement flatten_with_keys
    def tree_flatten(self):
        """ JAX PyTree encoder. See `jax.tree_util.tree_flatten`. """
        children = tuple(self._dict[key] for key in self.dtype.names)
        aux = dict(shape=self.shape, dtype=self.dtype)
        return children, aux
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        """ JAX PyTree decoder. See `jax.tree_util.tree_unflatten`. """

        # if there are no fields, keep original shape
        if not children:
            return cls._array(aux['shape'], aux['dtype'], {})
        
        # convert children to arrays because tree_util.tree_flatten unpacks 0d
        # arrays
        children = list(map(asarray, children))
        
        # if possible, keep original dtype shapes
        oldtype = aux['dtype']
        compatible_tail_shapes = all(
            x.shape[max(0, x.ndim - oldtype[i].ndim):] == oldtype[i].shape
            for i, x in enumerate(children)
        )
        head_shapes = [
            x.shape[:max(0, x.ndim - oldtype[i].ndim)]
            for i, x in enumerate(children)
        ]
        compatible_head_shapes = all(head_shapes[0] == s for s in head_shapes)
        if compatible_tail_shapes and compatible_head_shapes:
            dtype = numpy.dtype([
                (oldtype.names[i], x.dtype, oldtype[i].shape)
                for i, x in enumerate(children)
            ])
        else:
            dtype = None

        d = dict(zip(oldtype.names, children))

        return cls._array(None, dtype, d)

        # TODO this breaks jax.jit(...).lower(...).compile()(...) because
        # apparently `lower` saves the pytree def after a step of dummyfication,
        # so the shape and dtype bases of the StructuredArray are () and object.
        # JAX expects pytrees to have a structure which does not depend on what
        # they store. => Quick hack: preserve the shape and dtype
        # unconditionally, i.e., tree_unflatten can produce malformed
        # StructuredArrays. The dictionary will contain whatever JAX puts into
        # it. => Quicker hack: it seems to me that jax always uses None as
        # dummy, so I could detect if all childrens are None or StructuredArray.

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

        # TODO try simply using the __repr__ of self._dict
    
    def __array__(self, copy=None, dtype=None):
        if copy is False:
            raise ValueError('StructuredArray has to be copied when converted to a numpy array')
        if dtype is not None:
            dtype = numpy.dtype(dtype)
            if dtype != self.dtype:
                raise ValueError('StructuredArray can not be converted to a numpy array with a different dtype')
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
        """ Register an __array_function__ implementation """
        def decorator(func):
            cls._handled_functions[np_function] = func
            newdoc = f"""\
Implementation of `{np_function.__module__}.{np_function.__name__}` for `StructuredArray`.

"""
            if func.__doc__:
                newdoc += textwrap.dedent(func.__doc__) + '\n'
            newdoc += 'Original docstring below:\n\n'
            newdoc += textwrap.dedent(np_function.__doc__)
            func.__doc__ = newdoc
            return func
        return decorator

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

def asarray(x, dtype=None):
    """
    Version of `numpy.asarray` that works with `StructuredArray` and JAX arrays.
    If `x` is not an array already, returns a JAX array if possible.
    """
    if isinstance(x, (StructuredArray, jnp.ndarray, numpy.ndarray)):
        return x if dtype is None else x.astype(dtype)
    if x is None:
        return numpy.asarray(x, dtype)
        # partial workaround for jax issue #14506, None would be interpreted as
        # nan by jax
    try:
        return jnp.asarray(x, dtype)
    except (TypeError, ValueError):
        return numpy.asarray(x, dtype)

def _asarray_jaxifpossible(x):
    x = asarray(x)
    if x.dtype.names:
        return tree_util.tree_map(_asarray_jaxifpossible, StructuredArray(x))
    if isinstance(x, numpy.ndarray):
        try:
            return jnp.asarray(x)
        except (TypeError, ValueError):
            pass
    return x

@StructuredArray._implements(numpy.squeeze)
def _squeeze(a, axis=None):
    return a.squeeze(axis)

@StructuredArray._implements(numpy.ix_)
def _ix(*args):
    args = tuple(map(asarray, args))
    assert all(x.ndim == 1 for x in args)
    n = len(args)
    return tuple(
        x.reshape((1,) * i + (-1,) + (1,) * (n - i - 1))
        for i, x in enumerate(args)
    )

def unstructured_to_structured(arr,
    dtype=None,
    names=None,
    align=False, # TODO maybe align is totally inapplicable even with numpy arrays? What does it mean?
    copy=False,
    casting='unsafe'):
    """ Like `numpy.lib.recfunctions.unstructured_to_structured`, but outputs a
    `StructuredArray`. """
    arr = asarray(arr)
    if not arr.ndim:
        raise ValueError('arr must have at least one dimension')
    mockup = numpy.empty((0,) + arr.shape[-1:], arr.dtype)
    dummy = recfunctions.unstructured_to_structured(mockup,
        dtype=dtype, names=names, align=align, copy=copy, casting=casting)
    out, length = _unstructured_to_structured_recursive(0, (), arr, dummy.dtype, copy, casting)
    assert length == arr.shape[-1]
    return out

def _unstructured_to_structured_recursive(idx, shape, arr, dtype, copy, casting, *strides):
    arrays = {}
    for i, name in enumerate(dtype.names):
        base = dtype[i].base
        subshape = shape + dtype[i].shape
        size = math.prod(dtype[i].shape)
        stride = _nd(base)
        substrides = strides + ((size, stride),)
        if base.names is not None:
            y, newidx = _unstructured_to_structured_recursive(idx, subshape, arr, base, copy, casting, *substrides)
            shift = newidx - idx
            assert shift == stride
            idx += size * stride
        else:
            assert stride == 1
            if all(size == 1 for size, _ in strides):
                indices = numpy.s_[idx:idx + size]
                srcsize = size
            else:
                indices = sum((
                    stride * numpy.arange(size)[numpy.s_[:,] + (None,) * i]
                    for i, (size, stride) in enumerate(reversed(substrides))
                ), start=idx)
                indices = indices.reshape(-1)
                srcsize = indices.size
            key = numpy.s_[..., indices]
            x = arr[key]
            x = x.reshape(arr.shape[:-1] + subshape)
            if isinstance(x, jnp.ndarray):
                y = x.astype(base)
            else:
                y = x.astype(base, copy=copy, casting=casting)
            idx += size
        arrays[name] = y
    return StructuredArray._array(arr.shape[:-1] + shape, dtype, arrays), idx

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
    out, length = _structured_to_unstructured_recursive(0, arr, out)
    assert length == dummy.shape[-1]
    return out

def _nd(dtype):
    """ Count the number of scalars in a dtype """
    base = dtype.base
    shape = dtype.shape
    size = math.prod(shape)
    if base.names is None:
        return size
    else:
        return size * sum(_nd(base[name]) for name in base.names)

    # I use this function in many parts of the package so it should not have an
    # underscore, even if I don't export it in the main namespace. And move it
    # to utils, it's not specific to StructuredArray.

def _structured_to_unstructured_recursive(idx, arr, out, *strides):
    dtype = arr.dtype
    for i, name in enumerate(dtype.names):
        subarr = arr[name]
        base = dtype[i].base
        size = math.prod(dtype[i].shape)
        stride = _nd(base)
        substrides = strides + ((size, stride),)
        if base.names is not None:
            out, newidx = _structured_to_unstructured_recursive(idx, subarr, out, *substrides)
            shift = newidx - idx
            assert shift == stride
            idx += size * stride
        else:
            assert stride == 1
            if all(size == 1 for size, _ in strides):
                indices = numpy.s_[idx:idx + size]
                srcsize = size
            else:
                indices = sum((
                    stride * numpy.arange(size)[numpy.s_[:,] + (None,) * i]
                    for i, (size, stride) in enumerate(reversed(substrides))
                ), start=idx)
                indices = indices.reshape(-1)
                srcsize = indices.size
            key = numpy.s_[..., indices]
            src = subarr.reshape(out.shape[:-1] + (srcsize,))
            if hasattr(out, 'at'):
                out = out.at[key].set(src)
            else:
                out[key] = src
            idx += size
    return out, idx

@StructuredArray._implements(numpy.empty_like)
def _empty_like(prototype, dtype=None, *, shape=None):
    shape = prototype.shape if shape is None else shape
    dtype = prototype.dtype if dtype is None else dtype
    return _empty(shape, dtype)

@StructuredArray._implements(numpy.empty)
def _empty(shape, dtype=float):
    if hasattr(shape, '__len__'):
        shape = tuple(shape)
    else:
        shape = (int(shape),)
    dtype = numpy.dtype(dtype)
    arrays = {}
    for i, name in enumerate(dtype.names):
        dtbase = dtype[i].base
        dtshape = shape + dtype[i].shape
        if dtbase.names is not None:
            y = _empty(dtshape, dtbase)
        else:
            try:
                y = jnp.empty(dtshape, dtbase)
            except TypeError:
                y = numpy.empty(dtshape, dtbase)
        arrays[name] = y
    return StructuredArray._array(shape, dtype, arrays)

@StructuredArray._implements(numpy.concatenate)
def _concatenate(arrays, axis=0, dtype=None, casting='same_kind'):

    # checks arrays is a non-empty sequence
    arrays = list(arrays)
    if not arrays:
        raise ValueError('need at least one array to concatenate')

    # parse axis argument
    if axis is None:
        axis = 0
        arrays = [a.reshape(-1) for a in arrays]
    else:
        ndim = arrays[0].ndim
        assert all(a.ndim == ndim for a in arrays)
        assert -ndim <= axis < ndim
        axis %= ndim
        shape = arrays[0].shape
        assert all(a.shape[:axis] == shape[:axis] for a in arrays)
        assert all(a.shape[axis + 1:] == shape[axis + 1:] for a in arrays)

    dtype = numpy.result_type(*(a.dtype for a in arrays))
    assert all(numpy.can_cast(a.dtype, dtype, casting) for a in arrays)
    shape = (
        *arrays[0].shape[:axis],
        sum(a.shape[axis] for a in arrays),
        *arrays[0].shape[axis + 1:],
    )

    out = _concatenate_recursive(arrays, axis, dtype, shape, casting)
    assert out.shape == shape and out.dtype == dtype
    return out

def _concatenate_recursive(arrays, axis, dtype, shape, casting):
    cat = {}
    for name in dtype.names:
        subarrays = [a[name] for a in arrays]
        base = dtype[name].base
        if base.names is not None:
            subshape = shape + dtype[name].shape
            y = _concatenate_recursive(subarrays, axis, base, subshape, casting)
        else:
            try:
                y = jnp.concatenate(subarrays, axis=axis, dtype=base)
            except TypeError:
                y = numpy.concatenate(subarrays, axis=axis, dtype=base, casting=casting)
        cat[name] = y
    return StructuredArray._array(shape, dtype, cat)

@StructuredArray._implements(recfunctions.append_fields)
def _append_fields(base, names, data, usemask=True):
    assert not usemask, 'masked arrays not supported, set usemask=False'
    if isinstance(names, str):
        names = [names]
        data = [data]
    assert len(names) == len(data)
    arrays = base._dict.copy()
    arrays.update(zip(names, data))
    dtype = numpy.dtype(base.dtype.descr + [
        (name, array.dtype) for name, array in zip(names, data)
    ])
    return StructuredArray._array(base.shape, dtype, arrays)

@StructuredArray._implements(numpy.swapaxes)
def _swapaxes(x, i, j):
    return x.swapaxes(i, j)
