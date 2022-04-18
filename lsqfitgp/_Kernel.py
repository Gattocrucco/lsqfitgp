# lsqfitgp/_Kernel.py
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

import sys
import warnings
import copy

from ._imports import autograd
from ._imports import numpy as np
from ._imports import isinstance

from . import _array
from . import _Deriv

__all__ = [
    'Kernel',
    'IsotropicKernel',
    'kernel',
    'isotropickernel',
    'where'
]

def _asfloat(x):
    if not np.issubdtype(x.dtype, np.floating):
        return np.array(x, dtype=float, subok=True)
    else:
        return x

def _reduce_recurse_dtype(fun, *args, reductor=None, npreductor=None):
    x = args[0]
    if x.dtype.names is None:
        return fun(*args)
    else:
        acc = None
        for name in x.dtype.names:
            recargs = (arg[name] for arg in args)
            reckw = dict(reductor=reductor, npreductor=npreductor)
            result = _reduce_recurse_dtype(fun, *recargs, **reckw)
            
            dtype = x.dtype.fields[name][0]
            if dtype.shape:
                axis = tuple(range(-len(dtype.shape), 0))
                result = npreductor(result, axis=axis)
            
            if acc is None:
                acc = result
            else:
                acc = reductor(acc, result)
        
        assert acc.shape == _array.broadcast(*args).shape
        return acc

def _sum_recurse_dtype(fun, *args):
    plus = lambda a, b: a + b
    return _reduce_recurse_dtype(fun, *args, reductor=plus, npreductor=np.sum)

def _prod_recurse_dtype(fun, *args):
    times = lambda a, b: a * b
    return _reduce_recurse_dtype(fun, *args, reductor=times, npreductor=np.prod)

def _transf_recurse_dtype(transf, x):
    if x.dtype.names is None:
        return transf(x)
    else:
        x = _array.StructuredArray(x)
        for name in x.dtype.names:
            x[name] = _transf_recurse_dtype(transf, x[name])
        return x

class _KernelBase:
    
    # This class is only used to share implementation between Kernel and
    # _CrossKernel, so the docstrings are meant to be read as docstrings of
    # Kernel. The docstring is all in __init__ otherwise autoclass does not
    # read it.
    
    def __init__(self, kernel, *, dim=None, loc=None, scale=None, forcebroadcast=False, forcekron=False, derivable=False, **kw):
        """
        
        Base class for objects representing covariance kernels.
        
        A Kernel object is callable, the signature is obj(x, y). Kernel objects
        can be summed and multiplied between them and with scalars, or raised
        to power with a scalar exponent.
    
        Attributes
        ----------
        derivable : int
            How many times the kernel is derivable. ``sys.maxsize`` if it is
            smooth.
        
        Parameters
        ----------
        kernel : callable
            A function with signature ``kernel(x, y)``, where `x` and `y` are
            two broadcastable numpy arrays, which computes the covariance of
            f(x) with f(y) where f is the Gaussian process.
        dim : None or str
            When the input arrays are structured arrays, if `dim` is None the
            kernel will operate on all fields, i.e., it will be passed the whole
            arrays. If `dim` is a string, `kernel` will see only the arrays for
            the field named `dim`. If `dim` is a string and the array is not
            structured, an exception is raised. If the field for name `dim` has
            a nontrivial shape, the array passed to `kernel` is still
            structured but has only field `dim`.
        loc, scale : scalar
            The inputs to `kernel` are transformed as (x - loc) / scale.
        forcebroadcast : bool
            If True, the inputs to `kernel` will always have the same shape.
            Default is False.
        forcekron : bool
            If True, when calling `kernel`, if `x` and `y` are structured
            arrays, i.e., if they represent multidimensional input, `kernel` is
            invoked separately for each dimension, and the result is the
            product. Default False. If `dim` is specified, `forcekron` will
            have no effect.
        derivable : bool, int or callable
            Specifies how many times the kernel can be derived, just for
            error checking purposes. Default is False. True means infinitely
            many times derivable. If callable, it is called with the same
            keyword arguments of `kernel`.
        **kw
            Additional keyword arguments are passed to `kernel`.
        
        """
        # TODO linear transformation of input that works with arbitrarily
        # nested dtypes. Use an array/dictionary of arrays/dictionaries etc.
        # to represent a matrix. Dictionaries implicitly allow the
        # matrix to be sparse over fields. The transformation is applied prior
        # to selecting a field with `dim`.
        
        # TODO allow dim to select arbitrarily nested fields, and also sets of
        # fields and slices of array fields. It shall be similar to the Deriv
        # syntax (still not done).
        
        # TODO allow a list of dim
        
        # TODO if derivable is a callable and it returns another callable,
        # the second callable is called with input points to determine on
        # which points exactly the kernel is derivable. (But still returns
        # a single boolean for an array of points, equivalent to an `all`.)
        
        # Check simple arguments.
        assert isinstance(dim, (str, type(None)))
        self._forcebroadcast = bool(forcebroadcast)
        forcekron = bool(forcekron)
        
        # Convert `derivable` to an integer.
        if callable(derivable):
            derivable = derivable(**kw)
        if isinstance(derivable, bool):
            derivable = sys.maxsize if derivable else 0
        elif isinstance(derivable, (int, np.integer)):
            assert derivable >= 0
        elif derivable:
            derivable = sys.maxsize
        else:
            derivable = 0
        self._minderivable = (derivable, derivable)
        self._maxderivable = (derivable, derivable)
        
        transf = lambda x: x
        
        if isinstance(dim, str):
            def transf(x):
                if x.dtype.names is None:
                    raise ValueError(f'kernel called on non-structured array but dim="{dim}"')
                elif x.dtype.fields[dim][0].shape:
                    return x[[dim]]
                else:
                    return x[dim]
        
        if loc is not None:
            assert np.isscalar(loc)
            assert np.isfinite(loc)
            transf1 = transf
            transf = lambda x: _transf_recurse_dtype(lambda x: x - loc, transf1(x))
        
        if scale is not None:
            assert np.isscalar(scale)
            assert np.isfinite(scale)
            assert scale > 0
            transf2 = transf
            transf = lambda x: _transf_recurse_dtype(lambda x: x / scale, transf2(x))
        
        # TODO when dim becomes deep, forcekron must apply also to subfields
        # for consistence. Maybe it should do it now already.
        if dim is None and forcekron:
            def _kernel(x, y):
                x = transf(x)
                y = transf(y)
                fun = lambda x, y: kernel(x, y, **kw)
                return _prod_recurse_dtype(fun, x, y)
        else:
            _kernel = lambda x, y: kernel(transf(x), transf(y), **kw)
        
        self._kernel = _kernel
    
    def __call__(self, x, y):
        x = _array.asarray(x)
        y = _array.asarray(y)
        np.result_type(x.dtype, y.dtype)
        shape = _array.broadcast(x, y).shape
        if self._forcebroadcast:
            x, y = _array.broadcast_arrays(x, y)
        result = self._kernel(x, y)
        assert isinstance(result, (np.ndarray, np.number))
        assert np.issubdtype(result.dtype, np.number), result.dtype
        assert result.shape == shape
        return result
    
    def _binary(self, value, op):
        if isinstance(value, _KernelBase):
            obj = _KernelBase(op(self._kernel, value._kernel))
            obj._minderivable = tuple(np.minimum(self._minderivable, value._minderivable))
            obj._maxderivable = tuple(np.maximum(self._maxderivable, value._maxderivable))
            obj._forcebroadcast = self._forcebroadcast or value._forcebroadcast
        elif np.isscalar(value):
            assert np.isfinite(value), value
            obj = _KernelBase(op(self._kernel, lambda x, y: value))
            obj._minderivable = self._minderivable
            obj._maxderivable = self._maxderivable
            obj._forcebroadcast = self._forcebroadcast
        else:
            obj = NotImplemented
        return obj
    
    # TODO when using autograd, an autograd scalar is an ArrayBox that
    # has an all-in method __add__ that will be called if the autograd scalar
    # is the first addend. Then autograd will complain that it can't compute
    # derivatives w.r.t. a Kernel. Solve this bug.
    
    def __add__(self, value):
        return self._binary(value, lambda k, q: lambda x, y: k(x, y) + q(x, y))
    
    __radd__ = __add__
    
    def __mul__(self, value):
        return self._binary(value, lambda k, q: lambda x, y: k(x, y) * q(x, y))
    
    __rmul__ = __mul__
    
    def __pow__(self, value):
        if np.isscalar(value):
            return self._binary(value, lambda k, q: lambda x, y: k(x, y) ** q(x, y))
        else:
            return NotImplemented

    def _swap(self):
        """permute the arguments (cross kernels are not symmetric)"""
        obj = copy.copy(self)
        kernel = obj._kernel
        obj._kernel = lambda x, y: kernel(y, x)
        obj._minderivable = obj._minderivable[::-1]
        obj._maxderivable = obj._maxderivable[::-1]
        return obj
    
    def rescale(self, xfun=None, yfun=None):
        """
        
        Multiply the kernel by functions of its arguments.
        
        .. math::
            h(x, y) = f(x) k(x, y) g(y)
        
        Parameters
        ----------
        xfun, yfun : callable or None
            Functions from the type of the arguments of the kernel to scalar.
            If both are None, this is a no-op.
        
        Returns
        -------
        h : Kernel-like
            The rescaled kernel. If xfun is yfun, it is a Kernel object,
            otherwise a Kernel-like one.
        
        """
        
        # TODO option to specify derivability of xfun and yfun, to avoid
        # zeroing _minderivable
        
        if xfun is None and xfun is None:
            return self
        
        kernel = self._kernel
        
        if xfun is None:
            def fun(x, y):
                return yfun(y) * kernel(x, y)
        elif yfun is None:
            def fun(x, y):
                return xfun(x) * kernel(x, y)
        else:
            def fun(x, y):
                return xfun(x) * yfun(y) * kernel(x, y)
        
        cls = Kernel if xfun is yfun and isinstance(self, Kernel) else _CrossKernel
        obj = cls(fun, forcebroadcast=self._forcebroadcast)
        obj._maxderivable = self._maxderivable
        return obj
    
    def diff(self, xderiv, yderiv):
        """
        
        Return a Kernel-like object that computes the derivatives of this
        kernel. The derivatives are computed automatically with autograd. If
        `xderiv` and `yderiv` are trivial, this is a no-op.
        
        Parameters
        ----------
        xderiv, yderiv : Deriv-like
            A :class:`Deriv` object or something that can be converted to a
            Deriv object.
        
        Returns
        -------
        Kernel-like
            An object representing the derivatives of this one. If ``xderiv ==
            yderiv``, it is actually another Kernel.
        
        Raises
        ------
        RuntimeError
            The derivative orders are greater than the `derivative` attribute.
            
        """
        
        # TODO maximum dimensionality should be checked too, see PPKernel. But
        # this would probably fall well under the double callable derivative
        # case.
        
        # TODO to check the derivability precisely, I would need to make a new
        # class Derivable (somewhat similar to Deriv) that has two separate
        # counters, `other` and `all`. `all` decrements each time you take a
        # derivate (w.r.t. any variable) while `other` is the assumed starting
        # point for variables which still don't have their own counter, which is
        # copied from `other` and then decremented when deriving w.r.t. that
        # variable. However this still lacks generality: you would need to also
        # have groups of variables which behave like `all` or `other` but only
        # within the group. For example, `all` corresponds to the behavior of
        # isotropic kernels, `other` to separable ones. And there is also the
        # case where the user gives a custom nontrivial derivability
        # specification and then applies the kernel to a subfield, which would
        # require to prefix the subfield to all variable (group) specifications.
        
        xderiv = _Deriv.Deriv(xderiv)
        yderiv = _Deriv.Deriv(yderiv)
        
        if not xderiv and not yderiv:
            return self
        
        # Check kernel is derivable.
        
        # best case: max derivability + only single variable order matters
        maxs   = (  xderiv.max,   yderiv.max)
        if any(  maxs[i] > self._maxderivable[i] for i in range(2)):
            raise RuntimeError(f'maximum single-variable derivative orders {maxs} greater than kernel maximum {self._maxderivable}')
        
        # worst case: min derivability + total derivation order matters
        orders = (xderiv.order, yderiv.order)
        if any(orders[i] > self._minderivable[i] for i in range(2)):
            warnings.warn(f'total derivative orders {orders} greater than kernel minimum {self._minderivable}')
        
        # Check derivatives are ok for x and y.
        def check(x, y):
            if x.dtype.names is not None:
                for deriv in xderiv, yderiv:
                    for dim in deriv:
                        if dim not in x.dtype.names:
                            raise ValueError(f'derivative along missing field {dim!r}')
                        if not np.issubdtype(x.dtype.fields[dim][0], np.number):
                            raise TypeError(f'derivative along non-numeric field {dim!r}')
            elif not xderiv.implicit or not yderiv.implicit:
                raise ValueError('explicit derivatives with non-structured array')
                
        # Handle the non-structured case.
        if xderiv.implicit and yderiv.implicit:
            
            f = self._kernel
            for _ in range(xderiv.order):
                f = autograd.elementwise_grad(f, 0)
            for _ in range(yderiv.order):
                f = autograd.elementwise_grad(f, 1)
            
            def fun(x, y):
                check(x, y)
                if xderiv:
                    x = _asfloat(x)
                if yderiv:
                    y = _asfloat(y)
                return f(x, y)
        
        # Structured case.
        else:
            
            # Wrap of kernel with derivable arguments only.
            kernel = self._kernel
            def f(x, y, *args):
                i = -1
                for i, dim in enumerate(xderiv):
                    x[dim] = args[i]
                for j, dim in enumerate(yderiv):
                    y[dim] = args[1 + i + j]
                return kernel(x, y)
                
            # Make derivatives.
            i = -1
            for i, dim in enumerate(xderiv):
                for _ in range(xderiv[dim]):
                    f = autograd.elementwise_grad(f, 2 + i)
            for j, dim in enumerate(yderiv):
                for _ in range(yderiv[dim]):
                    f = autograd.elementwise_grad(f, 2 + 1 + i + j)
            
            def fun(x, y):
                check(x, y)
                                
                # Autograd-friendly wrap of structured arrays.
                if xderiv:
                    x = _array.StructuredArray(x)
                if yderiv:
                    y = _array.StructuredArray(y)
            
                # Make argument list and call function.
                args = []
                for dim in xderiv:
                    args.append(_asfloat(x[dim]))
                for dim in yderiv:
                    args.append(_asfloat(y[dim]))
                return f(x, y, *args)
        
        cls = Kernel if xderiv == yderiv and isinstance(self, Kernel) else _CrossKernel
        obj = cls(fun, forcebroadcast=True)
        obj._minderivable = tuple(self._minderivable[i] - orders[i] for i in range(2))
        obj._maxderivable = tuple(self._maxderivable[i] -   maxs[i] for i in range(2))
        return obj

class _CrossKernel(_KernelBase):
    pass
    
class Kernel(_KernelBase):
    
    @property
    def derivable(self):
        assert self._minderivable[0] == self._minderivable[1]
        assert self._maxderivable[0] == self._maxderivable[1]
        if self._minderivable == self._maxderivable:
            return self._minderivable[0]
        else:
            return None
        
    # TODO if I implement double callable derivable, derivable should always be
    # a callable, and _binary should propagate it properly (with an `and`?).
    
    def _binary(self, value, op):
        # redefine _KernelBase._binary to forbid Kernel-_CrossKernel operations
        # and multiplication by negative scalar
        if isinstance(value, Kernel):
            obj = super(Kernel, self)._binary(value, op)
            obj.__class__ = Kernel
        elif np.isscalar(value):
            assert value >= 0, value
            obj = super(Kernel, self)._binary(value, op)
            obj.__class__ = Kernel
        else:
            obj = NotImplemented
        return obj
    
# TODO Add a class StationaryKernel. It should not be a superclass of
# IsotropicKernel because that holds only for the distances based on the
# difference. Or maybe I should fix that IsotropicKernel is for distances
# based on the difference. Or maybe generalize the concept of difference in
# StationaryKernel, so that it is reasonable that IsotropicKernel has a
# general concept of distance. Question: the known isotropic kernels
# effectively work for any distance, or the proofs are only for the 2-norm?

class IsotropicKernel(Kernel):
    
    # TODO add the `distance` parameter to supply an arbitrary distance, maybe
    # allow string keywords for premade distances, like euclidean, hamming,
    # p-norms.
    
    # TODO it is not efficient that the distance is computed separately for
    # each kernel in a kernel expression, but probably it would be difficult
    # to support everything without bugs while also computing the distance once.
    # A possible way is adding a keyword argument to the _kernel member
    # that kernels use to memoize things, the first IsotropicKernel that gets
    # called puts the distance there. Possible name: _cache.
    
    # TODO the scale parameter is specified to divide the distance, which is
    # currently equivalent to dividing the values. If I introduced
    # inhomogeneous distances, would it be better to divide the values or the
    # distance?
    
    def __init__(self, kernel, *, input='squared', scale=None, **kw):
        """
        
        Subclass of :class:`Kernel` for isotropic kernels.
    
        Parameters
        ----------
        kernel : callable
            A function taking one argument `r2` which is the squared distance
            between x and y, plus optionally keyword arguments.
        input : {'squared', 'soft'}
            If 'squared' (default), `kernel` is passed the squared distance.
            If 'soft', `kernel` is passed the distance, and the distance of
            equal points is a small number instead of zero.
        scale : scalar
            The distance is divided by `scale`.
        **kw
            Additional keyword arguments are passed to the :class:`Kernel` init.
                
        """
        allowed_input = ('squared', 'soft')
        if input not in allowed_input:
            raise ValueError('input option {!r} not valid, must be one of {!r}'.format(input, allowed_input))
        if input == 'soft':
            func = lambda x, y: _softabs(x - y) ** 2
        else:
            func = lambda x, y: (x - y) ** 2
        
        transf = lambda q: q
        if scale is not None:
            assert np.isscalar(scale)
            assert np.isfinite(scale)
            assert scale > 0
            transf = lambda q : q / scale ** 2
        if input == 'soft':
            transf0 = transf
            transf = lambda q: np.sqrt(transf0(q))  
        
        def function(x, y, **kwargs):
            q = _sum_recurse_dtype(func, x, y)
            return kernel(transf(q), **kwargs)
        
        super().__init__(function, **kw)
    
    def _binary(self, value, op):
        
        # TODO this logic could be made generic and moved to Kernel, since
        # any subclass I can think of forms a subalgebra.
        
        obj = super()._binary(value, op)
        if isinstance(obj, Kernel) and isinstance(value, __class__):
            obj.__class__ = __class__
        return obj

def _eps(x):
    if np.issubdtype(x.dtype, np.inexact):
        return np.finfo(x.dtype).eps
    else:
        return np.finfo(float).eps

# I define my own version of abs because autograd defines the derivative in 0
# to be 0, while I need it to be 1 for stationary/isotropic kernels.
@autograd.extend.primitive
def _abs(x):
    return np.abs(x)

autograd.extend.defvjp(
    _abs,
    lambda ans, x: lambda g: g * np.where(x >= 0, 1, -1)
)

autograd.extend.defjvp(
    _abs,
    lambda g, ans, x: (g.T * np.where(x >= 0, 1, -1).T).T
)

def _softabs(x):
    return _abs(x) + _eps(x)

def _makekernelsubclass(kernel, superclass, **prekw):
    assert issubclass(superclass, Kernel)
    
    if hasattr(kernel, 'pyfunc'): # np.vectorize objects
        named_object = kernel.pyfunc
    else:
        named_object = kernel
    
    name = getattr(named_object, '__name__', 'DecoratedKernel')
    newclass = type(name, (superclass,), {})
    
    prekwset = set(prekw)
    def __init__(self, **kw):
        kwargs = prekw.copy()
        shared_keys = prekwset & set(kw)
        if shared_keys:
            msg = 'overriding init argument(s) ' + ', '.join(shared_keys)
            msg += ' of kernel ' + name
            warnings.warn(msg)
        kwargs.update(kw)
        super(newclass, self).__init__(kernel, **kwargs)
    
    newclass.__init__ = __init__
    newclass.__wrapped__ = named_object
    newclass.__doc__ = named_object.__doc__
    newclass.__qualname__ = getattr(named_object, '__qualname__', name)
    
    # TODO functools.wraps raised an error, but maybe functools.update_wrapper
    # with appropriate arguments would work even on a class.
    
    return newclass

def _kerneldecoratorimpl(cls, *args, **kw):
    functional = lambda kernel: _makekernelsubclass(kernel, cls, **kw)
    if len(args) == 0:
        return functional
    elif len(args) == 1:
        return functional(*args)
    else:
        raise ValueError(len(args))

def kernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of :class:`Kernel`. Example::
    
        @kernel(loc=10) # the default loc will be 10
        def MyKernel(x, y, cippa=1, lippa=42):
            return cippa * (x * y) ** lippa
    
    """
    return _kerneldecoratorimpl(Kernel, *args, **kw)

def isotropickernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of :class:`IsotropicKernel`.
    Example::
    
        @isotropickernel(derivable=True)
        def MyKernel(distsquared, cippa=1, lippa=42):
            return cippa * np.exp(-distsquared) + lippa
    
    """
    return _kerneldecoratorimpl(IsotropicKernel, *args, **kw)

def where(condfun, kernel1, kernel2, dim=None):
    """
    
    Make a kernel(x, y) that yields:
    
      * kernel1(x, y) where condfun(x) and condfun(y) are True
    
      * kernel2(x, y) where condfun(x) and condfun(y) are False
    
      * zero where condfun(x) is different from condfun(y)
    
    Parameters
    ----------
    condfun : callable
        Function that is applied on an array of points and must return
        a boolean array with the same shape.
    kernel1 : Kernel
        Kernel used where condfun yields True.
    kernel2 : Kernel
        Kernel used where condfun yields False.
    dim : str or None
        If specified, when the input arrays are structured, `condfun` is
        applied only to the field `dim`. If the field has a shape, the
        array passed to `condfun` still has `dim` as explicit field.
    
    Returns
    -------
    Kernel
        If both kernel1 and kernel2 are IsotropicKernel, the class is
        IsotropicKernel.
    
    """
    assert isinstance(kernel1, Kernel)
    assert isinstance(kernel2, Kernel)
    assert callable(condfun)
    
    assert isinstance(dim, (str, type(None)))
    if isinstance(dim, str):
        def transf(x):
            if x.dtype.names is None:
                raise ValueError('kernel called on non-structured array but condition dim="{}"'.format(dim))
            elif x.dtype.fields[dim][0].shape:
                # TODO should probably use subdtype, such that when the user
                # explicitly specifies an empty shape the behaviour is the
                # same as with nontrivial shapes. This applies also to Kernel.
                return x[[dim]]
            else:
                return x[dim]
        condfun0 = condfun
        condfun = lambda x: condfun0(transf(x))
    
    def kernel_op(k1, k2):
        def kernel(x, y):
            # TODO this is inefficient, kernels should be computed only on
            # the relevant points. To support this with autograd, make a
            # custom np.where that uses assignment and define its vjp.
            
            # TODO this may produce a very sparse matrix, when I implement
            # sparse support do it here too.
            
            # TODO it will probably often be the case that the result is
            # either dense or all zero. The latter case can be optimized this
            # way: if it's zero, broadcast a 0-d array to the required shape,
            # and flag it as all zero with an instance variable.
            
            xcond = condfun(x)
            ycond = condfun(y)
            r = np.where(xcond & ycond, k1(x, y), k2(x, y))
            return np.where(xcond ^ ycond, 0, r)
        return kernel
    
    # TODO when I implement double callable derivable, propagate it
    # properly by overwriting it in the returned object since _binary will
    # require both kernels to be derivable on a given point.
    
    return kernel1._binary(kernel2, kernel_op)

# TODO add a function `choose` to extend `where`. Interface:
# choose(keyfun, mapping)
# example where `comp` is an integer field selecting the kernel:
# choose(lambda comp: comp, [kernel0, kernel1, kernel2, ...], dim='comp')
# example where `comp` is a string field, and without using `dim`:
# choose(lambda x: x['comp'], {'a': kernela, 'b': kernelb})
# define Kernel._nary using a cycle of _binary
