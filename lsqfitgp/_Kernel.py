from __future__ import division

import sys

import autograd
from autograd import numpy as np
from autograd.builtins import isinstance

from . import _array
from . import _Deriv

__all__ = [
    'Kernel',
    'IsotropicKernel',
    'kernel',
    'isotropickernel',
    'where'
]

def _asarray(x):
    if isinstance(x, _array.StructuredArray):
        return x
    else:
        return np.array(x, copy=False)

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
    """
    
    Base class for objects representing covariance kernels. A Kernel object
    is callable, the signature is obj(x, y). Kernel objects can be summed and
    multiplied between them and with scalars, or raised to power with a scalar
    exponent.
    
    This class can be used directly by passing a callable at initialization, or
    it can be subclassed. The callable will be called with two arguments x, y
    that are two broadcastable numpy arrays. It must return Cov[f(x), f(y)]
    where `f` is the gaussian process.
    
    If `x` and `y` are structured arrays, they represent multidimensional
    input. Kernels can be specified to act only on a field of `x` and `y` or
    on all of them.
        
    The decorator `@kernel` can be used to quickly make subclasses.
    
    Methods
    -------
    diff :
        Derivatives of the kernel.
    
    """
    
    def __init__(self, kernel, *, dim=None, loc=None, scale=None, forcebroadcast=False, forcekron=False, derivable=False, **kw):
        """
        
        Initialize the object with callable `kernel`.
        
        Parameters
        ----------
        kernel : callable
            A function with signature `kernel(x, y)` where x and y are two
            broadcastable numpy arrays which computes the covariance of f(x)
            with f(y) where f is the gaussian process.
        dim : None or str
            When the input arrays are structured arrays, if dim=None the kernel
            will operate on all fields, i.e. it will be passed the whole
            arrays. If `dim` is a string, `kernel` will see only the arrays for
            the field `dim`. If `dim` is a string and the array is not
            structured, an exception is raised.
        loc, scale : scalars
            The inputs to `kernel` are transformed as (x - loc) / scale.
        forcebroadcast : bool
            If True, the inputs to `kernel` will always have the same shape.
        forcekron : bool
            If True, when calling `kernel`, if `x` and `y` are structured
            arrays, i.e. if they represent multidimensional input, `kernel` is
            invoked separately for each dimension, and the result is the
            product. Default False. If `dim` is specified, `forcekron` will
            have no effect.
        derivable : bool, int or callable
            Specifies how many times the kernel can be derived, just for
            error checking purposes. Default is False. True means infinitely
            many times derivable. If callable, it is called with the same
            keyword arguments of `kernel`.
        **kw :
            Other keyword arguments are passed to `kernel`: kernel(x, y, **kw).
        
        """
        # TODO linear transformation of input that works with arbitrarily
        # nested dtypes. Use an array/dictionary of arrays/dictionaries etc.
        # to represent a matrix. Dictionaries implicitly allow the
        # matrix to be sparse over fields. The transformation is applied prior
        # to selecting a field with `dim`.
        
        # TODO allow dim to select arbitrarily nested fields, and also sets of
        # fields and slices of array fields. It shall be similar to the Deriv
        # syntax (still not done).
        
        assert isinstance(dim, (str, type(None)))
        self._forcebroadcast = bool(forcebroadcast)
        forcekron = bool(forcekron)
        
        # Convert derivable to an integer.
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
        self._derivable = (derivable, derivable)
        
        transf = lambda x: x
        
        if isinstance(dim, str):
            def transf(x):
                if x.dtype.names is None:
                    raise ValueError('kernel called on non-structured array but dim="{}"'.format(dim))
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
        
        if dim is None and forcekron:
            def _kernel(x, y):
                x = transf(x)
                y = transf(y)
                if x.dtype.names is not None:
                    fun = lambda x, y: kernel(x, y, **kw)
                    return _prod_recurse_dtype(fun, x, y)
                else:
                    return kernel(x, y, **kw)
        else:
            _kernel = lambda x, y: kernel(transf(x), transf(y), **kw)
        
        self._kernel = _kernel
    
    def __call__(self, x, y):
        x = _asarray(x)
        y = _asarray(y)
        assert x.dtype == y.dtype
        shape = _array.broadcast(x, y).shape
        if self._forcebroadcast:
            x, y = _array.broadcast_arrays(x, y)
        result = self._kernel(x, y)
        assert isinstance(result, (np.ndarray, np.number))
        assert np.issubdtype(result.dtype, np.number)
        assert result.shape == shape
        return result
    
    def diff(self, xderiv, yderiv):
        """
        
        Return a Kernel-like object that computes the derivatives of this
        kernel. The derivatives are computed automatically with `autograd`. If
        `xderiv` and `yderiv` are trivial, this is a no-op.
        
        Parameters
        ----------
        xderiv, yderiv : Deriv
            A Deriv object or something that can be converted to a Deriv object.
        
        Returns
        -------
        diffkernel :
            A Kernel-like object representing the derivatives of this one.
            If xderiv == yderiv, it is actually another Kernel.
        """
        xderiv = _Deriv.Deriv(xderiv)
        yderiv = _Deriv.Deriv(yderiv)
        
        if not xderiv and not yderiv:
            return self
        
        orders = (xderiv.order, yderiv.order)
        if any(orders[i] > self._derivable[i] for i in range(2)):
            raise RuntimeError('derivative orders {} greater than kernel maximum {}'.format(orders, self._derivable))
        
        kernel = self._kernel
        def fun(x, y):
            # Check derivatives are ok for x and y.
            if x.dtype.names is not None:
                for deriv in xderiv, yderiv:
                    for dim in deriv:
                        if dim not in x.dtype.names:
                            raise ValueError('derivative along missing field "{}"'.format(dim))
                        if not np.issubdtype(x.dtype.fields[dim][0], np.number):
                            raise TypeError('derivative along non-numeric field "{}"'.format(dim))
            elif not xderiv.implicit or not yderiv.implicit:
                raise ValueError('explicit derivatives with non-structured array')
            
            # Handle the non-structured case.
            if x.dtype.names is None:
                f = kernel
                for _ in range(xderiv.order):
                    f = autograd.elementwise_grad(f, 0)
                for _ in range(yderiv.order):
                    f = autograd.elementwise_grad(f, 1)
                if xderiv:
                    x = _asfloat(x)
                if yderiv:
                    y = _asfloat(y)
                return f(x, y)
                
            # Autograd-friendly wrap of structured arrays.
            if xderiv:
                x = _array.StructuredArray(x)
            if yderiv:
                y = _array.StructuredArray(y)
            
            # Wrap of kernel with derivable arguments only.
            def f(*args):
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
                    f = autograd.elementwise_grad(f, i)
            for j, dim in enumerate(yderiv):
                for _ in range(yderiv[dim]):
                    f = autograd.elementwise_grad(f, 1 + i + j)
            
            # Make argument list and call function.
            args = []
            for dim in xderiv:
                args.append(_asfloat(x[dim]))
            for dim in yderiv:
                args.append(_asfloat(y[dim]))
            return f(*args)
        
        cls = Kernel if xderiv == yderiv else _KernelDeriv
        obj = cls(fun, forcebroadcast=True)
        obj._derivable = tuple(self._derivable[i] - orders[i] for i in range(2))
        return obj

class _KernelDeriv(_KernelBase):
    pass
    
class Kernel(_KernelBase):
    
    @property
    def derivable(self):
        assert self._derivable[0] == self._derivable[1]
        return self._derivable[0]
    
    def _binary(self, value, op):
        if isinstance(value, Kernel):
            obj = Kernel(op(self._kernel, value._kernel))
            obj._derivable = tuple(np.minimum(self._derivable, value._derivable))
            obj._forcebroadcast = self._forcebroadcast or value._forcebroadcast
        elif np.isscalar(value):
            assert np.isfinite(value)
            assert value >= 0
            obj = Kernel(op(self._kernel, lambda x, y: value))
            obj._derivable = self._derivable
            obj._forcebroadcast = self._forcebroadcast
        else:
            obj = NotImplemented
        return obj
    
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
    
class IsotropicKernel(Kernel):
    """
    
    Subclass of `Kernel` that represents isotropic kernels, i.e. the result
    only depends on a distance defined between points. The decorator for
    making subclasses is `isotropickernel`.
    
    """
    
    # TODO add the `distance` parameter to supply an arbitrary distance, maybe
    # allow string keywords for premade distances, like euclidean, hamming.
    
    def __init__(self, kernel, *, input='squared', scale=None, **kw):
        """
        
        Parameters
        ----------
        kernel : callable
            A function taking one argument `r2` which is the squared distance
            between x and y, plus optionally keyword arguments.
        input : str
            See "input options" below.
        scale : scalar
            The distance is divided by `scale`.
        **kw :
            Other keyword arguments are passed to the `Kernel` init.
        
        Input options
        -------------
        squared :
            Pass the squared distance (default).
        soft :
            Pass the (non-squared) distance, but zeros are replaced by very
            small numbers in a smooth way.
        
        """
        allowed_input = ('squared', 'soft')
        if not (input in allowed_input):
            raise ValueError('input option `{}` not valid, must be one of {}'.format(input, allowed_input))
        
        if scale is not None:
            assert np.isscalar(scale)
            assert np.isfinite(scale)
            assert scale > 0
        
        def function(x, y, **kwargs):
            q = _sum_recurse_dtype(lambda x, y: (x - y) ** 2, x, y)
            if scale is not None:
                q = q / scale ** 2
            if input == 'soft':
                if np.issubdtype(x.dtype, np.inexact):
                    eps = np.finfo(x.dtype).eps
                else:
                    eps = np.finfo(float).eps
                q = np.sqrt(q + eps ** 2)
            return kernel(q, **kwargs)
        
        super().__init__(function, **kw)
    
    def _binary(self, value, op):
        obj = super()._binary(value, op)
        if isinstance(obj, Kernel) and isinstance(value, __class__):
            obj.__class__ = __class__
        return obj
    
def _makekernelsubclass(kernel, superclass, **prekw):
    assert issubclass(superclass, Kernel)
    
    supername = 'Specific' + superclass.__name__
    name = getattr(kernel, '__name__', supername)
    if name == '<lambda>':
        name = supername
    
    newclass = type(name, (superclass,), {})
    
    def __init__(self, **kw):
        kwargs = prekw.copy()
        kwargs.update(kw)
        return super(newclass, self).__init__(kernel, **kwargs)
    newclass.__init__ = __init__
    newclass.__doc__ = kernel.__doc__
    
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
    
    Decorator to convert a function to a subclass of `Kernel`. Use it like this:
    
    @kernel
    def MyKernel(x, y, cippa=1, lippa=42, ...):
        return ... # something computing Cov[f(x), f(y)]
    
    """
    return _kerneldecoratorimpl(Kernel, *args, **kw)

def isotropickernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of `IsotropicKernel`. Use it
    like this:
    
    @isotropickernel
    def MyKernel(rsquared, cippa=1, lippa=42, ...):
        return ...
        # something computing Cov[f(x), f(y)] where rsquared = ||x - y||^2
    
    """
    return _kerneldecoratorimpl(IsotropicKernel, *args, **kw)

def where(condfun, kernel1, kernel2, dim=None):
    """
    
    Make a kernel(x, y) that yields:
    
      * kernel1(x, y) when condfun(x) and condfun(y) are True
    
      * kernel2(x, y) when condfun(x) and condfun(y) are False
    
      * 0 when condfun(x) is different from condfun(y)
    
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
    kernel: Kernel
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
                return x[[dim]]
            else:
                return x[dim]
        condfun0 = condfun
        condfun = lambda x: condfun0(transf(x))
    
    def kernel_op(k1, k2):
        def kernel(x, y):
            # TODO this is inefficient, kernels should be computed only on
            # the relevant points
            xcond = condfun(x)
            ycond = condfun(y)
            r = np.where(xcond & ycond, k1(x, y), k2(x, y))
            return np.where(xcond ^ ycond, 0, r)
        return kernel
    
    return kernel1._binary(kernel2, kernel_op)

# TODO add a function `choose` to extend `where`. Interface:
# choose(keyfun, mapping)
# example where `comp` is an integer field selecting the kernel:
# choose(lambda comp: comp, [kernel0, kernel1, kernel2, ...], dim='comp')
# example where `comp` is a string field, and without using `dim`:
# choose(lambda x: x['comp'], {'a': kernela, 'b': kernelb})
# define Kernel._nary and make _binary just a wrapper
