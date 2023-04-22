# lsqfitgp/_Kernel.py
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

import sys
import warnings
import types
import enum

import jax
from jax import numpy as jnp
import numpy

from . import _array
from . import _Deriv
from . import _patch_jax

__all__ = [
    'CrossKernel',
    'Kernel',
    'StationaryKernel',
    'IsotropicKernel',
    'kernel',
    'stationarykernel',
    'isotropickernel',
    'where',
]

# TODO make some of these functions static methods of CrossKernel if they are
# not used elsewhere.

def _asfloat(x):
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(float)
    else:
        return x

def _isscalar(x):
    return not isinstance(x, CrossKernel) and jnp.ndim(x) == 0

# TODO reimplement with tree_reduce, closuring ndim to recognize shaped fields
def _reduce_recurse_dtype(fun, *args, reductor=None, npreductor=None, jnpreductor=None):
    x = args[0]
    if x.dtype.names is None:
        return fun(*args)
    else:
        acc = None
        for name in x.dtype.names:
            recargs = (arg[name] for arg in args)
            reckw = dict(reductor=reductor, npreductor=npreductor, jnpreductor=jnpreductor)
            result = _reduce_recurse_dtype(fun, *recargs, **reckw)
            
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
    plus = lambda a, b: a + b
    return _reduce_recurse_dtype(fun, *args, reductor=plus, npreductor=numpy.sum, jnpreductor=jnp.sum)

def prod_recurse_dtype(fun, *args):
    times = lambda a, b: a * b
    return _reduce_recurse_dtype(fun, *args, reductor=times, npreductor=numpy.prod, jnpreductor=jnp.prod)

# TODO reimplement with tree_map
def transf_recurse_dtype(transf, x, *args):
    if x.dtype.names is None:
        return transf(x, *args)
    else:
        x = _array.StructuredArray(x)
        # redundant creation of StructuredArrays with nested dtypes; whatever
        for name in x.dtype.names:
            newargs = tuple(y[name] for y in args)
            x = x.at[name].set(transf_recurse_dtype(transf, x[name], *newargs))
        return x

def _greatest_common_superclass(classes):
    # from https://stackoverflow.com/a/25787091/3942284
    classes = [x.mro() for x in classes]
    for x in classes[0]: # pragma: no branch
        if all(x in mro for mro in classes):
            return x

class CrossKernel:
    
    # TODO if I implement double callable derivable, derivable should always be
    # a callable, and _newkernel_from should propagate it properly (with an
    # `and`?).
    
    # TODO make some unit tests checking that Kernel classes are
    # propagated properly

    def _copywith(self, core):
        """
        Return a kernel object with the same attributes but a new kernel
        function.
        """
        obj = super().__new__(type(self))
        for n, v in vars(self).items():
            setattr(obj, n, v)
        obj._kernel = core
        return obj

    @staticmethod
    def _newkernel_from(core, kernels):
        """
        Make a new kernel object which is the result of an operation on the
        kernels in ``kernels``, with implementation callable ``core``.
        """
        assert kernels
        classes = (IsotropicKernel, *map(type, kernels))
        cls = _greatest_common_superclass(classes)
        assert issubclass(cls, CrossKernel)
        obj = super(CrossKernel, cls).__new__(cls)
        mind = [k._minderivable for k in kernels]
        maxd = [k._maxderivable for k in kernels]
        obj._minderivable = tuple(numpy.min(mind, axis=0))
        obj._maxderivable = tuple(numpy.max(maxd, axis=0)) # TODO or is it the sum, for the nd case?
        obj.initargs = None
        obj._maxdim = None
        # TODO implement maxdim like derivable (distinguish left/right
        # argument and have minimum/maximum). Here take as minimum the
        # minimum maxdim of the kernels and as maximum the sum of the
        # maxima.
        obj._kernel = core
        return obj
    
    def __new__(cls, kernel, *, dim=None, loc=None, scale=None, forcekron=False, derivable=None, saveargs=False, maxdim=sys.maxsize, batchbytes=None, **kw):
        """
        
        Base class for objects representing covariance kernels.
        
        A Kernel object is callable, the signature is obj(x, y). Kernel objects
        can be summed and multiplied between them and with scalars, or raised
        to power with a scalar exponent.
    
        Attributes
        ----------
        derivable : int or None
            How many times the process represented by the kernel is derivable.
            ``sys.maxsize`` if it is smooth. ``None`` if the derivability is
            unknown.
        maxdim : int or None
            Maximum input dimensionality. None means unknown.
        
        Parameters
        ----------
        kernel : callable
            A function with signature ``kernel(x, y)``, where ``x`` and ``y`` are
            two broadcastable numpy arrays, which computes the covariance of
            f(x) with f(y) where f is the Gaussian process.
        dim : None or str
            When the input arrays are structured arrays, if ``dim`` is None the
            kernel will operate on all fields, i.e., it will be passed the whole
            arrays. If ``dim`` is a string, ``kernel`` will see only the arrays for
            the field named ``dim``. If ``dim`` is a string and the array is not
            structured, an exception is raised. If the field for name `dim` has
            a nontrivial shape, the array passed to ``kernel`` is still
            structured but has only field ``dim``.
        loc, scale : scalar
            The inputs to ``kernel`` are transformed as (x - loc) / scale.
        forcekron : bool
            If True, when calling ``kernel``, if ``x`` and ``y`` are structured
            arrays, i.e., if they represent multidimensional input, ``kernel`` is
            invoked separately for each dimension, and the result is the
            product. Default False. The selection indicated by ``dim`` limits the
            dimensions over which ``forcekron`` applies.
        derivable : bool, int, None, or callable
            Specifies how many times the kernel can be derived, only for error
            checking purposes. True means infinitely many times derivable. If
            callable, it is called with the same keyword arguments as ``kernel``.
            If None (default) it means that the degree of derivability is
            unknown.
        saveargs : bool
            If True, save the all the initialization arguments in a
            dictionary under the attribute ``initargs``. Default False.
        maxdim : int, callable or None
            The maximum input dimensionality accepted by the kernel. If
            callable, it is called with the same keyword arguments as the
            kernel. Default sys.maxsize.
        batchbytes : number, optional
            If specified, apply ``batch(batchbytes)`` to the kernel.
        **kw
            Additional keyword arguments are passed to ``kernel``.
        
        Methods
        -------
        rescale : multiply the kernel by some functions
        diff : take derivatives of the kernel
        xtransf : transform the inputs to the kernel
        fourier : take the Fourier series
        taylor : take the Taylor series
        batch : batch the computation of the kernel
        
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
        
        # TODO maxdim's default should be None, not sys.maxsize

        self = super().__new__(cls)
                
        if saveargs:
            # TODO I want to get rid of this functionality
            self.initargs = dict(
                dim=dim,
                loc=loc,
                scale=scale,
                forcekron=forcekron,
                derivable=derivable,
                **kw,
            )
        else:
            self.initargs = None
        
        # Check simple arguments.
        assert isinstance(dim, (str, type(None)))
        forcekron = bool(forcekron)
        
        # Convert `derivable` to a tuple of integers.
        if callable(derivable):
            derivable = derivable(**kw)
        if derivable is None:
            derivable = (0, sys.maxsize)
        elif isinstance(derivable, bool):
            derivable = sys.maxsize if derivable else 0
        elif int(derivable) == derivable:
            assert derivable >= 0
            derivable = int(derivable)
        else:
            raise ValueError(f'derivability degree {derivable!r} not valid')
        if not isinstance(derivable, tuple):
            derivable = (derivable, derivable)
        self._minderivable = (derivable[0], derivable[0])
        self._maxderivable = (derivable[1], derivable[1])
        
        # Convert `maxdim` to a tuple of integers.
        if callable(maxdim):
            maxdim = maxdim(**kw)
        assert maxdim is None or int(maxdim) == maxdim and maxdim >= 0
        self._maxdim = maxdim
        
        transf = lambda x: x
        
        # TODO make maxdim, loc, scale, forcekron, derivable into separate
        # methods like batch, both for tidiness and to allow the user to
        # ovverride if needed.
        if maxdim is not None:
            def transf(x):
                nd = _array._nd(x.dtype)
                with _patch_jax.skipifabstract():
                    if nd > maxdim:
                        raise ValueError(f'kernel called on type with dimensionality {nd} > maxdim={maxdim}')
                return x
        
        if dim is not None:
            def transf(x, transf=transf):
                x = transf(x)
                if x.dtype.names is None:
                    raise ValueError(f'kernel called on non-structured array but dim={dim!r}')
                elif x.dtype.fields[dim][0].shape:
                    return x[[dim]]
                else:
                    return x[dim]
        
        if loc is not None:
            with _patch_jax.skipifabstract():
                assert -jnp.inf < loc < jnp.inf
            transf = lambda x, transf=transf: transf_recurse_dtype(lambda x: x - loc, transf(x))
        
        if scale is not None:
            with _patch_jax.skipifabstract():
                assert 0 < scale < jnp.inf
            transf = lambda x, transf=transf: transf_recurse_dtype(lambda x: x / scale, transf(x))
        
        # TODO when dim becomes deep, forcekron must apply also to subfields
        # for consistence. Maybe it should do it now already.
        if forcekron:
            def _kernel(x, y):
                x = transf(x)
                y = transf(y)
                fun = lambda x, y: kernel(x, y, **kw)
                return prod_recurse_dtype(fun, x, y)
        else:
            _kernel = lambda x, y: kernel(transf(x), transf(y), **kw)
        
        self._kernel = _kernel

        if batchbytes is not None:
            self = self.batch(batchbytes)

        return self
    
    # TODO it would be useful to be able to pass additional keyword arguments
    # here. Since _kernel may have been transformed, I'd need to add keyword
    # arguments support to all kernel transformations.
    def __call__(self, x, y):
        x = _array.asarray(x)
        y = _array.asarray(y)
        numpy.result_type(x.dtype, y.dtype)
        shape = _array.broadcast(x, y).shape
        # TODO allow the result to have a different shape and broadcast it,
        # this handles automatically constant matrices without using memory,
        # and makes writing the kernel simpler
        result = self._kernel(x, y)
        assert isinstance(result, (numpy.ndarray, jnp.number, jnp.ndarray))
        assert jnp.issubdtype(result.dtype, jnp.number), result.dtype
        assert result.shape == shape, (result.shape, shape)
        return result

    class _side(enum.Enum):
        LEFT = 0
        RIGHT = 1
        BOTH = 2
         
    @classmethod
    def _nary(cls, op, kernels, side):
        
        if side is cls._side.LEFT:
            wrapper = lambda c, _, y: lambda x: c(x, y)
            arg = lambda x, _: x
        elif side is cls._side.RIGHT:
            wrapper = lambda c, x, _: lambda y: c(x, y)
            arg = lambda _, y: y
        elif side is cls._side.BOTH:
            wrapper = lambda c, _, __: lambda xy: c(*xy)
            arg = lambda x, y: (x, y)
        else: # pragma: no cover
            raise KeyError(side)
        
        cores = [k._kernel for k in kernels]
        def core(x, y):
            wrapped = [wrapper(c, x, y) for c in cores]
            transformed = op(*wrapped)
            return transformed(arg(x, y))
        
        return cls._newkernel_from(core, kernels)
    
    def _binary(self, value, op):
        if _isscalar(value):
            val = value
            value = self._copywith(lambda x, y: val)
            # TODO this may add too much uncertainty in the derivability
            # check
        if not isinstance(value, CrossKernel):
            return NotImplemented
        return self._nary(op, [self, value], self._side.BOTH)
    
    def __add__(self, value):
        return self._binary(value, lambda f, g: lambda x: f(x) + g(x))
    
    __radd__ = __add__
    
    def __mul__(self, value):
        return self._binary(value, lambda f, g: lambda x: f(x) * g(x))
    
    __rmul__ = __mul__
    
    def __pow__(self, value):
        if not _isscalar(value):
            return NotImplemented
        with _patch_jax.skipifabstract():
            assert 0 <= value < jnp.inf, value
        return self._binary(value, lambda f, g: lambda x: f(x) ** g(x))
        # Only infinitely divisible kernels allow non-integer exponent.
        # It would be difficult to check for this though, since the user can
        # compose and write arbitrary kernels.

    def _swap(self):
        """permute the arguments (cross kernels are not symmetric)"""
        kernel = self._kernel
        obj = self._copywith(lambda x, y: kernel(y, x))
        obj._minderivable = obj._minderivable[::-1]
        obj._maxderivable = obj._maxderivable[::-1]
        return obj
        # TODO make _swap public?
                
    def rescale(self, xfun, yfun):
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
            The rescaled kernel. If ``xfun is yfun``, it is a Kernel object,
            otherwise a Kernel-like one.
        
        """
        
        # TODO option to specify derivability of xfun and yfun, to avoid
        # zeroing _minderivable
        
        if xfun is None and yfun is None:
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
        
        cls = Kernel if xfun is yfun and isinstance(self, Kernel) else CrossKernel
        obj = cls(fun)
        obj._maxderivable = self._maxderivable
        return obj
    
    def xtransf(self, xfun, yfun):
        """
        
        Transform the inputs of the kernel.
        
        .. math::
            h(x, y) = k(f(x), g(y))
        
        Parameters
        ----------
        xfun, yfun : callable or None
            Functions mapping a new kind of input to the kind of input
            accepted by the kernel. If both are None, this is a no-op.
        
        Returns
        -------
        h : Kernel-like
            The transformed kernel. If ``xfun is yfun``, it is a Kernel object,
            otherwise a Kernel-like one.
        
        """
        
        # TODO option to specify derivability of xfun and yfun, to avoid
        # zeroing _minderivable
        
        if xfun is None and yfun is None:
            return self
        
        kernel = self._kernel
        
        if xfun is None:
            def fun(x, y):
                return kernel(x, yfun(y))
        elif yfun is None:
            def fun(x, y):
                return kernel(xfun(x), y)
        else:
            def fun(x, y):
                return kernel(xfun(x), yfun(y))
        
        cls = Kernel if xfun is yfun and isinstance(self, Kernel) else CrossKernel
        obj = cls(fun)
        obj._maxderivable = self._maxderivable
        return obj

    def diff(self, xderiv, yderiv):
        """
        
        Return a Kernel-like object that computes the derivatives of this
        kernel. The derivatives are computed automatically with JAX. If
        ``xderiv`` and ``yderiv`` are trivial, this is a no-op.
        
        .. math::
            h(x, y) = \\frac{\\partial^n}{\\partial x^n}
                      \\frac{\\partial^m}{\\partial y^m}
                      k(x, y)
        
        Parameters
        ----------
        xderiv, yderiv : Deriv-like
            A `Deriv` object or something that can be converted to a
            Deriv object.
        
        Returns
        -------
        h : Kernel-like
            An object representing the derivatives of this one. If ``xderiv ==
            yderiv``, it is actually another Kernel.
        
        Raises
        ------
        RuntimeError
            The derivative orders are greater than the ``derivative`` attribute.
            
        """
                
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
                        if not jnp.issubdtype(x.dtype.fields[dim][0], jnp.number):
                            raise TypeError(f'derivative along non-numeric field {dim!r}')
            elif not xderiv.implicit or not yderiv.implicit:
                raise ValueError('explicit derivatives with non-structured array')
        
        
        # Handle the non-structured case.
        if xderiv.implicit and yderiv.implicit:
            
            f = self._kernel
            for _ in range(xderiv.order):
                f = _patch_jax.elementwise_grad(f, 0)
            for _ in range(yderiv.order):
                f = _patch_jax.elementwise_grad(f, 1)
            
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
                    x = x.at[dim].set(args[i])
                for j, dim in enumerate(yderiv):
                    y = y.at[dim].set(args[1 + i + j])
                return kernel(x, y)
                
            # Make derivatives.
            i = -1
            for i, dim in enumerate(xderiv):
                for _ in range(xderiv[dim]):
                    f = _patch_jax.elementwise_grad(f, 2 + i)
            for j, dim in enumerate(yderiv):
                for _ in range(yderiv[dim]):
                    f = _patch_jax.elementwise_grad(f, 2 + 1 + i + j)
            
            def fun(x, y):
                check(x, y)
                                
                # JAX-friendly wrap of structured arrays.
                x = _array.StructuredArray(x)
                y = _array.StructuredArray(y)
            
                # Make argument list and call function.
                args = []
                for dim in xderiv:
                    args.append(_asfloat(x[dim]))
                for dim in yderiv:
                    args.append(_asfloat(y[dim]))
                return f(x, y, *args)
        
        cls = Kernel if xderiv == yderiv and isinstance(self, Kernel) else CrossKernel
        obj = cls(fun)
        obj._minderivable = tuple(self._minderivable[i] - orders[i] for i in range(2))
        obj._maxderivable = tuple(self._maxderivable[i] -   maxs[i] for i in range(2))
        return obj

    def batch(self, maxnbytes):
        """
        Return a batched version of the kernel.

        The batched kernel processes its inputs in chunks to try to limit memory
        usage.

        Parameters
        ----------
        maxnbytes : number
            The maximum number of input bytes per chunk, counted after
            broadcasting the inputs (actual broadcasting may not occur if not
            induced by the operations in the kernel).

        Returns
        -------
        batched_kernel : CrossKernel
            The same kernel but with batched computations.
        """
        kernel = _patch_jax.batchufunc(self._kernel, maxnbytes=maxnbytes)
        return self._copywith(kernel)
    
    def fourier(self, dox, doy):
        """
        
        Compute the Fourier series of the kernel.
        
        .. math::
            h(k, y) = \\begin{cases}
                \\frac2T \\int_0^T \\mathrm dx\\, k(x, y)
                \\cos\\left(\\frac{2\\pi}T \\frac k2 x\\right)
                & \\text{if $k$ is even} \\\\
                \\frac2T \\int_0^T \\mathrm dx\\, k(x, y)
                \\sin\\left(\\frac{2\\pi}T \\frac{k+1}2 x\\right)
                & \\text{if $k$ is odd}
            \\end{cases}
        
        The period :math:`T` is implicit in the definition of the kernel.
        
        Parameters
        ----------
        dox, doy : bool
            Specify if to compute the series w.r.t. x, y or both. If both are
            False, this is a no-op.
        
        Returns
        -------
        h : Kernel-like
            A Kernel-like object computing the Fourier series. If dox and
            doy are equal, it is a Kernel.
        
        """
        if not dox and not doy:
            return self
        raise NotImplementedError
    
    def taylor(self, dox, doy):
        """
        
        Compute the Taylor series of the kernel.
        
        .. math::
            h(k, y) = \\left.
                \\frac{\\partial^k}{\\partial x^k} k(x, y)
            \\right|_{x_0}
        
        The expansion point :math:`x0` is implicit in the definition of the kernel.
        
        Parameters
        ----------
        dox, doy : bool
            Specify if to compute the series w.r.t. x, y or both. If both are
            False, this is a no-op.
        
        Returns
        -------
        h : Kernel-like
            A Kernel-like object computing the Taylor series. If dox and
            doy are equal, it is a Kernel.
        
        """ 
        if not dox and not doy:
            return self
        raise NotImplementedError

class Kernel(CrossKernel):
    
    @property
    def derivable(self):
        assert self._minderivable[0] == self._minderivable[1]
        assert self._maxderivable[0] == self._maxderivable[1]
        if self._minderivable == self._maxderivable:
            return self._minderivable[0]
        else:
            return None
    
    @property
    def maxdim(self):
        return self._maxdim
        
    def _binary(self, value, op):
        with _patch_jax.skipifabstract():
            assert not _isscalar(value) or 0 <= value < jnp.inf, value
        return super()._binary(value, op)
    
class StationaryKernel(Kernel):

    def __new__(cls, kernel, *, input='signed', scale=None, **kw):
        """
        
        Subclass of `Kernel` for isotropic kernels.
    
        Parameters
        ----------
        kernel : callable
            A function taking one argument ``delta`` which is the difference
            between x and y, plus optionally keyword arguments.
        input : {'signed', 'soft', 'hard'}
            If 'signed' (default), `kernel` is passed the bare difference. If
            'soft', `kernel` is passed the absolute value of the difference,
            and the difference of equal points is a small number instead of
            zero. If 'hard', the absolute value.
        scale : scalar
            The difference is divided by ``scale``.
        **kw
            Additional keyword arguments are passed to the `Kernel` init.
                
        """
        
        # TODO using 'signed', 'abs', 'softabs' as labels could be clearer.

        if input == 'soft':
            func = lambda x, y: _softabs(x - y)
        elif input == 'signed':
            func = lambda x, y: x - y
        elif input == 'hard':
            func = lambda x, y: jnp.abs(x - y)
        else:
            raise KeyError(input)
        
        transf = lambda q: q
        if scale is not None:
            with _patch_jax.skipifabstract():
                assert 0 < scale < jnp.inf
            transf = lambda q : q / scale
        
        def function(x, y, **kwargs):
            q = transf_recurse_dtype(func, x, y)
            return kernel(transf(q), **kwargs)
        
        return Kernel.__new__(cls, function, **kw)

class IsotropicKernel(StationaryKernel):
    
    # TODO add the `distance` parameter to supply an arbitrary distance, maybe
    # allow string keywords for premade distances, like euclidean, hamming,
    # p-norms. Question: the known isotropic kernels effectively work for any
    # distance, or the proofs are only for the 2-norm? (I guess only 2-norm)
    
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
    
    def __new__(cls, kernel, *, input='squared', scale=None, **kw):
        """
        
        Subclass of :class:`Kernel` for isotropic kernels.
    
        Parameters
        ----------
        kernel : callable
            A function taking one argument ``r2`` which is the squared distance
            between x and y, plus optionally keyword arguments.
        input : {'squared', 'hard', 'soft', 'raw'}
            If 'squared' (default), ``kernel`` is passed the squared distance.
            If 'hard', it is passed the distance (not squared). If 'soft', it
            is passed the distance, and the distance of equal points is a small
            number instead of zero. If 'raw', the kernel is passed both points
            separately like non-stationary kernels.
        scale : scalar
            The distance is divided by ``scale``.
        **kw
            Additional keyword arguments are passed to the `Kernel` init.
        
        Notes
        -----
        The 'soft' option will cause problems with second derivatives in more
        than one dimension.
                
        """
        if input == 'raw':
            return Kernel.__new__(cls, kernel, scale=scale, **kw)
            
        if input in ('squared', 'hard'):
            func = lambda x, y: (x - y) ** 2
        elif input == 'soft':
            func = lambda x, y: _softabs(x - y) ** 2
        else:
            raise KeyError(input)
        
        transf = lambda q: q
        
        if scale is not None:
            with _patch_jax.skipifabstract():
                assert 0 < scale < jnp.inf
            transf = lambda q: q / scale ** 2
        
        if input in ('soft', 'hard'):
            transf = (lambda t: lambda q: jnp.sqrt(t(q)))(transf)
            # I do square and then square root because I first have to
            # compute the sum of squares
        
        def function(x, y, **kwargs):
            q = sum_recurse_dtype(func, x, y)
            return kernel(transf(q), **kwargs)
        
        return Kernel.__new__(cls, function, **kw)

def _eps(x):
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.finfo(x.dtype).eps
        # finfo(x) does not work in numpy 1.20
    else:
        return jnp.finfo(jnp.empty(())).eps

def _softabs(x):
    return jnp.abs(x) + _eps(x)

class Zero(IsotropicKernel):

    def __new__(cls):
        self = object.__new__(cls)
        self._kernel = lambda x, y: jnp.broadcast_to(0., jnp.broadcast_shapes(x.shape, y.shape))
        self._minderivable = (sys.maxsize, sys.maxsize)
        self._maxderivable = (sys.maxsize, sys.maxsize)
        self.initargs = None
        self._maxdim = sys.maxsize
        return self
    
    _swap = lambda self: self
    diff = lambda self, xderiv, yderiv: self
    batch = lambda self, maxnbytes: self

def _makekernelsubclass(kernel, superclass, **prekw):
    assert issubclass(superclass, Kernel)
    
    if hasattr(kernel, 'pyfunc'): # np.vectorize objects
        named_object = kernel.pyfunc
    else:
        named_object = kernel
    
    name = getattr(named_object, '__name__', 'DecoratedKernel')
    newclass = types.new_class(name, (superclass,))
        
    prekwset = set(prekw)
    def __new__(cls, **kw):
        kwargs = prekw.copy()
        shared_keys = prekwset & set(kw)
        if shared_keys:
            msg = 'overriding init argument(s) ' + ', '.join(shared_keys)
            msg += ' of kernel ' + name
            warnings.warn(msg)
        kwargs.update(kw)
        return super(newclass, cls).__new__(cls, kernel, **kwargs)
    
    newclass.__new__ = __new__
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
    
    Decorator to convert a function to a subclass of `Kernel`. Example::
    
        @kernel(loc=10) # the default loc will be 10
        def MyKernel(x, y, cippa=1, lippa=42):
            return cippa * (x * y) ** lippa
    
    """
    return _kerneldecoratorimpl(Kernel, *args, **kw)

def stationarykernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of `StationaryKernel`.
    Example::
    
        @stationarykernel(input='soft')
        def MyKernel(absdelta, cippa=1, lippa=42):
            return cippa * sum(
                np.exp(-absdelta[name] / lippa)
                for name in absdelta.dtype.names
            )
    
    """
    return _kerneldecoratorimpl(StationaryKernel, *args, **kw)

def isotropickernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of `IsotropicKernel`.
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
        If specified, when the input arrays are structured, ``condfun`` is
        applied only to the field ``dim``. If the field has a shape, the
        array passed to ``condfun`` still has ``dim`` as explicit field.
    
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
        condfun = lambda x, condfun=condfun: condfun(transf(x))
    
    def kernel_op(k1, k2):
        def kernel(xy):
            # TODO this is inefficient, kernels should be computed only on
            # the relevant points. To support this with autograd, make a
            # custom np.where that uses assignment and define its vjp.
            
            # TODO this may produce a very sparse matrix, when I implement
            # sparse support do it here too.
            
            # TODO it will probably often be the case that the result is
            # either dense or all zero. The latter case can be optimized this
            # way: if it's zero, broadcast a 0-d array to the required shape,
            # and flag it as all zero with an instance variable.
            
            x, y = xy
            xcond = condfun(x)
            ycond = condfun(y)
            r = jnp.where(xcond & ycond, k1(xy), k2(xy))
            return jnp.where(xcond ^ ycond, 0, r)
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
