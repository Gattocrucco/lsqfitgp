# lsqfitgp/_Kernel/_crosskernel.py
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
import enum
import warnings

import numpy
from jax import numpy as jnp

from .. import _Deriv
from .. import _array
from .. import _jaxext

from . import _util

def _asfloat(x):
    return x.astype(_jaxext.float_type(x))

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
        assert issubclass(cls, __class__)
        obj = super(__class__, cls).__new__(cls)
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
        
        if dim is not None:
            def transf(x, transf=transf):
                x = transf(x)
                if x.dtype.names is None:
                    raise ValueError(f'kernel called on non-structured array but dim={dim!r}')
                elif x.dtype[dim].shape:
                    return x[[dim]]
                else:
                    return x[dim]
        
        # TODO make maxdim, loc, scale, forcekron, derivable into separate
        # methods like batch, both for tidiness and to allow the user to
        # override if needed.
        if maxdim is not None:
            def transf(x, transf=transf):
                x = transf(x)
                nd = _array._nd(x.dtype)
                with _jaxext.skipifabstract():
                    if nd > maxdim:
                        raise ValueError(f'kernel called on type with dimensionality {nd} > maxdim={maxdim}')
                return x
        
        if loc is not None:
            with _jaxext.skipifabstract():
                assert -jnp.inf < loc < jnp.inf
            transf = lambda x, transf=transf: _util.transf_recurse_dtype(lambda x: x - loc, transf(x))
        
        if scale is not None:
            with _jaxext.skipifabstract():
                assert 0 < scale < jnp.inf
            transf = lambda x, transf=transf: _util.transf_recurse_dtype(lambda x: x / scale, transf(x))
        
        # TODO when dim becomes deep, forcekron must apply also to subfields
        # for consistence. Maybe it should do it now already.
        if forcekron:
            def _kernel(x, y):
                x = transf(x)
                y = transf(y)
                fun = lambda x, y: kernel(x, y, **kw)
                return _util.prod_recurse_dtype(fun, x, y)
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
        if _util.is_numerical_scalar(value):
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
        if not _util.is_numerical_scalar(value):
            return NotImplemented
        with _jaxext.skipifabstract():
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
        
        cls = Kernel if xfun is yfun and isinstance(self, Kernel) else __class__
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
        
        cls = Kernel if xfun is yfun and isinstance(self, Kernel) else __class__
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
                f = _jaxext.elementwise_grad(f, 0)
            for _ in range(yderiv.order):
                f = _jaxext.elementwise_grad(f, 1)
            
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
                    f = _jaxext.elementwise_grad(f, 2 + i)
            for j, dim in enumerate(yderiv):
                for _ in range(yderiv[dim]):
                    f = _jaxext.elementwise_grad(f, 2 + 1 + i + j)
            
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
        
        cls = Kernel if xderiv == yderiv and isinstance(self, Kernel) else __class__
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
        kernel = _jaxext.batchufunc(self._kernel, maxnbytes=maxnbytes)
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

