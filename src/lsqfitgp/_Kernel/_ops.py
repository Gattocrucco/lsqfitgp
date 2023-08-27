# lsqfitgp/_Kernel/_ops.py
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

""" register transformations on CrossKernel """

import functools
import numbers
import warnings
import sys

from jax import numpy as jnp
import numpy

from .. import _jaxext
from .. import _Deriv
from .. import _array

from . import _util
from ._crosskernel import CrossKernel

def rescale_argparser(fun):
    if not callable(fun):
        raise ValueError("argument to 'rescale' must be a function")
    return fun

@functools.partial(CrossKernel.register_corelinop, argparser=rescale_argparser)
def rescale(core, xfun, yfun):
    r"""
    
    Rescale the output of the function.
    
    .. math::
        T(f)(x) = \mathrm{xfun}(x) f(x)
    
    Parameters
    ----------
    xfun, yfun : callable or None
        Functions from the type of the arguments of the kernel to scalar.
    
    """
    if not xfun:
        return lambda x, y: yfun(y) * core(x, y)
    elif not yfun:
        return lambda x, y: xfun(x) * core(x, y)
    else:
        return lambda x, y: xfun(x) * yfun(y) * core(x, y)

def _asfloat(x):
    return x.astype(_jaxext.float_type(x))

def diff_argparser(deriv):
    deriv = _Deriv.Deriv(deriv)
    return deriv if deriv else None

@functools.partial(CrossKernel.register_linop, argparser=diff_argparser)
def diff(self, xderiv, yderiv):
    r"""
    
    Derive the function.
    
    .. math::
        T(f)(x) = \frac{\partial^n f}{\partial x^n} (x)
    
    Parameters
    ----------
    xderiv, yderiv : Deriv_like
        A `Deriv` or something that can be converted to a `Deriv`.
    
    Raises
    ------
    RuntimeError
        The derivative orders are greater than the `derivative` attribute.
        
    """

    xderiv = _Deriv.Deriv(xderiv)
    yderiv = _Deriv.Deriv(yderiv)

    # Check kernel is derivable.
    for i, (deriv, derivability) in enumerate(zip((xderiv, yderiv), self._derivable)):
        if derivability is not None:
            # best case: only single variable order matters
            if deriv.max > derivability:
                raise ValueError(f'maximum single-variable derivative order '
                    f'{max} greater than kernel derivability {derivability} '
                    f'for argument {i}')
            # worst case: total derivation order matters
            if deriv.order > derivability:
                warnings.warn(f'total derivative order {deriv.order} greater '
                    f'than kernel derivability {derivability} for argument {i}')
    
    # Check derivatives are ok for x and y.
    def check_arg(x, deriv):
        if x.dtype.names is not None:
            for dim in deriv:
                if dim not in x.dtype.names:
                    raise ValueError(f'derivative along missing field {dim!r}')
                if not jnp.issubdtype(x.dtype[dim], jnp.number):
                    raise TypeError(f'derivative along non-numeric field {dim!r}')
        elif not deriv.implicit:
            raise ValueError('explicit derivatives with non-structured array')
        elif not jnp.issubdtype(x.dtype, jnp.number):
            raise TypeError('derivative along non-numeric array')
    def check(x, y):
        check_arg(x, xderiv)
        check_arg(y, yderiv)
    
    # Handle the non-structured case.
    if xderiv.implicit and yderiv.implicit:
        
        f = self._core
        for _ in range(xderiv.order):
            f = _jaxext.elementwise_grad(f, 0)
        for _ in range(yderiv.order):
            f = _jaxext.elementwise_grad(f, 1)
        
        def core(x, y):
            check(x, y)
            if xderiv:
                x = _asfloat(x)
            if yderiv:
                y = _asfloat(y)
            return f(x, y)
    
    # Structured case.
    else:
        
        # Wrap of kernel with derivable arguments only.
        def f(x, y, *args, core=self._core):
            i = -1
            for i, dim in enumerate(xderiv):
                x = x.at[dim].set(args[i])
            for j, dim in enumerate(yderiv):
                y = y.at[dim].set(args[1 + i + j])
            return core(x, y)
            
        # Make derivatives.
        i = -1
        for i, dim in enumerate(xderiv):
            for _ in range(xderiv[dim]):
                f = _jaxext.elementwise_grad(f, 2 + i)
        for j, dim in enumerate(yderiv):
            for _ in range(yderiv[dim]):
                f = _jaxext.elementwise_grad(f, 2 + 1 + i + j)
        
        def core(x, y):
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
    
    return self._clone(_core=core)

@CrossKernel.register_xtransf
def xtransf(fun):
    r"""
    
    Transform the inputs of the function.
    
    .. math::
        T(f)(x) = f(\mathrm{fun}(x))
    
    Parameters
    ----------
    xfun, yfun : callable or None
        Functions mapping a new kind of input to the kind of input accepted by
        the kernel.
    
    """
    if not callable(fun):
        raise ValueError("argument to 'xtransf' must be a function")
    return fun

@CrossKernel.register_xtransf
def dim(dim):
    """
    Restrict the function to a field of a structured input::

        T(f)(x) = f(x[dim])

    If the array is not structured, an exception is raised. If the field for
    name `dim` has a nontrivial shape, the array passed to the kernel is still
    structured but has only field `dim`.

    Parameters
    ----------
    xdim, ydim: None, str, list of str
        Field names or lists of field names.

    """
    if not isinstance(dim, (str, list)):
        raise TypeError(f'dim must be a (list of) string, found {dim!r}')
    def fun(x):
        if x.dtype.names is None:
            raise ValueError(f'cannot get dim={dim!r} from non-structured input')
        elif x.dtype[dim].shape:
            return x[[dim]]
        else:
            return x[dim]
    return fun

@CrossKernel.register_xtransf
def maxdim(maxdim):
    """

    Restrict the process to a maximum input dimensionality.

    Parameters
    ----------
    xmaxdim, ymaxdim: None, int
        Maximum dimensionality of the input.
    
    Notes
    -----
    Once applied a restriction, the check is hardcoded into the kernel core and
    it is not possible to remove it by applying again `maxdim` with a larger
    limit.

    """
    if not isinstance(maxdim, numbers.Integral) or maxdim < 0:
        raise ValueError(f'maximum dimensionality {maxdim!r} not valid')

    def fun(x):
        nd = _array._nd(x.dtype)
        with _jaxext.skipifabstract():
            if nd > maxdim:
                raise ValueError(f'kernel applied to input with {nd} '
                    f'fields > maxdim={maxdim}')
        return x
    
    return fun

@CrossKernel.register_xtransf
def loc(loc):
    r"""
    Translate the process inputs:

    .. math::
        T(f)(x) = f(x - \mathrm{loc})

    Parameters
    ----------
    xloc, yloc: None, number
        Translations.

    """
    with _jaxext.skipifabstract():
        assert -jnp.inf < loc < jnp.inf, loc
    return lambda x: _util.ufunc_recurse_dtype(lambda x: x - loc, x)

@CrossKernel.register_xtransf
def scale(scale):
    r"""
    Rescale the process inputs:

    .. math::
        T(f)(x) = f(x / \mathrm{scale})

    Parameters
    ----------
    xscale, yscale: None, number
        Rescaling factors.

    """
    with _jaxext.skipifabstract():
        assert 0 < scale < jnp.inf, scale
    return lambda x: _util.ufunc_recurse_dtype(lambda x: x / scale, x)

def derivable_argparser(derivable):
    if isinstance(derivable, bool):
        return sys.maxsize if derivable else 0
    elif isinstance(derivable, numbers.Integral) and derivable >= 0:
        return int(derivable)
    else:
        raise ValueError(f'derivability degree {derivable!r} not valid')

@functools.partial(CrossKernel.register_linop, argparser=derivable_argparser)
def derivable(self, xderivable, yderivable):
    """
    Specify the degree of derivability of the function.

    Parameters
    ----------
    xderivable, yderivable: int or None
        Degree of derivability of the function. None means unknown.

    """
    self = self._clone()
    self._derivable = xderivable, yderivable
    return self

    # TODO hardcode the derivability check into the core by inspecting jax
    # tracers. This 1) makes diff and derivable coretransfs, 2) removes
    # fuzziness from the derivability check. To support nd, use pytrees, since
    # diff makes sure it is a StructuredArray.

def normalize_argparser(do):
    return do if do else None

@functools.partial(CrossKernel.register_corelinop, argparser=normalize_argparser)
def normalize(core, dox, doy):
    r"""
    Rescale the process to unit variance.

    .. math::
        T(f)(x) &= f(x) / \sqrt{\mathrm{Std}[f(x)]} \\
                &= f(x) / \sqrt{\mathrm{kernel}(x, x)}

    Parameters
    ----------
    dox, doy : bool
        Whether to rescale.
    """
    if dox and doy:
        return lambda x, y: core(x, y) / jnp.sqrt(core(x, x) * core(y, y))
    elif dox:
        return lambda x, y: core(x, y) / jnp.sqrt(core(x, x))
    else:
        return lambda x, y: core(x, y) / jnp.sqrt(core(y, y))

@CrossKernel.register_algop
def add(self, other):
    r"""
    
    Sum of kernels.
    
    .. math::
        \mathrm{newkernel}(x, y) &= \mathrm{kernel}(x, y) + \mathrm{other}(x, y), \\
        \mathrm{newkernel}(x, y) &= \mathrm{kernel}(x, y) + \mathrm{other}.
    
    Parameters
    ----------
    other : CrossKernel or scalar
        The other kernel.
    
    """
    if _util.is_numerical_scalar(other):
        core = lambda x, y, core=self._core: core(x, y) + other
    elif isinstance(other, CrossKernel):
        core = lambda x, y, core=self._core, other=other._core: core(x, y) + other(x, y)
    else:
        return NotImplemented
    return self._clone(_core=core)

@CrossKernel.register_algop
def mul(self, other):
    r"""
    
    Product of kernels.
    
    .. math::
        \mathrm{newkernel}(x, y) &= \mathrm{kernel}(x, y) \cdot \mathrm{other}(x, y), \\
        \mathrm{newkernel}(x, y) &= \mathrm{kernel}(x, y) \cdot \mathrm{other}.
    
    Parameters
    ----------
    other : CrossKernel or scalar
        The other kernel.
    
    """
    if _util.is_numerical_scalar(other):
        core = lambda x, y, core=self._core: core(x, y) * other
    elif isinstance(other, CrossKernel):
        core = lambda x, y, core=self._core, other=other._core: core(x, y) * other(x, y)
    else:
        return NotImplemented
    return self._clone(_core=core)

@CrossKernel.register_algop
def pow(self, *, exponent):
    r"""
    
    Power of the kernel.
    
    .. math::
        \mathrm{newkernel}(x, y) = \mathrm{kernel}(x, y)^{\mathrm{exponent}}
    
    Parameters
    ----------
    exponent : nonnegative integer
        The exponent. If traced by jax, it must have unsigned integer type.
    
    """
    if _util.is_nonnegative_integer_scalar(exponent):
        core = lambda x, y, core=self._core: core(x, y) ** exponent
    else:
        return NotImplemented
    return self._clone(_core=core)
