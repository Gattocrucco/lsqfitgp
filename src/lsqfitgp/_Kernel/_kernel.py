# lsqfitgp/_Kernel/_kernel.py
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
from . import _crosskernel

class Kernel(_crosskernel.CrossKernel):
    r"""

    Subclass of `CrossKernel` to represent the kernel of a single function:

    .. math::
        \mathrm{kernel}(x, y) = \mathrm{Cov}[f(x), f(y)].

    Attributes
    ----------
    derivable : int or None
        The degree of differentiability of the function.
    maxdim : int or None
        Maximum dimensionality of the function input.

    See also
    --------
    StationaryKernel, IsotropicKernel

    """

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        del cls._transf

    @property
    def derivable(self):
        assert self._derivable[0] == self._derivable[1]
        return self._derivable[0]
    
    @property
    def maxdim(self):
        assert self._maxdim[0] == self._maxdim[1]
        return self._maxdim[0]
        
    def _binary(self, value, op):
        if _util.is_numerical_scalar(value):
            with _jaxext.skipifabstract():
                assert 0 <= value < jnp.inf, value
        return super()._binary(value, op)

    def forcekron(self):
        r"""
        
        Force the kernel to be a separate product over dimensions:

        .. math::
            \mathrm{newkernel}(x, y) = \prod_i \mathrm{kernel}(x_i, y_i)
        
        Returns
        -------
        newkernel : Kernel
            The transformed kernel.

        """

        core = lambda x, y, core=self._core: _util.prod_recurse_dtype(core, x, y)
        return self._clone(core=core, cls=__class__)

    # TODO make some unit tests checking that Kernel classes are
    # propagated properly

_crosskernel.Kernel = Kernel

def _asfloat(x):
    return x.astype(_jaxext.float_type(x))

@Kernel.register_coretransf
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

def diff_argparser(deriv):
    deriv = _Deriv.Deriv(deriv)
    return deriv if deriv else None

@functools.partial(Kernel.register_transf, argparser=diff_argparser)
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
                raise RuntimeError(f'maximum single-variable derivative order '
                    f'{max} greater than kernel derivability {derivability} '
                    f'for argument {i}')
            # worst case: total derivation order matters
            if deriv.order > derivability:
                warnings.warn(f'total derivative order {order} greater than '
                    f'kernel derivability {derivability} for argument {i}')
    
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
    
    return self._clone(core=core)

@Kernel.register_xtransf
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
    return fun

@Kernel.register_xtransf
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
    def fun(x):
        if x.dtype.names is None:
            raise ValueError(f'cannot get dim={dim!r} from non-structured input')
        elif x.dtype[dim].shape:
            return x[[dim]]
        else:
            return x[dim]
    return fun

def maxdim_argparser(maxdim):
    if maxdim is None:
        return None
    elif isinstance(maxdim, numbers.Integral):
        return int(maxdim)
    elif maxdim == numpy.inf:
        return sys.maxsize
    else:
        raise ValueError(f'maximum dimensionality {maxdim!r} not valid')

@functools.partial(Kernel.register_transf, argparser=maxdim_argparser)
def maxdim(self, xmaxdim, ymaxdim):
    """

    Restrict the process to a maximum input dimensionality.

    Parameters
    ----------
    xmaxdim, ymaxdim: None, int
        Maximum dimensionality of the input.

    """
    def xtransf(maxdim):
        def fun(x):
            nd = _array._nd(x.dtype)
            with _jaxext.skipifabstract():
                if nd > maxdim:
                    raise ValueError(f'kernel applied to input with {nd} '
                        f'fields > maxdim={maxdim}')
            return x
        return None if maxdim is None else fun
    self = self.transf('xtransf', xtransf(xmaxdim), xtransf(ymaxdim))
    self._maxdim = xmaxdim, ymaxdim
    return self

@Kernel.register_xtransf
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
    return lambda x: _util.transf_recurse_dtype(lambda x: x - loc, x)

@Kernel.register_xtransf
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
    return lambda x: _util.transf_recurse_dtype(lambda x: x / scale, x)

def derivable_argparser(derivable):
    if derivable is None:
        return None
    elif isinstance(derivable, bool):
        return sys.maxsize if derivable else 0
    elif (asint := int(derivable)) == derivable:
        assert asint >= 0
        return asint
    else:
        raise ValueError(f'derivability degree {derivable!r} not valid')

@functools.partial(Kernel.register_transf, argparser=derivable_argparser)
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

# TODO transf 'normalize' to make a correlation function.
