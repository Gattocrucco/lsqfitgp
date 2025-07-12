# lsqfitgp/_Kernel/_ops.py
#
# Copyright (c) 2020, 2022, 2023, 2025, Giacomo Petrillo
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

""" register linops on CrossKernel and AffineSpan """

import functools
import numbers
import sys

from jax import numpy as jnp
import numpy

from .. import _jaxext
from .. import _Deriv
from .. import _array

from . import _util
from ._crosskernel import CrossKernel, AffineSpan

def rescale_argparser(fun):
    if not callable(fun):
        raise ValueError("argument to 'rescale' must be a function")
    return fun

@functools.partial(CrossKernel.register_corelinop, argparser=rescale_argparser)
def rescale(core, xfun, yfun):
    r"""
    
    Rescale the output of the function.
    
    .. math::
        T(f)(x) = \mathrm{fun}(x) f(x)
    
    Parameters
    ----------
    xfun, yfun : callable or None
        Functions from the type of the arguments of the kernel to scalar.
    
    """
    if not xfun:
        return lambda x, y, **kw: core(x, y, **kw) * yfun(y)
    elif not yfun:
        return lambda x, y, **kw: xfun(x) * core(x, y, **kw)
    else:
        return lambda x, y, **kw: xfun(x) * core(x, y, **kw) * yfun(y)

@CrossKernel.register_xtransf
def derivable(derivable):
    """
    Specify the degree of derivability of the function.

    Parameters
    ----------
    xderivable, yderivable: int or None
        Degree of derivability of the function. None means unknown.

    Notes
    -----
    The derivability check is hardcoded into the kernel core and it is not
    possible to remove it afterwards by applying ``'derivable'`` again with a
    higher limit.

    """
    if isinstance(derivable, bool):
        derivable = sys.maxsize if derivable else 0
    elif not isinstance(derivable, numbers.Integral) or derivable < 0:
        raise ValueError(f'derivability degree {derivable!r} not valid')

    def error_func(current, n):
        raise ValueError(f'Took {current} derivatives > limit {n} on argument '
            'of a kernel. This error may be spurious if there are '
            'derivatives on values that define the input to the kernel, for '
            'example if a hyperparameter enters the calculation of x. To '
            'suppress the error, initialize the kernel with derivable=True.')

    def xtransf(x):
        if hasattr(x, 'dtype'):
            # this branch handles limit_derivatives not accepting non-jax types
            # because of being based on jax.custom_jvp; this restriction on
            # custom_jvp appeared in jax 0.4.17
            if x.dtype.names is not None:
                x = _array.StructuredArray(x) # structured arrays are not
                    # compatible with jax but common in lsqfitgp, so I wrap them
            elif not _jaxext.is_jax_type(x.dtype):
                return x # since anyway there would be an error if a derivative
                    # tried to pass through a non-jax type
        return _jaxext.limit_derivatives(x, n=derivable, error_func=error_func)

    return xtransf

    # TODO this system does not ignore additional derivatives that are not
    # taken by .transf('diff'). Plan:
    # - understand how to associate a jax transformation to a frame
    # - make a context manager with a global stack
    # - at initialization it takes the frames of the derivatives made by diff
    # - the context is around calling core in diff.newcore
    # - it adds the frames to the stack
    # - derivable asks the context manager class to find the frames in x
    # - raises if their number is above the limit

def _asfloat(x):
    return x.astype(_jaxext.float_type(x))

def diff_argparser(deriv):
    deriv = _Deriv.Deriv(deriv)
    return deriv if deriv else None

@functools.partial(CrossKernel.register_corelinop, argparser=diff_argparser)
def diff(core, xderiv, yderiv):
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

    # reparse derivatives because they could be None
    xderiv = _Deriv.Deriv(xderiv)
    yderiv = _Deriv.Deriv(yderiv)

    # wrapper of kernel with derivable arguments unpacked
    def f(x, y, *args, **kw):
        i = -1
        if not xderiv.implicit:
            for i, dim in enumerate(xderiv):
                x = x.at[dim].set(args[i])
        if not yderiv.implicit:
            for j, dim in enumerate(yderiv):
                y = y.at[dim].set(args[1 + i + j])
        return core(x, y, **kw)
    
    # last x index in case iteration on x does not run but does on y
    i = -1

    # derive w.r.t. first argument
    if xderiv.implicit:
        for _ in range(xderiv.order):
            f = _jaxext.elementwise_grad(f, 0)
    else:
        for i, dim in enumerate(xderiv):
            for _ in range(xderiv[dim]):
                f = _jaxext.elementwise_grad(f, 2 + i)

    # derive w.r.t. second argument
    if yderiv.implicit:
        for _ in range(yderiv.order):
            f = _jaxext.elementwise_grad(f, 1)
    else:
        for j, dim in enumerate(yderiv):
            for _ in range(yderiv[dim]):
                f = _jaxext.elementwise_grad(f, 2 + 1 + i + j)

    # check derivatives are ok for actual input arrays, wrap structured arrays
    def process_arg(x, deriv, pos):
        if x.dtype.names is not None:
            for dim in deriv:
                if dim not in x.dtype.names:
                    raise ValueError(f'derivative along missing field {dim!r} '
                        f'on {pos} argument')
                if not jnp.issubdtype(x.dtype[dim], jnp.number):
                    raise TypeError(f'derivative along non-numeric field '
                        f'{dim!r} on {pos} argument')
            return _array.StructuredArray(x)
        elif not deriv.implicit:
            raise ValueError('derivative on named fields with non-structured '
                f'array on {pos} argument')
        elif not jnp.issubdtype(x.dtype, jnp.number):
            raise TypeError(f'derivative along non-numeric array on '
                f'{pos} argument')
        return x
    
    def newcore(x, y, **kw):
        x = process_arg(x, xderiv, 'left')
        y = process_arg(y, yderiv, 'right')
                            
        args = []

        if not xderiv.implicit:
            for dim in xderiv:
                args.append(_asfloat(x[dim]))
        elif xderiv:
            x = _asfloat(x)
        
        if not yderiv.implicit:
            for dim in yderiv:
                args.append(_asfloat(y[dim]))
        elif yderiv:
            y = _asfloat(y)

        return f(x, y, *args, **kw)

    return newcore

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
        return lambda x, y, **kw: core(x, y, **kw) / jnp.sqrt(core(x, x, **kw) * core(y, y, **kw))
    elif dox:
        return lambda x, y, **kw: core(x, y, **kw) / jnp.sqrt(core(x, x, **kw))
    else:
        return lambda x, y, **kw: core(x, y, **kw) / jnp.sqrt(core(y, y, **kw))

@CrossKernel.register_corelinop
def cond(core, cond1, cond2, other):
    r"""

    Switch between two independent processes based on a condition.

    .. math::
        T(f, g)(x) = \begin{cases}
            f(x) & \text{if $\mathrm{cond}(x)$,} \\
            g(x) & \text{otherwise.}
        \end{cases}
        
    Parameters
    ----------
    cond1, cond2 : callable
        Function that is applied on an array of points and must return
        a boolean array with the same shape.
    other :
        Kernel of the process used where the condition is false.
    
    """
    def newcore(x, y, **kw):        
        xcond = cond1(x)
        ycond = cond2(y)
        r = jnp.where(xcond & ycond, core(x, y, **kw), other(x, y, **kw))
        return jnp.where((xcond ^ ycond).astype('u8'), 0, r)
        # the .astype('u8') fixes a weird bug with some old dependencies
    
    return newcore

    # TODO add a function `choose` to extend `cond`,
    # kernel0.linop('choose', kernel1, kernel2, ..., lambda x: x['index'])

AffineSpan.inherit_transf('maxdim')
AffineSpan.inherit_transf('derivable')

@functools.partial(AffineSpan.register_linop, transfname='loc')
def affine_loc(tcls, self, xloc, yloc):
    dynkw = dict(self.dynkw)
    newself = tcls.super_transf('loc', self, xloc, yloc)
    dynkw['lloc'] = dynkw['lloc'] + xloc * dynkw['lscale']
    dynkw['rloc'] = dynkw['rloc'] + yloc * dynkw['rscale']
    return newself._clone(self.__class__, dynkw=dynkw)

@functools.partial(AffineSpan.register_linop, transfname='scale')
def affine_scale(tcls, self, xscale, yscale):
    dynkw = dict(self.dynkw)
    newself = tcls.super_transf('scale', self, xscale, yscale)
    dynkw['lscale'] = dynkw['lscale'] * xscale
    dynkw['rscale'] = dynkw['rscale'] * yscale
    return newself._clone(self.__class__, dynkw=dynkw)
