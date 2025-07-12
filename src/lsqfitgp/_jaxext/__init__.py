# lsqfitgp/_jaxext/__init__.py
#
# Copyright (c) 2022, 2023, 2025, Giacomo Petrillo
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

import traceback
import functools

import jax
from jax import numpy as jnp

from ._batcher import batchufunc
from ._fasthash import fasthash64, fasthash32

def makejaxufunc(ufunc, *derivs, excluded=None, floatcast=False):
    """
    
    Wrap a numpy ufunc to add jax support.
    
    Parameters
    ----------
    ufunc : callable
        Elementwise function following numpy broadcasting and type promotion
        rules. Keyword arguments not supported.
    derivs : sequence of callable
        Derivatives of the function w.r.t. each positional argument, with the
        same signature as `ufunc`. Pass None to indicate a missing derivative.
        There must be as many derivatives as the arguments to `ufunc`.
    excluded : sequence of int, optional
        The indices of arguments that are not broadcasted.
    floatcast : bool, default False
        If True, cast all arguments to float before calling the ufunc.
    
    Return
    ------
    func : callable
        Wrapped `ufunc`. Supports jit, but the calculation is performed on cpu.
    
    """

    nondiff_argnums = [i for i, d in enumerate(derivs) if d is None]
    
    @functools.wraps(ufunc)
    @functools.partial(jax.custom_jvp, nondiff_argnums=nondiff_argnums)
    def func(*args):
        args = tuple(map(jnp.asarray, args))
        if floatcast:
            flt = float_type(*args)
            args = tuple(a.astype(flt) for a in args)
        return pure_callback_ufunc(ufunc, jnp.result_type(*args), *args, excluded=excluded)

    @func.defjvp
    def func_jvp(*allargs):
        ndargs = allargs[:-2]
        dargs = allargs[-2]
        dargst = allargs[-1]
        
        itnd = iter(ndargs)
        itd = iter(dargs)
        args = [next(itnd) if d is None else next(itd) for d in derivs]
        
        result = func(*args)
        tangent = sum([
            d(*args) * t for t, d in
            zip(dargst, (d for d in derivs if d is not None))
        ])
        return result, tangent
    
    return func

def elementwise_grad(fun, argnum=0):
    assert int(argnum) == argnum and argnum >= 0, argnum
    @functools.wraps(fun)
    def funderiv(*args, **kw):
        preargs = args[:argnum]
        postargs = args[argnum + 1:]
        def oneargfun(arg):
            args = preargs + (arg,) + postargs
            return fun(*args, **kw)
        primal = args[argnum]
        shape = getattr(primal, 'shape', ())
        dtype = getattr(primal, 'dtype', type(primal))
        tangent = jnp.ones(shape, dtype)
        primal_out, tangent_out = jax.jvp(oneargfun, (primal,), (tangent,))
        return tangent_out
    return funderiv

class skipifabstract:
    """
    Context manager to try to do all operations eagerly even during jit, and
    skip entirely if it is not possible.
    """
    # I feared this would be slow because of the slow jax exception handling,
    # but %timeit suggests it isn't

    ENSURE_COMPILE_TIME_EVAL = True
    ENABLED = True
    
    def __enter__(self):
        if self.ENSURE_COMPILE_TIME_EVAL and self.ENABLED:
            self.mgr = jax.ensure_compile_time_eval()
            self.mgr.__enter__()
    
    def __exit__(self, exc_type, exc_value, tb):
        if not self.ENABLED:
            return
        exit = None
        if self.ENSURE_COMPILE_TIME_EVAL:
            exit = self.mgr.__exit__(exc_type, exc_value, tb)
        ignorable_error = (
            exc_type is not None
            and issubclass(exc_type, (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerArrayConversionError,
                    # TODO why isn't this a subclass of the former like
                    # TracerBoolConversionError? Open an issue
            ))
        )
        if exit or ignorable_error:
            return True
        
        weird_cond = exc_type is IndexError and (
            traceback.extract_tb(tb)[-1].name in ('arg_info_pytree', '_origin_msg'),
        )
        if weird_cond: # pragma: no cover
            # TODO this ignores a jax internal bug I don't understand, appears
            # in examples/pdf4.py
            return True

def float_type(*args):
    t = jnp.result_type(*args)
    return jnp.sin(jnp.empty(0, t)).dtype
    # numpy does this with common_type, but that supports only arrays, not
    # dtypes in the input. jnp.common_type is not defined.

def is_jax_type(dtype):
    dtype = jnp.dtype(dtype)
    try:
        jnp.empty(0, dtype)
        return True
    except TypeError as e:
        if 'JAX only supports number' in str(e):
            return False
        raise

def pure_callback_ufunc(callback, dtype, *args, excluded=None, **kwargs):
    """ version of jax.pure_callback that deals correctly with ufuncs,
    see https://github.com/google/jax/issues/17187 """
    if excluded is None:
        excluded = ()
    shape = jnp.broadcast_shapes(*(
        a.shape
        for i, a in enumerate(args)
        if i not in excluded
    ))
    ndim = len(shape)
    padded_args = [
        a if i in excluded
        else jnp.expand_dims(a, tuple(range(ndim - a.ndim)))
        for i, a in enumerate(args)
    ]
    result = jax.ShapeDtypeStruct(shape, dtype)
    return jax.pure_callback(callback, result, *padded_args, vmap_method='expand_dims', **kwargs)

    # TODO when jax solves this, check version and piggyback on original if new

def limit_derivatives(x, n, error_func=None):
    """
    Limit the number of derivatives that goes through a value.

    Parameters
    ----------
    x : array_like
        The value.
    n : int
        The maximum number of derivatives allowed. Must be an integer.
    error_func : callable, optional
        A function that takes (derivatives taken, n) and returns an exception.

    Return
    ------
    x : array_like
        The value, unchanged.
    """
    assert n == int(n)
    if error_func is None:
        def error_func(current, n):
            return ValueError(f'took {current} derivatives > limit {n}')
    return _limit_derivatives_impl(0, n, error_func, x)

@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _limit_derivatives_impl(current, limit, func, x):
    if current > limit:
        raise func(current, limit)
    return x

@_limit_derivatives_impl.defjvp
def _limit_derivatives_impl_jvp(current, limit, func, primals, tangents):
    x, = primals
    xdot, = tangents
    return _limit_derivatives_impl(current + 1, limit, func, x), xdot
