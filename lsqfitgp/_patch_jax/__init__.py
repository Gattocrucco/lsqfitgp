# lsqfitgp/_patch_jax/__init__.py
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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

from jax.config import config
config.update("jax_enable_x64", True)

import traceback
import functools

import jax
from jax import core
from jax import lax
from jax import numpy as jnp
from jax import tree_util

from ._batcher import batchufunc
from ._fasthash import fasthash64, fasthash32

def makejaxufunc(ufunc, *derivs):
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
        class ResultDummy:
            shape = jnp.broadcast_shapes(*(arg.shape for arg in args))
            dtype = jnp.result_type(*args)
        return jax.pure_callback(ufunc, ResultDummy, *args, vectorized=True)

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
    def funderiv(*args):
        preargs = args[:argnum]
        postargs = args[argnum + 1:]
        def oneargfun(arg):
            args = preargs + (arg,) + postargs
            return fun(*args)
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
                    # why isn't this a subclass of the former like
                    # TracerBoolConversionError?
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

# TODO make stop_hessian work in reverse mode
# see jax issue #10994
# from jax.interpreters import ad
# from jax._src import ad_util
# ad.primitive_transposes[ad_util.stop_gradient_p] = lambda ct, _: [lax.stop_gradient(ct)]
# try with jax.jvp and lax.stop_gradient, I tried once and I failed but I did
# not try hard enough

@jax.custom_jvp
def stop_hessian(x):
    return x

@stop_hessian.defjvp
def stop_hessian_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    return x, lax.stop_gradient(x_dot)

def value_and_ops(f, *ops, has_aux=False, **kw):
    """
    Creates a function returning the values of f and its derivatives defined
    by stacking the operators in ops. Example:
    value_and_grad_and_hess = value_and_ops(f, jax.grad, jax.jacfwd)
    """
    if not ops:
        return f
    def fop(*args, **kw):
        y = f(*args, **kw)
        if has_aux:
            y, aux = y
            return y, (aux,)
        else:
            return y, ()
    def nextfop(fop):
        def nextfop(*args, **kw):
            y, aux = fop(*args, **kw)
            return y, aux + (y,)
        return nextfop
    for op in ops:
        fop = op(nextfop(fop), has_aux=True, **kw)
    @functools.wraps(f)
    def lastfop(*args, **kw):
        y, aux = fop(*args, **kw)
        if has_aux:
            return aux[1:] + (y,), aux[0]
        else:
            return aux + (y,)
    return lastfop

def float_type(*args):
    t = jnp.result_type(*args)
    return jnp.sin(jnp.empty(0, t)).dtype
