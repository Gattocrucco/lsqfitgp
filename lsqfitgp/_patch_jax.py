# lsqfitgp/_patch_jax.py
#
# Copyright (c) 2022, Giacomo Petrillo
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

import functools

import jax
from jax import core
from jax import lax
from jax import numpy as jnp
from jax.interpreters import ad
from jax.interpreters import batching
from jax import tree_util
from jax._src import ad_util

def makejaxufunc(ufunc, *derivs):
    # TODO use jax.something.standard_primitive
    
    prim = core.Primitive(ufunc.__name__)
    
    @functools.wraps(ufunc)
    def func(*args):
        return prim.bind(*args)

    @prim.def_impl
    def impl(*args):
        return ufunc(*args)

    @prim.def_abstract_eval
    def abstract_eval(*args):
        shape = jnp.broadcast_shapes(*(x.shape for x in args))
        dtype = jnp.result_type(*(x.dtype for x in args))
        return core.ShapedArray(shape, dtype)

    jvps = (
        None if d is None
        else (lambda d: lambda g, *args: d(*args) * g)(d)
        for d in derivs
    )
    ad.defjvp(prim, *jvps)

    batching.defbroadcasting(prim)
    
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

def _isconcrete(x):
    return not isinstance(x, core.Tracer) or isinstance(x.aval, core.ConcreteArray)

def isconcrete(*args):
    children, _ = tree_util.tree_flatten(args)
    return all(map(_isconcrete, children))

def _concrete(x):
    # This is not needed from simple operations with operators on scalars.
    # It is needed for functions that expect a numpy array, and then you
    # have to use the actual numpy function instead of the jax version,
    # because during jit tracing all jax functions produce abstract arrays.
    return x.aval.val if isinstance(x, core.Tracer) else x
    # TODO maybe use jax.core.concrete_aval? But I'm not sure of what it does

def concrete(*args):
    result = tree_util.tree_map(_concrete, args)
    if len(args) == 1:
        result, = result
    return result

# TODO make stop_hessian work in reverse mode
# see jax issue #10994
# ad.primitive_transposes[ad_util.stop_gradient_p] = lambda ct, _: [lax.stop_gradient(ct)]

@jax.custom_jvp
def stop_hessian(x):
    return x

@stop_hessian.defjvp
def stop_hessian_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    return x, lax.stop_gradient(x_dot)

def value_and_ops(f, *ops, has_aux=False, **kw): # pragma: no cover
    # currently not used, written for empbayes_fit
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

def tree_all(predicate, *trees):
    pred = tree_util.tree_map(predicate, *trees)
    return tree_util.tree_reduce(lambda acc, p: acc and p, pred, True)

def float_type(*args):
    t = jnp.result_type(*args)
    return t if jnp.issubdtype(t, jnp.inexact) else jnp.float64
    # TODO in jax somewhere there is _promote_dtypes_inexact, look it up
