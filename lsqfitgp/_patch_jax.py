# lsqfitgp/_patch_jax.py
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
from scipy import special
from jax.scipy import special as jspecial

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

j0 = makejaxufunc(special.j0, lambda x: -j1(x))
j1 = makejaxufunc(special.j1, lambda x: (j0(x) - jv(2, x)) / 2.0)
jv = makejaxufunc(special.jv, None, lambda v, z: jvp(v, z, 1))
jvp = makejaxufunc(special.jvp, None, lambda v, z, n: jvp(v, z, n + 1), None)
iv = makejaxufunc(special.iv, None, lambda v, z: ivp(v, z, 1))
ivp = makejaxufunc(special.ivp, None, lambda v, z, n: ivp(v, z, n + 1), None)
kv = makejaxufunc(special.kv, None, lambda v, z: kvp(v, z, 1))
kvp = makejaxufunc(special.kvp, None, lambda v, z, n: kvp(v, z, n + 1), None)
ci = makejaxufunc(lambda x: special.sici(x)[1], lambda x: jnp.cos(x) / x)

# See jax #1870, #2466, #9956, #11002 and
# https://github.com/josipd/jax/blob/master/jax/experimental/jambax.py
# to implement special functions in jax

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

@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def taylor(coefgen, args, n, m, x):
    c = coefgen(n, n + m, *args)
    k = jnp.arange(n, n + m)
    c = c * jnp.exp(jspecial.gammaln(1 + k) - jspecial.gammaln(1 + k - n))
    return jnp.polyval(c[::-1], x)

@taylor.defjvp
def taylor_jvp(coefgen, args, n, m, primals, tangents):
    x, = primals
    xt, = tangents
    return taylor(coefgen, args, n, m, x), taylor(coefgen, args, n + 1, m, x) * xt

def coefgen_sinc(s, e):
    m = jnp.arange(s, e)
    return (-1) ** m / jnp.exp(jspecial.gammaln(2 + 2 * m))

def sinc(x):
    nearzero = taylor(coefgen_sinc, (), 0, 6, jnp.square(jnp.pi * x))
    normal = jnp.sinc(x)
    return jnp.where(jnp.abs(x) < 1e-1, nearzero, normal)

def sgngamma(x):
    return jnp.where((x > 0) | (x % 2 < 1), 1, -1)

def gamma(x):
    return sgngamma(x) * jnp.exp(jspecial.gammaln(x))

# def coefgen_jvmod(s, e, nu):
#     m = jnp.arange(s, e)
#     u = 1 + m + nu
#     return sgngamma(u) * (-1) ** m / jnp.exp(jspecial.gammaln(1 + m) + jspecial.gammaln(u))

@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def jvmodx2(nu, x2):
    x = jnp.sqrt(x2)
    normal = (x / 2) ** -nu * jv(nu, x)
    # nearzero = taylor(coefgen_jvmod, (nu,), 0, 5, x2 / 4)
    # return jnp.where(x2 < 1e-4, nearzero, normal)
    return jnp.where(x2, normal, 1 / gamma(nu + 1))

# (1/x d/dx)^m (x^-v J_v(x)) = (-1)^m x^-(v+m) J_v+m(x)
#                                   (Abramowitz and Stegun, p. 361, 9.1.30)
# --> 1/x d/dx (x^-v J_v(x)) = -x^-(v+1) J_v+1(x)
# --> d/dx (x^-v J_v(x)) = -x^-v J_v+1(x)
# --> d/ds ~J_v(s) =
#   = d/ds (√s/2)^-v J_v(√s) =
#   = 2^v d/ds √s^-v J_v(√s) =
#   = -2^v √s^-v J_v+1(√s) 1/2√s =
#   = -2^(v-1) √s^-(v+1) J_v+1(√s) =
#   = -1/4 (√s/2)^(v+1) J_v+1(√s) =
#   = -1/4 ~J_v+1(s)

@jvmodx2.defjvp
def jvmodx2_jvp(nu, primals, tangents):
    x2, = primals
    x2t, = tangents
    return jvmodx2(nu, x2), -x2t * jvmodx2(nu + 1, x2) / 4

# def coefgen_ivmod(s, e, nu):
#     m = jnp.arange(s, e)
#     u = 1 + m + nu
#     return sgngamma(u) * jnp.exp(-jspecial.gammaln(1 + m) - jspecial.gammaln(u))
#
# def ivmodx2_nearzero(nu, x2):
#     return taylor(coefgen_ivmod, (nu,), 0, 5, x2 / 4)
#
# def kvmodx2_nearzero(nu, x2):
#     factor = jnp.pi / (2 * jnp.sin(jnp.pi * nu))
#     return factor * (ivmodx2_nearzero(-nu, x2) - (x2 / 4) ** nu * ivmodx2_nearzero(nu, x2))

@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def kvmodx2(nu, x2, norm_offset=0):
    # assert int(nu) != nu, nu
    x = jnp.sqrt(x2)
    normal = 2 / gamma(nu + norm_offset) * (x / 2) ** nu * kv(nu, x)
    # nearzero = kvmodx2_nearzero(nu, x2)
    # return jnp.where(x2 < 1e-4, nearzero, normal)
    return jnp.where(x2, normal, 1 / jnp.prod(nu + jnp.arange(norm_offset)))

# d/dx (x^v Kv(x)) = -x^v Kv-1(x)       (Abrahamsen 1997, p. 43)
# d/dx ~Kv(x) =
#   = d/dx (√x/2)^v Kv(√x) =
#   = 2^-v d/dx (√x)^v Kv(√x) =
#   = -2^-v (√x)^v Kv-1(√x) 1/(2√x) =
#   = -2^-(v+1) √x^(v-1) Kv-1(√x) =
#   = -1/4 (√x/2)^(v-1) Kv-1(√x) =
#   = -1/4 ~Kv-1(x)

@kvmodx2.defjvp
def kvmodx2_jvp(nu, norm_offset, primals, tangents):
    x2, = primals
    x2t, = tangents
    primal = kvmodx2(nu, x2, norm_offset)
    tangent = -x2t * kvmodx2(nu - 1, x2, norm_offset + 1) / 4
    return primal, tangent
    
# TODO implement kvmodx2_hi to replace maternp

def tree_all(predicate, *trees):
    pred = tree_util.tree_map(predicate, *trees)
    return tree_util.tree_reduce(lambda acc, p: acc and p, pred, True)

def companion(a):
    """mimics scipy.linalg.companion
    coefficients ordered high to low"""
    a = jnp.asarray(a)
    assert a.ndim == 1 and a.size >= 2 and a[0] != 0, a
    row = -a[1:] / (1.0 * a[0])
    n = a.size
    col_indices = jnp.arange(n - 2)
    row_indices = 1 + col_indices
    c = jnp.zeros((n - 1, n - 1), row.dtype)
    return c.at[0, :].set(row).at[row_indices, col_indices].set(1)

def polyroots(c):
    """mimics numpy.polynomial.polynomial.polyroots
    coefficients low to high"""
    m = companion(c[::-1])
    return jnp.sort(jnp.linalg.eigvals(m.T))
    # transpose to mimic polyroots, which says that this particular ordering
    # increases precision, but I'm not sure the transposition actually helps,
    # it's probably more about the difference between scipy's companion and
    # numpy's polycompanion
