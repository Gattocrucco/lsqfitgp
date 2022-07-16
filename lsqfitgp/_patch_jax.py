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
    """
    coefgen : function = start, end -> taylor coefficients for powers start:end
    args : tuple = additional arguments to coefgen
    n : int = derivation order
    m : int = number of coefficients used
    x : argument
    """
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

@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def jvmodx2(nu, x2):
    x = jnp.sqrt(x2)
    normal = (x / 2) ** -nu * jv(nu, x)
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

@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def kvmodx2(nu, x2, norm_offset=0):
    x = jnp.sqrt(x2)
    normal = 2 / gamma(nu + norm_offset) * (x / 2) ** nu * kv(nu, x)
    atzero = 1 / jnp.prod(nu + jnp.arange(norm_offset))
    atzero = jnp.where(nu > 0, atzero, 1) # for nu < 0 the correct limit
                                          # would be inf, but in practice it
                                          # gets cancelled by a stronger 0
                                          # when taking derivatives of Matern
                                          # and this is a cheap way to avoid
                                          # nans
    return jnp.where(x2, normal, atzero)

# d/dx (x^v Kv(x)) = -x^v Kv-1(x)       (Abrahamsen 1997, p. 43)
# d/ds ~Kv(s) =
#   = d/ds (√s/2)^v Kv(√s) =
#   = 2^-v d/ds (√s)^v Kv(√s) =
#   = -2^-v (√s)^v Kv-1(√s) 1/(2√s) =
#   = -2^-(v+1) √s^(v-1) Kv-1(√s) =
#   = -1/4 (√s/2)^(v-1) Kv-1(√s) =
#   = -1/4 ~Kv-1(s)

@kvmodx2.defjvp
def kvmodx2_jvp(nu, norm_offset, primals, tangents):
    x2, = primals
    x2t, = tangents
    primal = kvmodx2(nu, x2, norm_offset)
    tangent = -x2t * kvmodx2(nu - 1, x2, norm_offset + 1) / 4
    return primal, tangent

@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def kvmodx2_hi(x2, p):
    # nu = p + 1/2, p integer >= 0
    x = jnp.sqrt(x2)
    poly = 1
    for k in reversed(range(p)):
        c_kp1_over_ck = (p - k) / ((2 * p - k) * (k + 1))
        poly = 1 + poly * c_kp1_over_ck * 2 * x
    return jnp.exp(-x) * poly

@kvmodx2_hi.defjvp
def kvmodx2_hi_jvp(p, primals, tangents):
    x2, = primals
    x2t, = tangents
    primal = kvmodx2_hi(x2, p)
    if p == 0:
        x = jnp.sqrt(x2)
        tangent = -x2t * jnp.exp(-x) / (2 * x) # <--- problems!
    else:
        tangent = -x2t / (p - 1/2) * kvmodx2_hi(x2, p - 1) / 4
    return primal, tangent

def tree_all(predicate, *trees):
    pred = tree_util.tree_map(predicate, *trees)
    return tree_util.tree_reduce(lambda acc, p: acc and p, pred, True)

@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def expn_imag(n, x):
    """
    Compute E_n(-ix), n integer >= 2, x real >= 0
    """
    
    # expn_imag_smallx loses accuracy due to cancellation between two terms
    # ~ x^n-2, while the result ~ x^-1, thus the relative error ~ x^-1/x^n-2 =
    # = x^-(n-1)
    #
    # error of expn_imag_smallx: eps z^n-1 E_1(z) / Gamma(n) ~
    #                            ~ eps z^n-2 / Gamma(n)
    #
    # error of expn_asymp: e^-z/z (n)_nt e^z/z^nt-1 E_n+nt(z) =
    #                      = (n)_nt / z^nt E_n+nt(z) ~
    #                      ~ (n)_nt / z^nt+1
    #
    # set the errors equal:
    #   eps z^n-2 / Gamma(n) = (n)_nt / z^nt+1  -->
    #   -->  z = (Gamma(n + nt) / eps)^1/(n+nt-1)
    
    # TODO improve accuracy at large n, it is probably sufficient to use
    # something like softmin(1/(n-1), 1/x) e^-ix, where the softmin scale
    # increases with n (how?)
    
    dt = jnp.empty(0).dtype
    if dt == jnp.float32:
        nt = 10 # TODO optimize to raise maximum n
    else:
        nt = 20 # TODO optimize to raise maximum n
    eps = jnp.finfo(dt).eps
    knee = (special.gamma(n + nt) / eps) ** (1 / (n + nt - 1))
    small = expn_imag_smallx(n, x)
    large = expn_asymp(n, -1j * x, nt)
    return jnp.where(x < knee, small, large)

@expn_imag.defjvp
def expn_imag_jvp(n, primals, tangents):
    
    # DLMF 8.19.13
    
    x, = primals
    xt, = tangents
    return expn_imag(n, x), xt * 1j * expn_imag(n - 1, x)

def expn_imag_smallx(n, x):
        
    # DLMF 8.19.7
    
    k = jnp.arange(n).astype(float)
    fact = jnp.cumprod(k.at[0].set(1))
    n_1fact = fact[-1]
    ix = 1j * x
    E_1 = exp1_imag(x) # E_1(-ix)
    E_1 = jnp.where(x, E_1, 0) # Re E_1(-ix) ~ log(x) for x -> 0
    part1 = ix ** (n - 1) * E_1
    coefs = fact[:-1][(...,) + (None,) * ix.ndim]
    part2 = jnp.exp(ix) * jnp.polyval(coefs, ix)
    return (part1 + part2) / n_1fact
    
    # TODO to make this work with jit n, since the maximum n is something
    # like 30, I can always compute all the terms and set some of them to zero

def poch(x, k):
    return jnp.exp(jspecial.gammaln(x + k) - jspecial.gammaln(x)) # DLMF 5.2.5
    
def expn_asymp_coefgen(s, e, n):
    k = jnp.arange(s, e)
    return (-1) ** k * poch(n, k)

def expn_asymp(n, z, nt):
    """
    Compute E_n(z) for large |z|, |arg z| < 3/2 π. `nt` is the number of terms
    used in the asymptotic series.
    """
    
    # DLMF 8.20.2
    
    invz = 1 / z
    return jnp.exp(-z) * invz * taylor(expn_asymp_coefgen, (n,), 0, nt, invz)

_si_num = [
    1,
    -4.54393409816329991e-2, # x^2
    1.15457225751016682e-3, # x^4
    -1.41018536821330254e-5, # x^6
    9.43280809438713025e-8, # x^8
    -3.53201978997168357e-10, # x^10
    7.08240282274875911e-13, # x^12
    -6.05338212010422477e-16, # x^14
]

_si_denom = [
    1,
    1.01162145739225565e-2, # x^2
    4.99175116169755106e-5, # x^4
    1.55654986308745614e-7, # x^6
    3.28067571055789734e-10, # x^8
    4.5049097575386581e-13, # x^10
    3.21107051193712168e-16, # x^12
]

_ci_num = [
    -0.25,
    7.51851524438898291e-3, # x^2
    -1.27528342240267686e-4, # x^4
    1.05297363846239184e-6, # x^6
    -4.68889508144848019e-9, # x^8
    1.06480802891189243e-11, # x^10
    -9.93728488857585407e-15, # x^12
]

_ci_denom = [
    1,
    1.1592605689110735e-2, # x^2
    6.72126800814254432e-5, # x^4
    2.55533277086129636e-7, # x^6
    6.97071295760958946e-10, # x^8
    1.38536352772778619e-12, # x^10
    1.89106054713059759e-15, # x^12
    1.39759616731376855e-18, # x^14
]

_f_num = [
    1,
    7.44437068161936700618e2, # x^-2
    1.96396372895146869801e5, # x^-4
    2.37750310125431834034e7, # x^-6
    1.43073403821274636888e9, # x^-8
    4.33736238870432522765e10, # x^-10
    6.40533830574022022911e11, # x^-12
    4.20968180571076940208e12, # x^-14
    1.00795182980368574617e13, # x^-16
    4.94816688199951963482e12, # x^-18
    -4.94701168645415959931e11, # x^-20
]

_f_denom = [
    1,
    7.46437068161927678031e2, # x^-2
    1.97865247031583951450e5, # x^-4
    2.41535670165126845144e7, # x^-6
    1.47478952192985464958e9, # x^-8
    4.58595115847765779830e10, # x^-10
    7.08501308149515401563e11, # x^-12
    5.06084464593475076774e12, # x^-14
    1.43468549171581016479e13, # x^-16
    1.11535493509914254097e13, # x^-18
]

_g_num = [
    1,
    8.1359520115168615e2, # x^-2
    2.35239181626478200e5, # x^-4
    3.12557570795778731e7, # x^-6
    2.06297595146763354e9, # x^-8
    6.83052205423625007e10, # x^-10
    1.09049528450362786e12, # x^-12
    7.57664583257834349e12, # x^-14
    1.81004487464664575e13, # x^-16
    6.43291613143049485e12, # x^-18
    -1.36517137670871689e12, # x^-20
]

_g_denom = [
    1,
    8.19595201151451564e2, # x^-2
    2.40036752835578777e5, # x^-4
    3.26026661647090822e7, # x^-6
    2.23355543278099360e9, # x^-8
    7.87465017341829930e10, # x^-10
    1.39866710696414565e12, # x^-12
    1.17164723371736605e13, # x^-14
    4.01839087307656620e13, # x^-16
    3.99653257887490811e13, # x^-18
]

def _si_smallx(x):
    """ Compute Si(x) = int_0^x dt sin t / t, for x < 4"""
    x2 = jnp.square(x)
    num = jnp.polyval(jnp.array(_si_num[::-1]), x2)
    denom = jnp.polyval(jnp.array(_si_denom[::-1]), x2)
    return x * num / denom

def _minus_cin_smallx(x):
    """ Compute -Cin(x) = int_0^x dt (cos t - 1) / t, for x < 4 """
    x2 = jnp.square(x)
    num = jnp.polyval(jnp.array(_ci_num[::-1]), x2)
    denom = jnp.polyval(jnp.array(_ci_denom[::-1]), x2)
    return x2 * num / denom

def _ci_smallx(x):
    """ Compute Ci(x) = -int_x^oo dt cos t / t, for x < 4 """
    gamma = 0.57721566490153286060
    return gamma + jnp.log(x) + _minus_cin_smallx(x)

def _f_largex(x):
    """ Compute f(x) = int_0^oo dt sin t / (x + t), for x > 4 """
    x2 = 1 / jnp.square(x)
    num = jnp.polyval(jnp.array(_f_num[::-1]), x2)
    denom = jnp.polyval(jnp.array(_f_denom[::-1]), x2)
    return num / denom / x

def _g_largex(x):
    """ Compute g(x) = int_0^oo dt cos t / (x + t), for x > 4 """
    x2 = 1 / jnp.square(x)
    num = jnp.polyval(jnp.array(_g_num[::-1]), x2)
    denom = jnp.polyval(jnp.array(_g_denom[::-1]), x2)
    return x2 * num / denom

def _exp1_imag_smallx(x):
    """ Compute E_1(-ix), for x < 4 """
    return -_ci_smallx(x) + 1j * (jnp.pi / 2 - _si_smallx(x))

def _exp1_imag_largex(x):
    """ Compute E_1(-ix), for x > 4 """
    s = jnp.sin(x)
    c = jnp.cos(x)
    f = _f_largex(x)
    g = _g_largex(x)
    real = -f * s + g * c
    imag = f * c + g * s
    return real + 1j * imag  # e^ix (g + if)

@jax.jit
def exp1_imag(x):
    """
    Compute E_1(-ix) = int_1^oo dt e^ixt / t, for x > 0
    Reference: Rowe et al. (2015, app. B)
    """
    return jnp.where(x < 4, _exp1_imag_smallx(x), _exp1_imag_largex(x))
    
    # TODO This is 40x faster than special.exp1(-1j * x) and 2x than
    # special.sici(x), and since the jit has to run (I'm guessing) through both
    # branches of jnp.where, a C/Cython implementation would be 4x faster. Maybe
    # PR it to scipy for sici, after checking the accuracy against mpmath and
    # the actual C performance.

    # Do Padé approximants work for complex functions?

@jax.custom_jvp
def ci(x):
    return -exp1_imag(x).real

@ci.defjvp
def _ci_jvp(primals, tangents):
    x, = primals
    xt, = tangents
    return ci(x), xt * jnp.cos(x) / x
