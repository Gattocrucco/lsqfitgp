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
import math

import jax
from jax import core
from jax import lax
from jax import numpy as jnp
import numpy
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
    
    k = jnp.arange(n)
    fact = jnp.cumprod(k.at[0].set(1), dtype=float)
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
    # TODO does not handle properly special cases with x and/or k nonpositive
    # integers
    
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

@jax.custom_jvp
@jax.jit
def expm1x(x):
    """
    Compute accurately exp(x) - 1 - x = x^2/2 1F1(1, 3, x).
    """
    n = 10 if x.dtype == jnp.float32 else 17
    k = jnp.arange(2, n + 1)
    f = jnp.cumprod(k)
    coef = jnp.array(1, x.dtype) / f[::-1]
    smallx = x * x * jnp.polyval(coef, x, unroll=n)
    return jnp.where(jnp.abs(x) < 1, smallx, jnp.expm1(x) - x)

@expm1x.defjvp
def _expm1x_jvp(p, t):
    x, = p
    xt, = t
    return expm1x(x), jnp.expm1(x) * xt

def _float_type(*args):
    t = jnp.result_type(*args)
    return t if jnp.issubdtype(t, jnp.inexact) else jnp.float64
    # TODO in jax somewhere there is _promote_dtypes_inexact, look it up

def _hurwitz_zeta_series(m, x, a1, onlyeven=False, onlyodd=False, skipterm=None):
    """
    hurwitz zeta(s = m + x, a = 1 - a1) with integer m
    meant to be used with |x| ≤ 1/2, but no actual restriction
    assuming -S <= s <= 0 and |a1| <= 1/2 with S ~ some decade
    https://dlmf.nist.gov/25.11.E10
    """
    
    # decide number of terms to sum
    t = _float_type(m, x, a1)
    nmax = _hze_nmax(t)
    n = jnp.arange(nmax + 1)
    
    # make arguments broadcastable with n
    x = x[..., None]
    m = m[..., None]
    a1 = a1[..., None]
    if skipterm is not None:
        skipterm = skipterm[..., None]
    
    # compute pochhammer symbol, factorial and power terms
    nm = n + m
    ns1 = nm - 1 + x # = n + s - 1
    ns1_limit = jnp.where(ns1 == 0, 1, ns1) # pochhammer zero cancels zeta pole
    ns1_limit = jnp.where(ns1 == 1, 0, ns1_limit)
    # TODO this == 1 worries me, maybe sometimes it's violated, use shift
    factor = jnp.cumprod((ns1_limit * a1 / n).at[..., 0].set(1), -1, t)
    
    # handle tweaks to the series
    if onlyeven:
        sl = slice(None, None, 2)
    elif onlyodd:
        sl = slice(1, None, 2)
    if onlyeven or onlyodd:
        n = n[sl]
        nm = nm[..., sl]
        ns1 = ns1[..., sl]
        factor = factor[..., sl]
    if skipterm is not None:
        factor = jnp.where(n == skipterm, 0, factor)
    
    # compute zeta term
    zet = zeta(x, nm) # = zeta(n + s)
    zet_limit = jnp.where(ns1 == 0, 1, zet) # pole cancelled by pochhammer
    
    # sum series
    kw = dict(precision=lax.Precision.HIGHEST)
    series = jnp.matmul(factor[..., None, :], zet_limit[..., :, None], **kw)
    return series.squeeze((-2, -1))

def _hze_nmax(t):
    minz = 0.0037 # = min(2 gamma(s) / (2 pi)^s) for s <= 0
    return int(math.ceil(-math.log2(jnp.finfo(t).eps * minz)))

def _dds_hzs_onlyevenodd_evenodds(m, a1, even):
    """
    Derivative w.r.t. x of _hurwitz_zeta_series(m, x, a1, onlyeven/odd=True)
    with negative even/odd integer m at x = 0.
    """

    # decide number of terms to sum
    t = _float_type(m, a1)
    nmax = _hze_nmax(t)
    n = jnp.arange(nmax + 1)

    # make arguments broadcastable with n
    a1, m = jnp.broadcast_arrays(a1, m) # needed for take_along_axis
    m = m[..., None]
    a1 = a1[..., None]

    # common calculations
    ns = n + m
    ns1 = ns - 1
    negs = -m
    pfa = jnp.cumprod((ns1 * a1 / n).at[..., 0].set(1), -1, t)
    n_fact = jnp.cumprod(n.at[0].set(1), -1, t)
    s_fact = n_fact.at[negs].get(mode='fill')
    base = jnp.repeat(a1, n.size, -1).at[..., 0].set(1)
    pow_a1_n = jnp.cumprod(base, -1, t)

    # take even/odd terms
    start = 0 if even else 1
    sl = slice(start, None, 2)
    ns = ns[..., sl]
    pfa = pfa[..., sl]
    ns1 = ns1[..., sl]

    # terms with n < -s
    dds1 = pfa * _zeta_deriv_evens(ns)

    # term with n = -s
    harm = jnp.cumsum((1 / n).at[0].set(0))
    harm_s = harm.at[negs].get(mode='fill')
    pow_a1_negs = jnp.take_along_axis(pow_a1_n, negs, -1)
    dds2 = (1/2 * harm_s + _zeta_deriv_evens(0)) * pow_a1_negs

    # terms with n > -s
    gamma_ns = n_fact.at[ns1].get(mode='fill')
    dds3 = gamma_ns / n_fact[sl] * s_fact * zeta(0, ns) * pow_a1_n[..., sl]
    
    if not even:
        dds2 = -dds2
        dds3 = -dds3

    # sum series
    n = n[sl]
    terms = jnp.where(n < negs, dds1, jnp.where(n > negs, dds3, dds2))
    return jnp.sum(terms, -1)

def _zeta_deriv_evens(s):
    t = _float_type(s)
    lut = jnp.array(_zeta_deriv_evens_lut, t)
    lut = jnp.pad(lut, (1, 1), constant_values=jnp.nan)
    index = (s - _zde_lut_start) // 2
    index = jnp.clip(index + 1, 0, lut.size - 1)
    return lut.at[index].get(mode='clip')

_zde_lut_start = -12

_zeta_deriv_evens_lut = [   # = _gen_zeta_deriv_evens_lut(-12, 1)
    0.063270583341463,
    -0.018929926338140373,
    0.008316161985602248,
    -0.005899759143515937,
    0.007983811450268625,
    -0.03044845705839327,
    -0.9189385332046728,
]

def _gen_zeta_deriv_evens_lut(start, stop):
    import mpmath as mp
    with mp.workdps(32):
        return [float(mp.diff(mp.zeta, s)) for s in range(start, stop, 2)]
    
# @jax.jit
def hurwitz_zeta(s, a):
    """
    For 0 <= a <= 1 and -S <= s <= 0 with S not too large
    """
    s = jnp.asarray(s)
    a = jnp.asarray(a)
    
    cond = a < 1/2  # do a + 1 to bring a closer to 1
    a1 = jnp.where(cond, -a, 1. - a)
    zero = jnp.array(0)
    zeta = _hurwitz_zeta_series(zero, s, a1)  # https://dlmf.nist.gov/25.11.E10
    zeta += jnp.where(cond, a ** -s, 0)  # https://dlmf.nist.gov/25.11.E3
    return zeta

# @functools.partial(jax.jit, static_argnums=(2,))
def periodic_zeta(x, s, imag=False):
    """
    compute F(x,s) = Li_s(e^2πix) for real s > 1, real x
    """
    
    x = jnp.asarray(x)
    s = jnp.asarray(s)
    
    # decide boundary for large/small s implementation
    t = _float_type(x, s)
    eps = jnp.finfo(t).eps
    nmax = 50
    larges = math.ceil(-math.log(eps) / math.log(nmax)) # 1/nmax^s < eps
    
    z_smalls = _periodic_zeta_smalls(x, s, imag)
    z_larges = _periodic_zeta_larges(x, s, nmax, imag)

    return jnp.where(s < larges, z_smalls, z_larges)

def _standard_x(x):
    """ bring x in [1, 1/2] by modulus and reflection """
    x %= 1
    neg = x > 1/2
    return neg, jnp.where(neg, 1 - x, x)

def _periodic_zeta_larges(x, s, nmax, imag):
    """ https://dlmf.nist.gov/25.13.E1 """

    t = _float_type(x, s)
    s = s.astype(t) # avoid n^s overflow with integer s
    n = jnp.arange(1, nmax + 1)
    neg, nx = _standard_x(n * x[..., None])
    func = jnp.sin if imag else jnp.cos
    terms = func(2 * jnp.pi * nx) / n ** s[..., None]
    if imag:
        terms *= jnp.where(neg, -1, 1)
    return jnp.sum(terms, -1)

def _periodic_zeta_smalls(x, s, imag):
    """
    https://dlmf.nist.gov/25.11.E10 and https://dlmf.nist.gov/25.11.E3 expanded
    into https://dlmf.nist.gov/25.13.E2
    """
    neg, x = _standard_x(x) # x in [0, 1/2]

    s1 = 1 - s  # < 0
    q = -jnp.around(s1).astype(int)
    a = s1 + q
    # now s1 == -q + a with q integer >= 0 and |a| <= 1/2
    
    pi = (2 * jnp.pi) ** -s1
    gam = gamma(s1)
    func = _sin_pi2 if imag else _cos_pi2
    pha = func(-q, a) # = sin/cos(π/2 s1), numerically accurate for small a
    hzs = 2 * _hurwitz_zeta_series(-q, a, -x, onlyeven=not imag, onlyodd=imag, skipterm=q)
    # hzs = ζ(s1,1+x) -/+ ζ(s1,1-x) but without the x^q term in the series
    pdiff = _power_diff(x, q, a)
    # pdiff accurately handles the sum of the external power x^-s1 due to
    # 25.11.E3 with the q-th term (cancellation) with even q for the real part
    # and odd q for the imaginary part
    cancelcond = jnp.logical_and(imag, q % 2 == 1)
    cancelcond |= jnp.logical_and(not imag, q % 2 == 0)
    power = jnp.where(cancelcond, pdiff, x ** -s1)
    hz = power + hzs # = ζ(s1,x) -/+ ζ(s1,1-x)
    
    # pole canceling
    gam = jnp.where(s % 1, gam, jnp.where(s % 2, 1, -1) / gamma(s)) # int s1
    pderiv = jnp.where(x, -jnp.log(x) * x ** -s1, 0)
    if imag:
        pha = jnp.where(s1 % 2 == 0, jnp.where(s1 % 4, -1, 1) * jnp.pi / 2, pha)
        hz = jnp.where(s1 % 2 == 1, pderiv + 2 * _dds_hzs_onlyevenodd_evenodds(-q, -x, False), hz)
    else:
        pha = jnp.where(s1 % 2 == 1, jnp.where(s % 4, 1, -1) * jnp.pi / 2, pha)
        hz = jnp.where(s1 % 2 == 0, pderiv + 2 * _dds_hzs_onlyevenodd_evenodds(-q, -x, True), hz)
        
    out = (pi * gam * pha) * hz
    if imag:
        out *= jnp.where(neg, -1, 1)
    return out

def _cos_pi2(n, x):
    """ compute cos(π/2 (n + x)) for n integer, accurate for small x """
    arg = -jnp.pi / 2 * x
    cos = jnp.where(n % 2, jnp.sin(arg), jnp.cos(arg))
    return cos * jnp.where(n // 2 % 2, -1, 1)

def _sin_pi2(n, x):
    return _cos_pi2(n - 1, x)

def _power_diff(x, q, a):
    """
    compute x^q-a + (-1)^q * [q-th term of 2 * hurwitz_zeta_series(-q, a, x)]
    """
    pint = x ** q
    pz = jnp.where(q, 0, jnp.where(a, -1, 0)) # * 0^q = 0^q-a - 0^q
    pdif = jnp.where(x, jnp.expm1(-a * jnp.log(x)), pz) # * x^q = x^q-a - x^q
    gamincr = jnp.where(q, _gamma_incr(1 + q, -a), 0)
    # gamincr = Γ(1+q-a) / Γ(1+q)Γ(1-a)  -  1
    zz = _zeta_zero(a) # = ζ(a) - ζ(0)
    qdif = 2 * (1 + gamincr) * zz - gamincr # = (q-th term) - (q-th term)|_a=0
    return pint * (pdif + qdif)

def _zeta_zero(s):
    """
    Compute zeta(s) - zeta(0) for |s| < 1 accurately
    """
    
    # f(s) = zeta(s) - 1 / (s - 1)
    # I have the Taylor series of f(s)
    # zeta(s) - zeta(0) = f(s) + 1 / (s - 1) + 1/2 =
    # = f(s) + 1/(s-1) + 1 - 1 + 1/2 =
    # = f(s) - 1/2 + s/(s-1)
    
    t = _float_type(s)
    coef = jnp.array(_zeta_zero_coef, t).at[0].set(0)
    fact = jnp.cumprod(jnp.arange(coef.size).at[0].set(1), dtype=t)
    coef /= fact
    f = jnp.polyval(coef[::-1], s)
    return f + s / (s - 1)

_zeta_zero_coef = [         # = _gen_zeta_zero_coef(17)
    0.5,
    0.08106146679532726,
    -0.006356455908584851,
    -0.004711166862254448,
    0.002896811986292041,
    -0.00023290755845472455,
    -0.0009368251300509295,
    0.0008498237650016692,
    -0.00023243173551155957,
    -0.00033058966361229646,
    0.0005432341157797085,
    -0.00037549317290726367,
    -1.960353628101392e-05,
    0.00040724123256303315,
    -0.0005704920132817777,
    0.0003939270789812044,
    8.345880582550168e-05,
]

def _gen_zeta_zero_coef(n):
    """
    Compute first n derivatives of zeta(s) - 1/(s-1) at s = 0
    """
    import mpmath as mp
    with mp.workdps(32):
        func = lambda s: mp.zeta(s) - 1 / (s - 1)
        return [float(mp.diff(func, 0, k)) for k in range(n)]

def _gamma_incr(x, e):
    """
    Compute Γ(x+e) / (Γ(x)Γ(1+e)) - 1 accurately for x >= 2 and |e| < 1/2
    """
    
    # G(x + e) / G(x)G(1+e) - 1 =
    # = expm1(log(G(x + e) / G(x)G(1+e))) =
    # = expm1(log G(x + e) - log G(x) - log G(1 + e))
    
    t = _float_type(x, e)
    n = 23 if t == jnp.float64 else 10
    # n such that 1/2^n 1/n! d^n/dx^n log G(x) |_x=2 < eps
    k = jnp.arange(n).reshape((n,) + (1,) * max(x.ndim, e.ndim))
    coef = jspecial.polygamma(k, x)
    fact = jnp.cumprod(1 + k, 0, t)
    coef /= fact
    gammaln = e * jnp.polyval(coef[::-1], e)
    return jnp.expm1(gammaln - _gammaln1(e))
    
    # Like _gammaln1, I thought that writing as log Γ(1+x+e) - log Γ(1+x) +
    # - log1p(e/x) would increase accuracy, instead it deteriorates

def _gammaln1(x):
    """ compute log Γ(1+x) accurately for |x| <= 1/2 """
    
    t = _float_type(x)
    coef = jnp.array(_gammaln1_coef_1[:48], t) # 48 found by trial and error
    return x * jnp.polyval(coef[::-1], x)
    
    # I thought that writing this as log Γ(2+x) - log1p(x) would be more
    # accurate but it isn't, probably there's a cancellation for some values of
    # x

_gammaln1_coef_1 = [        # = _gen_gammaln1_coef(53, 1)
    -0.5772156649015329,
    0.8224670334241132,
    -0.40068563438653143,
    0.27058080842778454,
    -0.20738555102867398,
    0.1695571769974082,
    -0.1440498967688461,
    0.12550966952474304,
    -0.11133426586956469,
    0.1000994575127818,
    -0.09095401714582904,
    0.083353840546109,
    -0.0769325164113522,
    0.07143294629536133,
    -0.06666870588242046,
    0.06250095514121304,
    -0.058823978658684585,
    0.055555767627403614,
    -0.05263167937961666,
    0.05000004769810169,
    -0.047619070330142226,
    0.04545455629320467,
    -0.04347826605304026,
    0.04166666915034121,
    -0.04000000119214014,
    0.03846153903467518,
    -0.037037037312989324,
    0.035714285847333355,
    -0.034482758684919304,
    0.03333333336437758,
    -0.03225806453115042,
    0.03125000000727597,
    -0.030303030306558044,
    0.029411764707594344,
    -0.02857142857226011,
    0.027777777778181998,
    -0.027027027027223673,
    0.02631578947377995,
    -0.025641025641072283,
    0.025000000000022737,
    -0.024390243902450117,
    0.023809523809529224,
    -0.023255813953491015,
    0.02272727272727402,
    -0.022222222222222855,
    0.021739130434782917,
    -0.021276595744681003,
    0.02083333333333341,
    -0.02040816326530616,
    0.020000000000000018,
    -0.019607843137254912,
    0.019230769230769235,
    -0.01886792452830189,
]

def _gen_gammaln1_coef(n, x):
    """ compute Taylor coefficients of log Γ(x) """
    import mpmath as mp
    with mp.workdps(32):
        return [float(mp.polygamma(k, x) / mp.fac(k + 1)) for k in range(n)]

def zeta(s, n=0):
    """ compute ζ(n + s) with integer n, accurate for even n < 0 and small s """
    return jnp.where(n + s >= 1, _zeta_right(s, n), _zeta_left(s, n))

def _zeta_right(s, n):
    return jspecial.zeta(n + s, 1)

def _zeta_left(s, n):
    # reflection formula https://dlmf.nist.gov/25.4.E1
    m = 1 - n
    x = -s
    # m + x = 1 - (n + s) = 1 - n - s
    mx = m + x # > 0
    pi = 2 * (2 * jnp.pi) ** -mx
    cos = _cos_pi2(m, x) # = cos(π/2 (m + x)) but accurate for small x
    gam = gamma(mx)
    zet = _zeta_right(x, m)
    
    # cancel zeta pole at 1
    cos = jnp.where(mx == 1, -jnp.pi / 2, cos)
    zet = jnp.where(mx == 1,           1, zet)
        
    return pi * cos * gam * zet

def _periodic_bernoulli(n, x):
    # TODO to make this jittable, hardcode size to 60 and truncate by writing
    # zeros
    n = int(n)
    bernoulli = special.bernoulli(n)
    k = numpy.arange(n + 1)
    binom = special.binom(n, k)
    coeffs = binom[::-1] * bernoulli
    x = x % 1
    cond = x < 0.5
    x = jnp.where(cond, x, 1 - x)
    out = jnp.polyval(coeffs, x)
    if n % 2 == 1:
        out = out * jnp.where(cond, 1, -1)
    return out

@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def scaled_periodic_bernoulli(n, x):
    """ periodic Bernoulli polynomial scaled such that B_n(0) = ζ(n) """
    tau = 2 * jnp.pi
    lognorm = n * jnp.log(tau) - jspecial.gammaln(n + 1)
    norm = jnp.exp(lognorm) / 2
    cond = n < 60
    smalls = norm * _periodic_bernoulli(n if cond else 1, x)
    # n -> ∞: -Re e^2πix / i^n
    # don't use 1j ** n because it is very inaccurate
    arg = tau * x
    sign = jnp.where((n // 2) % 2, 1, -1)
    larges = sign * jnp.where(n % 2, jnp.sin(arg), jnp.cos(arg))
    return jnp.where(cond, smalls, larges)

@scaled_periodic_bernoulli.defjvp
def _scaled_periodic_bernoulli_jvp(n, primals, tangents):
    x, = primals
    xt, = tangents
    primal = scaled_periodic_bernoulli(n, x)
    tangent = 2 * jnp.pi * scaled_periodic_bernoulli(n - 1, x) * xt
    return primal, tangent

# TODO in general use jnp.piecewise, which only computes necessary cases, and
# lax.while_loop, to avoid unnecessary iterations. Note that piecewise requires
# traceable functions. Also, piecewise needs the input to be a single array,
# use lax.switch manually for multiple arguments.
