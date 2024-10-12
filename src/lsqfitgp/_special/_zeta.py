# lsqfitgp/_special/_zeta.py
#
# Copyright (c) 2022, 2023, 2024, Giacomo Petrillo
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

import collections
import functools
import math

import jax
from jax import lax
from jax import numpy as jnp
from jax.scipy import special as jspecial

from .. import _jaxext
from . import _gamma

def hurwitz_zeta_series(m, x, a1, onlyeven=False, onlyodd=False, skipterm=None):
    """
    hurwitz zeta(s = m + x, a = 1 - a1) with integer m
    meant to be used with |x| ≤ 1/2, but no actual restriction
    assuming -S <= s <= 0 and |a1| <= 1/2 with S ~ some decade
    https://dlmf.nist.gov/25.11.E10
    """
    
    # decide number of terms to sum
    t = _jaxext.float_type(m, x, a1)
    nmax = hze_nmax(t)
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

def hze_nmax(t):
    minz = 0.0037 # = min(2 gamma(s) / (2 pi)^s) for s <= 0
    return int(math.ceil(-math.log2(jnp.finfo(t).eps * minz)))

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
    zeta = hurwitz_zeta_series(zero, s, a1)  # https://dlmf.nist.gov/25.11.E10
    zeta += jnp.where(cond, a ** -s, 0)  # https://dlmf.nist.gov/25.11.E3
    return zeta
    
    # https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.zeta

@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
# @functools.partial(jax.jit, static_argnums=(2,))
def periodic_zeta(x, s, imag=False):
    """
    compute F(x,s) = Li_s(e^2πix) for real s > 1, real x
    """
    
    x = jnp.asarray(x)
    s = jnp.asarray(s)
    
    # decide boundary for large/small s implementation
    t = _jaxext.float_type(x, s)
    eps = jnp.finfo(t).eps
    nmax = 50
    larges = math.ceil(-math.log(eps) / math.log(nmax)) # 1/nmax^s < eps
    
    z_smalls = periodic_zeta_smalls(x, s, imag)
    z_larges = periodic_zeta_larges(x, s, nmax, imag)

    return jnp.where(s < larges, z_smalls, z_larges)
    
@periodic_zeta.defjvp
def periodic_zeta_jvp(s, imag, p, t):
    x, = p
    xt, = t
    primal = periodic_zeta(x, s, imag)
    sgn = 1 if imag else -1
    tangent = 2 * jnp.pi * sgn * periodic_zeta(x, s - 1, not imag) * xt
    return primal, tangent

def standard_x(x):
    """ bring x in [0, 1/2] by modulus and reflection """
    x %= 1
    neg = x > 1/2
    return neg, jnp.where(neg, 1 - x, x)

def periodic_zeta_larges(x, s, nmax, imag):
    """ https://dlmf.nist.gov/25.13.E1 """

    t = _jaxext.float_type(x, s)
    s = s.astype(t) # avoid n^s overflow with integer s
    n = jnp.arange(1, nmax + 1)
    neg, nx = standard_x(n * x[..., None])
    func = jnp.sin if imag else jnp.cos
    terms = func(2 * jnp.pi * nx) / n ** s[..., None]
    if imag:
        terms *= jnp.where(neg, -1, 1)
    return jnp.sum(terms, -1)

def periodic_zeta_smalls(x, s, imag):
    """
    https://dlmf.nist.gov/25.11.E10 and https://dlmf.nist.gov/25.11.E3 expanded
    into https://dlmf.nist.gov/25.13.E2
    """
    neg, x = standard_x(x) # x in [0, 1/2]
    
    eps = jnp.finfo(_jaxext.float_type(x, s)).eps
    s = jnp.where(s % 1, s, s * (1 + eps)) # avoid integer s

    s1 = 1 - s  # < 0
    q = -jnp.around(s1).astype(int)
    a = s1 + q
    # now s1 == -q + a with q integer >= 0 and |a| <= 1/2
    
    pi = (2 * jnp.pi) ** -s1
    gam = _gamma.gamma(s1)
    func = sin_pi2 if imag else cos_pi2
    pha = func(-q, a) # = sin or cos(π/2 s1), numerically accurate for small a
    hzs = 2 * hurwitz_zeta_series(-q, a, -x, onlyeven=not imag, onlyodd=imag, skipterm=q)
    # hzs = ζ(s1,1+x) -/+ ζ(s1,1-x) but without the x^q term in the series
    pdiff = zeta_series_power_diff(x, q, a)
    # pdiff accurately handles the sum of the external power x^-s1 due to
    # 25.11.E3 with the q-th term (cancellation) with even q for the real part
    # and odd q for the imaginary part
    cancelcond = jnp.logical_and(imag, q % 2 == 1)
    cancelcond |= jnp.logical_and(not imag, q % 2 == 0)
    power = jnp.where(cancelcond, pdiff, x ** -s1)
    hz = power + hzs # = ζ(s1,x) -/+ ζ(s1,1-x)
    
    out = (pi * gam * pha) * hz
    if imag:
        out *= jnp.where(neg, -1, 1)
    return out

def cos_pi2(n, x):
    """ compute cos(π/2 (n + x)) for n integer, accurate for small x """
    arg = -jnp.pi / 2 * x
    cos = jnp.where(n % 2, jnp.sin(arg), jnp.cos(arg))
    return cos * jnp.where(n // 2 % 2, -1, 1)

def sin_pi2(n, x):
    return cos_pi2(n - 1, x)

def zeta_series_power_diff(x, q, a):
    """
    compute x^q-a + (-1)^q * [q-th term of 2 * hurwitz_zeta_series(-q, a, x)]
    """
    pint = x ** q
    pz = jnp.where(q, 0, jnp.where(a, -1, 0)) # * 0^q = 0^q-a - 0^q
    pdif = jnp.where(x, jnp.expm1(-a * jnp.log(x)), pz) # * x^q = x^q-a - x^q
    gamincr = jnp.where(q, _gamma.gamma_incr(1 + q, -a), 0)
    # gamincr = Γ(1+q-a) / Γ(1+q)Γ(1-a)  -  1
    zz = zeta_zero(a) # = ζ(a) - ζ(0)
    qdif = 2 * (1 + gamincr) * zz - gamincr # = (q-th term) - (q-th term)|_a=0
    return pint * (pdif + qdif)

def zeta_zero(s):
    """
    Compute zeta(s) - zeta(0) for |s| < 1 accurately
    """
    
    # f(s) = zeta(s) - 1 / (s - 1)
    # I have the Taylor series of f(s)
    # zeta(s) - zeta(0) = f(s) + 1 / (s - 1) + 1/2 =
    # = f(s) + 1/(s-1) + 1 - 1 + 1/2 =
    # = f(s) - 1/2 + s/(s-1)
    
    t = _jaxext.float_type(s)
    coef = jnp.array(zeta_zero_coef, t).at[0].set(0)
    fact = jnp.cumprod(jnp.arange(coef.size).at[0].set(1), dtype=t)
    coef /= fact
    f = jnp.polyval(coef[::-1], s)
    return f + s / (s - 1)

zeta_zero_coef = [         # = gen_zeta_zero_coef(17)
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

def gen_zeta_zero_coef(n): # pragma: no cover
    """
    Compute first n derivatives of zeta(s) - 1/(s-1) at s = 0
    """
    import mpmath as mp
    with mp.workdps(32):
        func = lambda s: mp.zeta(s) - 1 / (s - 1)
        return [float(mp.diff(func, 0, k)) for k in range(n)]

# @jax.jit
def zeta(s, n=0):
    """ compute ζ(n + s) with integer n, accurate for even n < 0 and small s """
    s = jnp.asarray(s)
    return jnp.where(n + s >= 0,
        zeta_0_inf(n + s),
        zeta_neg(s, n),
    )

def zeta_neg(s, n):
    # reflection formula https://dlmf.nist.gov/25.4.E1
    m = 1 - n
    x = -s
    # m + x = 1 - (n + s) = 1 - n - s
    mx = m + x # > 1
    logpi = -mx * jnp.log(2 * jnp.pi)
    cos = cos_pi2(m, x) # = cos(π/2 (m + x)) but accurate for small x
    loggam = jspecial.gammaln(mx)
    zet = zeta_0_inf(mx)
    
    # cancel zeta pole at 1
    cos = jnp.where(mx == 1, -jnp.pi / 2, cos)
    zet = jnp.where(mx == 1,           1, zet)
        
    return 2 * jnp.exp(logpi + loggam) * cos * zet

# Below I have my custom implementation of zeta. jax.scipy.special.zeta does not
# work in (0, 1) (last checked v0.4.34), but I don't remember if I actually
# need that interval, maybe I always use s > 1.

##########################################################################
#    The following is adapted from gsl/specfunc/zeta.c (GPL license)     #
##########################################################################

ChebSeries = collections.namedtuple('ChebSeries', 'c a b')

def cheb_eval_e(cs, x):
    d = 0.0
    dd = 0.0
    y = (2.0 * x - cs.a - cs.b) / (cs.b - cs.a)
    y2 = 2.0 * y
    
    for c in cs.c[:0:-1]:
        d, dd = y2 * d - dd + c, d

    return y * d - dd + 0.5 * cs.c[0]

# chebyshev fit for (s(t)-1)Zeta[s(t)]
# s(t)= (t+1)/2
# -1 <= t <= 1
zeta_xlt1_cs = ChebSeries(jnp.array([
     1.48018677156931561235192914649,
     0.25012062539889426471999938167,
     0.00991137502135360774243761467,
    -0.00012084759656676410329833091,
    -4.7585866367662556504652535281e-06,
     2.2229946694466391855561441361e-07,
    -2.2237496498030257121309056582e-09,
    -1.0173226513229028319420799028e-10,
     4.3756643450424558284466248449e-12,
    -6.2229632593100551465504090814e-14,
    -6.6116201003272207115277520305e-16,
     4.9477279533373912324518463830e-17,
    -1.0429819093456189719660003522e-18,
     6.9925216166580021051464412040e-21,
]), -1, 1)

# chebyshev fit for (s(t)-1)Zeta[s(t)]
# s(t)= (19t+21)/2
# -1 <= t <= 1
zeta_xgt1_cs = ChebSeries(jnp.array([
    19.3918515726724119415911269006,
     9.1525329692510756181581271500,
     0.2427897658867379985365270155,
    -0.1339000688262027338316641329,
     0.0577827064065028595578410202,
    -0.0187625983754002298566409700,
     0.0039403014258320354840823803,
    -0.0000581508273158127963598882,
    -0.0003756148907214820704594549,
     0.0001892530548109214349092999,
    -0.0000549032199695513496115090,
     8.7086484008939038610413331863e-6,
     6.4609477924811889068410083425e-7,
    -9.6749773915059089205835337136e-7,
     3.6585400766767257736982342461e-7,
    -8.4592516427275164351876072573e-8,
     9.9956786144497936572288988883e-9,
     1.4260036420951118112457144842e-9,
    -1.1761968823382879195380320948e-9,
     3.7114575899785204664648987295e-10,
    -7.4756855194210961661210215325e-11,
     7.8536934209183700456512982968e-12,
     9.9827182259685539619810406271e-13,
    -7.5276687030192221587850302453e-13,
     2.1955026393964279988917878654e-13,
    -4.1934859852834647427576319246e-14,
     4.6341149635933550715779074274e-15,
     2.3742488509048340106830309402e-16,
    -2.7276516388124786119323824391e-16,
     7.8473570134636044722154797225e-17
]), -1, 1)

def zeta_0_1(s):
    return cheb_eval_e(zeta_xlt1_cs, 2.0 * s - 1.0) / (s - 1.0)

def zeta_1_20(s):
    return cheb_eval_e(zeta_xgt1_cs, (2.0 * s - 21.0) / 19.0) / (s - 1.0)

def zeta_20_inf(s):
    f2 = 1.0 - 2.0 ** -s
    f3 = 1.0 - 3.0 ** -s
    f5 = 1.0 - 5.0 ** -s
    f7 = 1.0 - 7.0 ** -s
    return 1.0 / (f2 * f3 * f5 * f7);

def zeta_0_inf(s):
    return jnp.where(s >= 20,
        zeta_20_inf(s),
        jnp.where(s >= 1,
            zeta_1_20(s),
            zeta_0_1(s),
        ),
    )

##########################################################################
