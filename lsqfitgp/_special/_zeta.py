# lsqfitgp/_special/_zeta.py
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

import functools
import math

import jax
from jax import lax
from jax import numpy as jnp
from jax.scipy import special as jspecial

from .. import _patch_jax
from . import _gamma

def _hurwitz_zeta_series(m, x, a1, onlyeven=False, onlyodd=False, skipterm=None):
    """
    hurwitz zeta(s = m + x, a = 1 - a1) with integer m
    meant to be used with |x| ≤ 1/2, but no actual restriction
    assuming -S <= s <= 0 and |a1| <= 1/2 with S ~ some decade
    https://dlmf.nist.gov/25.11.E10
    """
    
    # decide number of terms to sum
    t = _patch_jax.float_type(m, x, a1)
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
    t = _patch_jax.float_type(x, s)
    eps = jnp.finfo(t).eps
    nmax = 50
    larges = math.ceil(-math.log(eps) / math.log(nmax)) # 1/nmax^s < eps
    
    z_smalls = _periodic_zeta_smalls(x, s, imag)
    z_larges = _periodic_zeta_larges(x, s, nmax, imag)

    return jnp.where(s < larges, z_smalls, z_larges)
    
    # TODO rewrite without vectorization and then use vectorize, use switch
    # instead of where to avoid evaluating all branches all times

@periodic_zeta.defjvp
def _periodic_zeta_jvp(s, imag, p, t):
    x, = p
    xt, = t
    primal = periodic_zeta(x, s, imag)
    sgn = 1 if imag else -1
    tangent = 2 * jnp.pi * sgn * periodic_zeta(x, s - 1, not imag) * xt
    return primal, tangent

def _standard_x(x):
    """ bring x in [0, 1/2] by modulus and reflection """
    x %= 1
    neg = x > 1/2
    return neg, jnp.where(neg, 1 - x, x)

def _periodic_zeta_larges(x, s, nmax, imag):
    """ https://dlmf.nist.gov/25.13.E1 """

    t = _patch_jax.float_type(x, s)
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
    
    eps = jnp.finfo(_patch_jax.float_type(x, s)).eps
    s = jnp.where(s % 1, s, s * (1 + eps)) # avoid integer s

    s1 = 1 - s  # < 0
    q = -jnp.around(s1).astype(int)
    a = s1 + q
    # now s1 == -q + a with q integer >= 0 and |a| <= 1/2
    
    pi = (2 * jnp.pi) ** -s1
    gam = _gamma.gamma(s1)
    func = _sin_pi2 if imag else _cos_pi2
    pha = func(-q, a) # = sin or cos(π/2 s1), numerically accurate for small a
    hzs = 2 * _hurwitz_zeta_series(-q, a, -x, onlyeven=not imag, onlyodd=imag, skipterm=q)
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

def _cos_pi2(n, x):
    """ compute cos(π/2 (n + x)) for n integer, accurate for small x """
    arg = -jnp.pi / 2 * x
    cos = jnp.where(n % 2, jnp.sin(arg), jnp.cos(arg))
    return cos * jnp.where(n // 2 % 2, -1, 1)

def _sin_pi2(n, x):
    return _cos_pi2(n - 1, x)

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
    
    t = _patch_jax.float_type(s)
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

def _gen_zeta_zero_coef(n): # pragma: no cover
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
    return jnp.where(n + s >= 1, _zeta_right(s, n), _zeta_left(s, n))

def _zeta_right(s, n):
    return jspecial.zeta(n + s, 1)
    
    # jax's zeta implements a generic hurwitz algorithm, see the GSL
    # https://www.gnu.org/software/gsl/doc/html/specfunc.html#zeta-functions
    # to implement only the zeta in a possibly better way

def _zeta_left(s, n):
    # reflection formula https://dlmf.nist.gov/25.4.E1
    m = 1 - n
    x = -s
    # m + x = 1 - (n + s) = 1 - n - s
    mx = m + x # > 0
    logpi = -mx * jnp.log(2 * jnp.pi)
    cos = _cos_pi2(m, x) # = cos(π/2 (m + x)) but accurate for small x
    loggam = jspecial.gammaln(mx)
    zet = _zeta_right(x, m)
    
    # cancel zeta pole at 1
    cos = jnp.where(mx == 1, -jnp.pi / 2, cos)
    zet = jnp.where(mx == 1,           1, zet)
        
    return 2 * jnp.exp(logpi + loggam) * cos * zet

# TODO in general use jnp.piecewise, which only computes necessary cases, and
# lax.while_loop, to avoid unnecessary iterations. Note that piecewise requires
# traceable functions. Also, piecewise needs the input to be a single array,
# use lax.switch manually for multiple arguments.
