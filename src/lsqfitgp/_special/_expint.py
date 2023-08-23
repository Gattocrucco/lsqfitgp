# lsqfitgp/_special/_expint.py
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

from scipy import special
import jax
from jax import numpy as jnp
from jax.scipy import special as jspecial

from . import _gamma
from . import _taylor
from .. import _jaxext

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
    
    x = jnp.asarray(x)
    with jax.ensure_compile_time_eval():
        n = jnp.asarray(n)
        dt = _jaxext.float_type(n, x)
        if dt == jnp.float32:
            nt = jnp.array(10, 'i4') # TODO optimize to raise maximum n
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
    
    n, x = jnp.asarray(n), jnp.asarray(x)
    k = jnp.arange(n)
    fact = jnp.cumprod(k.at[0].set(1), dtype=_jaxext.float_type(n, x))
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

def expn_asymp_coefgen(s, e, n):
    k = jnp.arange(s, e, dtype=n.dtype)
    return (-1) ** k * _gamma.poch(n, k)

def expn_asymp(n, z, nt):
    """
    Compute E_n(z) for large |z|, |arg z| < 3/2 π. ``nt`` is the number of terms
    used in the asymptotic series.
    """
    
    # DLMF 8.20.2
    
    invz = 1 / z
    return jnp.exp(-z) * invz * _taylor.taylor(expn_asymp_coefgen, (n,), 0, nt, invz)

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
    dtype = _jaxext.float_type(x)
    num = jnp.polyval(jnp.array(_si_num[::-1], dtype), x2)
    denom = jnp.polyval(jnp.array(_si_denom[::-1], dtype), x2)
    return x * num / denom

def _minus_cin_smallx(x):
    """ Compute -Cin(x) = int_0^x dt (cos t - 1) / t, for x < 4 """
    x2 = jnp.square(x)
    dtype = _jaxext.float_type(x)
    num = jnp.polyval(jnp.array(_ci_num[::-1], dtype), x2)
    denom = jnp.polyval(jnp.array(_ci_denom[::-1], dtype), x2)
    return x2 * num / denom

def _ci_smallx(x):
    """ Compute Ci(x) = -int_x^oo dt cos t / t, for x < 4 """
    gamma = 0.57721566490153286060
    return gamma + jnp.log(x) + _minus_cin_smallx(x)

def _f_largex(x):
    """ Compute f(x) = int_0^oo dt sin t / (x + t), for x > 4 """
    x2 = 1 / jnp.square(x)
    dtype = _jaxext.float_type(x)
    num = jnp.polyval(jnp.array(_f_num[::-1], dtype), x2)
    denom = jnp.polyval(jnp.array(_f_denom[::-1], dtype), x2)
    return num / denom / x

def _g_largex(x):
    """ Compute g(x) = int_0^oo dt cos t / (x + t), for x > 4 """
    x2 = 1 / jnp.square(x)
    dtype = _jaxext.float_type(x)
    num = jnp.polyval(jnp.array(_g_num[::-1], dtype), x2)
    denom = jnp.polyval(jnp.array(_g_denom[::-1], dtype), x2)
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
