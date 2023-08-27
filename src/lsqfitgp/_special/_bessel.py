# lsqfitgp/_special/_bessel.py
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

from .. import _jaxext
from . import _gamma

j0 = _jaxext.makejaxufunc(special.j0, lambda x: -j1(x))
j1 = _jaxext.makejaxufunc(special.j1, lambda x: (j0(x) - jv(2, x)) / 2.0)
jv = _jaxext.makejaxufunc(special.jv, None, lambda v, z: jvp(v, z, 1))
jvp = _jaxext.makejaxufunc(special.jvp, None, lambda v, z, n: jvp(v, z, n + 1), None, excluded=(2,))

kv = _jaxext.makejaxufunc(special.kv, None, lambda v, z: kvp(v, z, 1))
kvp = _jaxext.makejaxufunc(special.kvp, None, lambda v, z, n: kvp(v, z, n + 1), None, excluded=(2,))

iv = _jaxext.makejaxufunc(special.iv, None, lambda v, z: ivp(v, z, 1))
ivp = _jaxext.makejaxufunc(special.ivp, None, lambda v, z, n: ivp(v, z, n + 1), None, excluded=(2,))

# See jax #1870, #2466, #9956, #11002 and
# https://github.com/josipd/jax/blob/master/jax/experimental/jambax.py
# to implement special functions in jax with numba

@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
@jax.jit
def jvmodx2(nu, x2):
    x = jnp.sqrt(x2)
    normal = (x / 2) ** -nu * jv(nu, x)
    return jnp.where(x2, normal, 1 / _gamma.gamma(nu + 1))

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
@functools.partial(jax.jit, static_argnums=(2,))
def kvmodx2(nu, x2, norm_offset=0):
    x = jnp.sqrt(x2)
    normal = 2 / _gamma.gamma(nu + norm_offset) * (x / 2) ** nu * kv(nu, x)
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
@functools.partial(jax.jit, static_argnums=(1,))
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
