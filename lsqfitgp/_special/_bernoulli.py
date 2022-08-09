# lsqfitgp/_special/_bernoulli.py
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

import functools

import numpy
from scipy import special
import jax
from jax import numpy as jnp
from jax.scipy import special as jspecial

def periodic_bernoulli(n, x):
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
    smalls = norm * periodic_bernoulli(n if cond else 1, x)
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
