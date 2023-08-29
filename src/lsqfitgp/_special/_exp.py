# lsqfitgp/_special/_exp.py
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

import jax
from jax import numpy as jnp

@jax.custom_jvp
@jax.jit
def expm1x(x):
    r"""
    Compute accurately :math:`e^x - 1 - x = x^2/2 {}_1F_1(1, 3, x)`.
    """
    n = 10 if x.dtype == jnp.float32 else 17
    k = jnp.arange(2, n + 1)
    f = jnp.cumprod(k)
    coef = jnp.array(1, x.dtype) / f[::-1]
    smallx = x * x * jnp.polyval(coef, x, unroll=n)
    return jnp.where(jnp.abs(x) < 1, smallx, jnp.expm1(x) - x)
    
    # see also the GSL
    # https://www.gnu.org/software/gsl/doc/html/specfunc.html#relative-exponential-functions

@expm1x.defjvp
def _expm1x_jvp(p, t):
    x, = p
    xt, = t
    return expm1x(x), jnp.expm1(x) * xt
