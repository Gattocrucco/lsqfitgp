# lsqfitgp/copula/_beta.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

"""JAX-compatible implementation of the beta distribution"""

import functools

from scipy import special
import jax
from jax.scipy import special as jspecial
from jax import numpy as jnp

from .. import _patch_jax

@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def betaincinv(a, b, y):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    y = jnp.asarray(y)
    class ResultDummy:
        dtype = jnp.result_type(a.dtype, b.dtype, y.dtype)
        shape = jnp.broadcast_shapes(a.shape, b.shape, y.shape)
    return jax.pure_callback(special.betaincinv, ResultDummy, a, b, y, vectorized=True)

dIdx_ = _patch_jax.elementwise_grad(jspecial.betainc, 2)

@betaincinv.defjvp
def betaincinv_jvp(a, b, primals, tangents):
    y, = primals
    yt, = tangents
    x = betaincinv(a, b, y)
    dIdx = dIdx_(a, b, x)
    return x, yt / dIdx

class beta:
    
    @staticmethod
    def ppf(q, a, b):
        return betaincinv(a, b, q)
