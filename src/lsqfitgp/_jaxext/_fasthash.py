# lsqfitgp/_jaxext/_fasthash.py
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

# JAX port of fasthash https://github.com/ztanml/fast-hash, original license:

# The MIT License
#
# Copyright (C) 2012 Zilong Tan (eric.zltan@gmail.com)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools

from jax import numpy as jnp
from jax import lax
import jax

# TODO this breaks down if jax_enable_x64=False. I have to write the operations
# on 64 bit integers in terms of operations on 32 bit integers, and always
# manipulate arrays with 2x uint32. I bet I can find such translations on the
# internet. I can make an interface for something that switches automatically
# between the two modes.

# TODO move to _special

# Compression function for Merkle-Damgard construction.
# This function is generated using the framework provided.
def mix(h):
    h ^= h >> 23
    h *= jnp.array(0x2127599bf4325c37, jnp.uint64)
    h ^= h >> 47
    return h

@functools.partial(jax.jit, static_argnames=('unroll',))
def fasthash64(buf, seed, *, unroll=4):
    # buf = jnp.asarray(buf) # needed without jit
    seed = jnp.array(seed, jnp.uint64)
    assert seed.dtype == jnp.uint64 # check that jax_enable_x64=True
    assert buf.ndim == 1
    buf = buf.view(jnp.uint8)
    m = jnp.array(0x880355f21e6d1965, jnp.uint64)
    pos = buf[:buf.size - buf.size % 8].view(jnp.uint64)
    h = seed ^ (buf.size * m)

    def loop(carry, v):
        h, m = carry
        h ^= mix(v)
        h *= m
        return (h, m), None
    (h, _), _ = lax.scan(loop, (h, m), pos, unroll=unroll)

    pos2 = buf[pos.nbytes:]
    assert pos2.size < 8
    assert pos.nbytes + pos2.size == buf.size
    v = jnp.array(0, jnp.uint64)

    if pos2.size >= 7: v ^= pos2[6].astype(jnp.uint64) << 48
    if pos2.size >= 6: v ^= pos2[5].astype(jnp.uint64) << 40
    if pos2.size >= 5: v ^= pos2[4].astype(jnp.uint64) << 32
    if pos2.size >= 4: v ^= pos2[3].astype(jnp.uint64) << 24
    if pos2.size >= 3: v ^= pos2[2].astype(jnp.uint64) << 16
    if pos2.size >= 2: v ^= pos2[1].astype(jnp.uint64) << 8
    if pos2.size >= 1: v ^= pos2[0].astype(jnp.uint64)
    if pos2.size:
        h ^= mix(v)
        h *= m

    assert h.dtype == jnp.uint64
    return mix(h)

def fasthash32(buf, seed):
    seed = jnp.array(seed, jnp.uint32)
    h = fasthash64(buf, seed)
    return (h - (h >> 32)).astype(jnp.uint32)
