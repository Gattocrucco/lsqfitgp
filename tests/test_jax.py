# lsqfitgp/tests/test_jax.py
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

import sys

import numpy as np
import gvar
import jax
from jax import lax
from jax import numpy as jnp
from pytest import mark

sys.path.insert(0, '.')
from lsqfitgp import _patch_jax
import lsqfitgp as lgp
import util

rng = np.random.default_rng(202303031632)

def test_elementwise_grad_1():
    def f(x):
        return 2 * x
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(f)(x)
    y2 = jax.vmap(jax.grad(f))(x)
    util.assert_equal(y, y2)

def test_elementwise_grad_2():
    def f(x, z):
        return 2 * x * z
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(f, 0)(x, x)
    y2 = jax.vmap(jax.grad(f, 0))(x, x)
    util.assert_equal(y, y2)

def test_elementwise_grad_3():
    def f(x):
        return 2 * x
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(_patch_jax.elementwise_grad(f))(x)
    y2 = jax.vmap(jax.grad(jax.grad(f)))(x)
    util.assert_equal(y, y2)

def test_elementwise_grad_4():
    def f(x, z):
        return 2 * x * z
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(_patch_jax.elementwise_grad(f, 0), 0)(x, x)
    y2 = jax.vmap(jax.grad(jax.grad(f, 0), 0))(x, x)
    util.assert_equal(y, y2)

def test_elementwise_grad_5():
    def f(x, z):
        return 2 * x * z
    x = np.arange(8.)
    with jax.checking_leaks():
        y = _patch_jax.elementwise_grad(_patch_jax.elementwise_grad(f, 0), 1)(x, x)
    y2 = jax.vmap(jax.grad(jax.grad(f, 0), 1))(x, x)
    util.assert_equal(y, y2)

def test_elementwise_grad_6():
    def f(x, z):
        return 2 * x * z
    x = np.arange(8.)
    with jax.checking_leaks():
        y = jax.jacrev(_patch_jax.elementwise_grad(f, 0), 1)(x, x)
    y2 = jax.jacrev(jax.vmap(jax.grad(f, 0)), 1)(x, x)
    util.assert_equal(y, y2)

@mark.parametrize('maxnbytes', [1, 10 * 8, 1000 * 8])
def test_batcher(maxnbytes):
    def func(x, y):
        return x * y
    batched = _patch_jax.batchufunc(func, maxnbytes=maxnbytes)
    x, y = rng.standard_normal((2, 20, 7))
    args = (x[None, :, :], y[:, None, :])
    p1 = func(*args)
    p2 = batched(*args)
    util.assert_equal(p1, p2)

@mark.parametrize('maxnbytes', [1, 10 * 8, 1000 * 8])
def test_batcher_structured(maxnbytes):
    def func(x, y):
        return jnp.sum(x['x'] * y['x'], axis=-1)
    batched = _patch_jax.batchufunc(func, maxnbytes=maxnbytes)
    xy = rng.standard_normal((2, 20, 7)).view([('x', float, 7)]).squeeze(-1)
    xy = lgp.StructuredArray(xy)
    x, y = xy
    args = (x[None, :], y[:, None])
    p1 = func(*args)
    p2 = batched(*args)
    util.assert_allclose(p1, p2, atol=2e-15)

def test_hash():
    """ check the jax port of fast-hash against the original code """
    inputs = [
        jnp.array([], dtype=jnp.uint8),
        jnp.array([234], dtype=jnp.uint8),
        jnp.array([194, 116], dtype=jnp.uint8),
        jnp.array([160, 237, 166], dtype=jnp.uint8),
        jnp.array([72, 56, 46, 113], dtype=jnp.uint8),
        jnp.array([152, 22, 163, 7, 234], dtype=jnp.uint8),
        jnp.array([100, 11, 190, 249, 103, 74], dtype=jnp.uint8),
        jnp.array([119, 52, 46, 248, 188, 178, 216], dtype=jnp.uint8),
        jnp.array([81, 10, 197, 4, 19, 41, 69, 164], dtype=jnp.uint8),
        jnp.array([53, 246, 128, 162, 79, 228, 71, 137, 255], dtype=jnp.uint8),
        jnp.array([145, 141, 43, 100, 125, 107, 12, 4, 147, 229], dtype=jnp.uint8),
        jnp.array([117, 92, 35, 144, 76, 140, 59, 36, 42, 13, 94], dtype=jnp.uint8),
        jnp.array([91, 207, 0, 152, 226, 159, 190, 164, 136, 176, 194, 59], dtype=jnp.uint8),
        jnp.array([126, 94, 132, 168, 44, 150, 242, 165, 199, 149, 248, 82, 141], dtype=jnp.uint8),
        jnp.array([26, 101, 134, 203, 216, 141, 100, 242, 248, 225, 83, 131, 27, 100], dtype=jnp.uint8),
        jnp.array([153, 2, 211, 91, 131, 54, 101, 233, 213, 71, 216, 126, 60, 48, 157], dtype=jnp.uint8),
        jnp.array([114, 165, 8, 26, 213, 17, 112, 170, 104, 161, 164, 95, 53, 17, 149, 170], dtype=jnp.uint8),
        jnp.array([40, 198, 242, 87, 28, 55, 234, 142, 22, 200, 236, 65, 198, 91, 197, 233, 46], dtype=jnp.uint8),
        jnp.array([208, 21, 5, 101, 61, 240, 41, 134, 164, 25, 109, 253, 108, 140, 229, 255, 39, 199], dtype=jnp.uint8),
        jnp.array([240, 22, 57, 231, 226, 172, 97, 114, 34, 20, 14, 47, 118, 129, 193, 93, 43, 209, 75], dtype=jnp.uint8),
    ]
    hashes64 = jnp.array([
        7502587791032603753,
        16941272163545924368,
        10988138224395471776,
        9507901428091620561,
        8215232957141175337,
        18053746358964717198,
        12425373722766252877,
        2946925277746383721,
        12402381367179054957,
        755910146092029036,
        3255785893224143811,
        12592656301469221220,
        428602295608661196,
        5824169786726525377,
        1508071078291841094,
        11448092356368632731,
        6157277036160160880,
        9731805725958528408,
        3366289320067065534,
        17424790981778646777,
    ], dtype=jnp.uint64)
    hashes32 = jnp.array([
        1004310665,
        2046185678,
        3566082500,
        2790396102,
        2182032100,
        2323244336,
        1312080940,
        2492442272,
        1823551547,
        3569298354,
        1867203821,
        2449676296,
        2938746445,
        3041190206,
        3115372248,
        3527005061,
        1217622642,
        4177513530,
        303099792,
        2425579332,
    ], dtype=jnp.uint32)
    seed32 = 2428169863
    seed64 = 6361217807637034346
    for inp, h32, h64 in zip(inputs, hashes32, hashes64):
        hash64 = _patch_jax.fasthash64(inp, seed64)
        hash32 = _patch_jax.fasthash32(inp, seed32)
        assert h64 == hash64
        assert h32 == hash32

def genint(dtype, size=()):
    """ generate integers spanning full type range """
    return rng.integers(np.iinfo(dtype).min, np.iinfo(dtype).max, endpoint=True, dtype=dtype, size=size)

def test_hash_bitflip():
    """ check a single bit flip in the input changes 50% of the hash bits """
    buf = genint('u1', 43)
    
    bufmod = buf.copy()
    i = rng.integers(buf.size)
    j = rng.integers(8)
    bufmod[i] ^= 1 << j

    assert np.sum(lax.population_count(buf ^ bufmod)) == 1

    seed = genint('u8')
    h = _patch_jax.fasthash64(buf, seed)
    hmod = _patch_jax.fasthash64(bufmod, seed)
    diff = h ^ hmod
    flipped_bits = lax.population_count(diff)
    total_bits = h.nbytes * 8
    frac_flipped = flipped_bits / total_bits
    prob_flip = 0.5
    std = np.sqrt(prob_flip * (1 - prob_flip) / total_bits)
    assert abs(frac_flipped - prob_flip) <= 3 * std

def test_hash_numpy():
    """ check numpy arrays do not break the hash """
    buf = genint('u1', 2)
    _patch_jax.fasthash64(buf, 12345)
