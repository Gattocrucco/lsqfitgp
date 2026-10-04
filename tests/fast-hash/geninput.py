# lsqfitgp/tests/fast-hash/geninput.py
#
# Copyright (c) 2023, 2026, Giacomo Petrillo
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

"""Generate random test inputs for the fast-hash tests as C and Python code."""

import numpy as np

gen = np.random.default_rng(202303181456)


def genint(dtype, size=()):
    """Return random integers spanning the full range of `dtype`."""
    return gen.integers(
        np.iinfo(dtype).min, np.iinfo(dtype).max, endpoint=True, dtype=dtype, size=size
    )


ninputs = 20

inputs = [genint('u1', size) for size in range(ninputs)]
seed32 = genint('u4')
seed64 = genint('u8')

print('\nC CODE:\n')
for size in range(ninputs):
    print(f'    uint8_t input{size}[] = {{{", ".join(map(str, inputs[size]))}}};')
print(
    f'    uint8_t *inputs[] = {{{", ".join(f"input{size}" for size in range(ninputs))}}};'
)
print(f'    uint32_t seed32 = {seed32}U;')
print(f'    uint64_t seed64 = {seed64}ULL;')

print('\nPYTHON CODE:\n')
print('    inputs = [')
for size in range(ninputs):
    print(f'        jnp.array([{", ".join(map(str, inputs[size]))}], dtype=jnp.uint8),')
print('    ]')
print(f'    seed32 = {seed32}')
print(f'    seed64 = {seed64}')
