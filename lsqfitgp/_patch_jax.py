# lsqfitgp/_patch_jax.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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

from jax.config import config
config.update("jax_enable_x64", True)

import builtins

import numpy as np

from scipy import special as special_noderiv
from autograd.scipy import special

# TODO kv is currently not implemented in JAX, I have to define a primitive
# and try to use the scipy cython implementation of kvp as XLA translation
# implementation
# autograd.extend.defvjp(
#     special.kvp,
#     lambda ans, v, z, n: lambda g: g * special.kvp(v, z, n + 1),
#     argnums=[1]
# )
