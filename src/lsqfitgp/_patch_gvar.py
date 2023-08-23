# lsqfitgp/_patch_gvar.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

""" modifications to the global state of gvar """

import functools

import gvar
from jax import numpy as jnp
from jax.scipy import special as jspecial

gvar_ufuncs = [
    'sin',
    'cos',
    'tan',
    'exp',
    'log',
    'sqrt',
    'fabs',
    'sinh',
    'cosh',
    'tanh',
    'arcsin',
    'arccos',
    'arctan',
    'arctan2',
    'arcsinh',
    'arccosh',
    'arctanh',
    'square',
    'erf',
]

for fname in gvar_ufuncs:
    fgvar = getattr(gvar, fname)
    fjax = getattr(jnp, fname, getattr(jspecial, fname, NotImplemented))
    fboth = functools.singledispatch(fgvar)
    fboth.register(jnp.ndarray, fjax)
    setattr(gvar, fname, fboth)

# reset transformations to support jax arrays
gvar.BufferDict.del_distribution('log')
gvar.BufferDict.del_distribution('sqrt')
gvar.BufferDict.del_distribution('erfinv')
gvar.BufferDict.add_distribution('log', gvar.exp)
gvar.BufferDict.add_distribution('sqrt', gvar.square)
gvar.BufferDict.add_distribution('erfinv', gvar.erf)
