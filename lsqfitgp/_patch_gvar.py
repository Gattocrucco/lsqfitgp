# lsqfitgp/_patch_gvar.py
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

import builtins
import functools

import jax
import gvar
from jax import numpy as jnp
from jax.scipy import special as jspecial
import numpy as np
from scipy import linalg

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

def jaxsupport(jax_ufunc):
    def decorator(gvar_ufunc):
        @functools.wraps(gvar_ufunc)
        def newfunc(x):
            if isinstance(x, jnp.ndarray):
                return jax_ufunc(x)
            else:
                return gvar_ufunc(x)
        return newfunc
    return decorator

for fname in gvar_ufuncs:
    fgvar = getattr(gvar, fname)
    fjax = getattr(jspecial if fname == 'erf' else jnp, fname)
    fboth = jaxsupport(fjax)(fgvar)
    setattr(gvar, fname, fboth)

gvar.BufferDict.del_distribution('log')
gvar.BufferDict.del_distribution('sqrt')
gvar.BufferDict.del_distribution('erfinv')
gvar.BufferDict.add_distribution('log', gvar.exp)
gvar.BufferDict.add_distribution('sqrt', gvar.square)
gvar.BufferDict.add_distribution('erfinv', gvar.erf)

def scipy_eigh(x):
    w, v = linalg.eigh(x)
    w = np.abs(w)
    si = np.argsort(w)
    return w[si], v.T[si]

gvar.SVD._analyzers = dict(scipy_eigh=scipy_eigh)
# default uses SVD instead of diagonalization because it once was more stable
