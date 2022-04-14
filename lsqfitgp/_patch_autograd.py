# lsqfitgp/_patch_autograd.py
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

"""Add vjp/jvp of asarray, fix #552 (creating object arrays from lists),
add missing scipy functions, add numpy.testing"""

import builtins

import autograd
import numpy as np

autograd.extend.defvjp(
    autograd.numpy.asarray,
    lambda ans, *args, **kw: lambda g: g
)
autograd.extend.defjvp(
    autograd.numpy.asarray,
    lambda g, ans, *args, **kw: g
)

autograd.numpy.numpy_boxes.ArrayBox.item = lambda self: self[(0,) * len(self.shape)]

def array(A, *args, **kwargs):
    t = builtins.type(A)
    if t in (list, tuple):
        return autograd.numpy.numpy_wrapper.array_from_args(args, kwargs, *map(lambda a: a if a.shape else a.item(), map(array, A)))
    else:
        return autograd.numpy.numpy_wrapper._array_from_scalar_or_array(args, kwargs, A)
autograd.numpy.numpy_wrapper.array = array
autograd.numpy.array = array

autograd.numpy.testing = np.testing

try:
    from scipy import special as special_noderiv
    from autograd.scipy import special
    
    missing_functions = [
        'bernoulli',
        'binom',
        'kv',
        'kvp',
        'factorial',
        'zeta'
    ]
    for f in missing_functions:
        if not hasattr(special, f):
            wf = autograd.extend.primitive(getattr(special_noderiv, f))
            setattr(special, f, wf)

    autograd.extend.defvjp(
        special.kvp,
        lambda ans, v, z, n: lambda g: g * special.kvp(v, z, n + 1),
        argnums=[1]
    )    

except (ImportError, ModuleNotFoundError):
    pass
    