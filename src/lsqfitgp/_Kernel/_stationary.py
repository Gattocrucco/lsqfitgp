# lsqfitgp/_Kernel/_stationary.py
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

from jax import numpy as jnp

from .. import _jaxext

from . import _util
from . import _crosskernel
from . import _kernel

class CrossStationaryKernel(_crosskernel.CrossKernel):
    """
    
    Subclass of `CrossKernel` for stationary kernels.

    A stationary kernel depends only on the difference, dimension by dimension,
    between its two arguments.

    Parameters
    ----------
    core : callable
        A function taking one positional argument ``delta = x - y`` and optional
        keyword arguments.
    input : {'signed', 'posabs', 'abs'}, default 'signed'
        If ``'signed'``, `kernel` is passed the bare difference. If
        ``'posabs'``, `kernel` is passed the absolute value of the difference,
        and the difference of equal points is a small number instead of zero. If
        ``'abs'``, the absolute value.
    **kw
        Additional keyword arguments are passed to the `CrossKernel`
        constructor.
            
    """

    def __new__(cls, core, *, input='signed', **kw):
        
        if input == 'posabs':
            dist = lambda x, y: _softabs(x - y)
        elif input == 'signed':
            dist = lambda x, y: x - y
        elif input == 'abs':
            dist = lambda x, y: jnp.abs(x - y)
        else:
            raise KeyError(input)
        
        def newcore(x, y, **kw):
            q = _util.ufunc_recurse_dtype(dist, x, y)
            return core(q, **kw)
        
        return super().__new__(cls, newcore, **kw)

    # TODO this class requires that, on both inputs, the thing depends only on
    # the distance. However, when transforming a stationary kernel, I often have
    # the property only on either side. Making a left/right class interferes
    # with _swap, because it is not clean to switch classes in the ancestors,
    # so there would be a class for a single side with a property.
    #
    # Maybe the elegant way to keep left/right properties separated is having
    # separate classes for the two processes.
    #
    # Drop the whole generic hierarchy. Keep a single class Kernel. Have two
    # attributes left and right which are instances of Process. Linops are
    # applied as methods on left and right, which operate on a partial
    # evaluation of the core. Kernel.__getattr__ sees if the missing attribute
    # is a method defined on both left and right and calls both of them; this
    # allows potential overrides for the joint transformation.
    #
    # The class logic operates on left/right. Linops looks one process at a
    # time. Algops are defined on the whole kernel but need somehow to indicate
    # how to change the classes of processes. Maybe a list of class-preserving
    # algops in the process.

class StationaryKernel(CrossStationaryKernel, _kernel.Kernel):
    pass

# make these transformations preserve the class StationaryKernel and upwards,
# other transformations are added by IsotropicKernel
StationaryKernel.inherit_transf('dim', intermediates=True)

def _eps(x):
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.finfo(x.dtype).eps
        # finfo(x) does not work in numpy 1.20
    else:
        return jnp.finfo(jnp.empty(())).eps

def _softabs(x):
    return jnp.abs(x) + _eps(x)
