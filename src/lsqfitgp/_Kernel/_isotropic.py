# lsqfitgp/_Kernel/_isotropic.py
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

import sys

from jax import numpy as jnp

from .. import _jaxext

from . import _util
from . import _crosskernel
from . import _kernel
from . import _stationary

class CrossIsotropicKernel(_stationary.CrossStationaryKernel):
    """
    
    Subclass of `CrossStationaryKernel` for isotropic kernels.

    An isotropic kernel depends only on the Euclidean distance between its two
    arguments.

    Parameters
    ----------
    core : callable
        A function taking one argument ``r2`` which is the squared distance
        between x and y, plus optionally keyword arguments.
    input : {'squared', 'abs', 'posabs', 'raw'}, default 'squared'
        If ``'squared'``, `core` is passed the squared distance. If ``'abs'``,
        it is passed the distance (not squared). If ``'posabs'``, it is passed
        the distance, and the distance of equal points is a small number instead
        of zero. If ``'raw'``, `core` is passed both points separately like
        non-stationary kernels.
    **kw
        Additional keyword arguments are passed to the `CrossKernel`
        constructor.
    
    Notes
    -----
    The ``input='posabs'`` option will cause problems with second derivatives in
    more than one dimension.
            
    """
     
    def __new__(cls, core, *, input='squared', **kw):
        if input == 'raw':
            return _crosskernel.CrossKernel.__new__(cls, core, **kw)
        
        if input in ('squared', 'abs'):
            dist = lambda x, y: jnp.square(x - y)
        elif input == 'posabs':
            dist = lambda x, y: jnp.square(_stationary._softabs(x - y))
        else:
            raise KeyError(input)
                
        if input in ('posabs', 'abs'):
            transf = jnp.sqrt
        else:
            transf = lambda ss: ss
        
        def newcore(x, y, **kwargs):
            ss = _util.sum_recurse_dtype(dist, x, y)
            return core(transf(ss), **kwargs)
        
        return _crosskernel.CrossKernel.__new__(cls, newcore, **kw)

    # TODO add a `distance` parameter to pick arbitrary distances, but since the
    # distance definition can not be changed arbitrarily, it may be better to
    # keep this class for the 2-norm and eventually add another if needed.
    
    # TODO it is not efficient that the distance is computed separately for
    # each kernel in a kernel expression, but probably it would be difficult
    # to support everything without bugs while also computing the distance once.
    # A possible way is adding a keyword argument to the _kernel member
    # that kernels use to memoize things, the first IsotropicKernel that gets
    # called puts the distance there. Possible name: _cache.
    
class IsotropicKernel(CrossIsotropicKernel, _stationary.StationaryKernel):
    pass

IsotropicKernel.inherit_all_algops(intermediates=True)
IsotropicKernel.inherit_transf('rescale', intermediates=True)
IsotropicKernel.inherit_transf('loc', intermediates=True)
IsotropicKernel.inherit_transf('scale', intermediates=True)
IsotropicKernel.inherit_transf('maxdim', intermediates=True)
IsotropicKernel.inherit_transf('derivable', intermediates=True)
IsotropicKernel.inherit_transf('normalize', intermediates=True)
IsotropicKernel.inherit_transf('cond', intermediates=True)

class CrossConstant(CrossIsotropicKernel):
    pass

class Constant(CrossConstant, IsotropicKernel):
    pass

def zero(x, y):
    return jnp.broadcast_to(0., jnp.broadcast_shapes(x.shape, y.shape))

class Zero(IsotropicKernel):
    """
    Represents a kernel that unconditionally yields zero.
    """

    def __new__(cls):
        return super().__new__(cls, zero, input='raw')

_crosskernel.IsotropicKernel = IsotropicKernel
_crosskernel.CrossIsotropicKernel = CrossIsotropicKernel
_crosskernel.Constant = Constant
_crosskernel.CrossConstant = CrossConstant
