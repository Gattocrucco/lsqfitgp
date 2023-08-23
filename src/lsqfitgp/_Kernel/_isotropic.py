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

class IsotropicKernel(_stationary.StationaryKernel):
    
    # I thought about adding a `distance` parameter to pick arbitrary distances,
    # but since the distance definition can not be changed arbitrarily, it is
    # better to keep this class for the 2-norm and eventually add another if
    # needed.
    
    # TODO it is not efficient that the distance is computed separately for
    # each kernel in a kernel expression, but probably it would be difficult
    # to support everything without bugs while also computing the distance once.
    # A possible way is adding a keyword argument to the _kernel member
    # that kernels use to memoize things, the first IsotropicKernel that gets
    # called puts the distance there. Possible name: _cache.
    
    def __new__(cls, kernel, *, input='squared', scale=None, **kw):
        """
        
        Subclass of :class:`Kernel` for isotropic kernels.
    
        Parameters
        ----------
        kernel : callable
            A function taking one argument ``r2`` which is the squared distance
            between x and y, plus optionally keyword arguments.
        input : {'squared', 'hard', 'soft', 'raw'}
            If 'squared' (default), ``kernel`` is passed the squared distance.
            If 'hard', it is passed the distance (not squared). If 'soft', it
            is passed the distance, and the distance of equal points is a small
            number instead of zero. If 'raw', the kernel is passed both points
            separately like non-stationary kernels.
        scale : scalar
            The distance is divided by ``scale``.
        **kw
            Additional keyword arguments are passed to the `Kernel` init.
        
        Notes
        -----
        The 'soft' option will cause problems with second derivatives in more
        than one dimension.
                
        """
        if input == 'raw':
            return _kernel.Kernel.__new__(cls, kernel, scale=scale, **kw)
            
        if input in ('squared', 'hard'):
            func = lambda x, y: (x - y) ** 2
        elif input == 'soft':
            func = lambda x, y: _stationary._softabs(x - y) ** 2
        else:
            raise KeyError(input)
        
        transf = lambda q: q
        
        if scale is not None:
            with _jaxext.skipifabstract():
                assert 0 < scale < jnp.inf
            transf = lambda q: q / scale ** 2
        
        if input in ('soft', 'hard'):
            transf = (lambda t: lambda q: jnp.sqrt(t(q)))(transf)
            # I do square and then square root because I first have to
            # compute the sum of squares
        
        def function(x, y, **kwargs):
            q = _util.sum_recurse_dtype(func, x, y)
            return kernel(transf(q), **kwargs)
        
        return _kernel.Kernel.__new__(cls, function, **kw)

_crosskernel.IsotropicKernel = IsotropicKernel

class Zero(IsotropicKernel):

    def __new__(cls):
        self = object.__new__(cls)
        self._kernel = lambda x, y: jnp.broadcast_to(0., jnp.broadcast_shapes(x.shape, y.shape))
        self._minderivable = (sys.maxsize, sys.maxsize)
        self._maxderivable = (sys.maxsize, sys.maxsize)
        self.initargs = None
        self._maxdim = sys.maxsize
        return self
    
    _swap = lambda self: self
    diff = lambda self, xderiv, yderiv: self
    batch = lambda self, maxnbytes: self
