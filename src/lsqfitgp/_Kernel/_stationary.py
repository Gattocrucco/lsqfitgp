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
from . import _kernel

class StationaryKernel(_kernel.Kernel):

    def __new__(cls, kernel, *, input='signed', scale=None, **kw):
        """
        
        Subclass of `Kernel` for isotropic kernels.
    
        Parameters
        ----------
        kernel : callable
            A function taking one argument ``delta`` which is the difference
            between x and y, plus optionally keyword arguments.
        input : {'signed', 'soft', 'hard'}
            If 'signed' (default), `kernel` is passed the bare difference. If
            'soft', `kernel` is passed the absolute value of the difference,
            and the difference of equal points is a small number instead of
            zero. If 'hard', the absolute value.
        scale : scalar
            The difference is divided by ``scale``.
        **kw
            Additional keyword arguments are passed to the `Kernel` init.
                
        """
        
        # TODO using 'signed', 'abs', 'softabs' as labels could be clearer.

        if input == 'soft':
            func = lambda x, y: _softabs(x - y)
        elif input == 'signed':
            func = lambda x, y: x - y
        elif input == 'hard':
            func = lambda x, y: jnp.abs(x - y)
        else:
            raise KeyError(input)
        
        transf = lambda q: q
        if scale is not None:
            with _jaxext.skipifabstract():
                assert 0 < scale < jnp.inf
            transf = lambda q : q / scale
        
        def function(x, y, **kwargs):
            q = _util.transf_recurse_dtype(func, x, y)
            return kernel(transf(q), **kwargs)
        
        return _kernel.Kernel.__new__(cls, function, **kw)

def _eps(x):
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.finfo(x.dtype).eps
        # finfo(x) does not work in numpy 1.20
    else:
        return jnp.finfo(jnp.empty(())).eps

def _softabs(x):
    return jnp.abs(x) + _eps(x)

