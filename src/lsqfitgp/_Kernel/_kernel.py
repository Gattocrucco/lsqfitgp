# lsqfitgp/_Kernel/_kernel.py
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

class Kernel(_crosskernel.CrossKernel):
    
    @property
    def derivable(self):
        assert self._minderivable[0] == self._minderivable[1]
        assert self._maxderivable[0] == self._maxderivable[1]
        if self._minderivable == self._maxderivable:
            return self._minderivable[0]
        else:
            return None
    
    @property
    def maxdim(self):
        return self._maxdim
        
    def _binary(self, value, op):
        with _jaxext.skipifabstract():
            assert not _util.is_numerical_scalar(value) or 0 <= value < jnp.inf, value
        return super()._binary(value, op)

_crosskernel.Kernel = Kernel
