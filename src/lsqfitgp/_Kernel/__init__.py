# lsqfitgp/_Kernel/__init__.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

from ._util import prod_recurse_dtype, sum_recurse_dtype, is_numerical_scalar
from ._crosskernel import CrossKernel, AffineSpan, PreservedBySwap
from . import _ops # keep first
from . import _alg # keep first
from ._kernel import Kernel
from ._stationary import CrossStationaryKernel, StationaryKernel
from ._isotropic import CrossIsotropicKernel, IsotropicKernel, Zero
from ._decorators import (crosskernel, kernel, crossstationarykernel,
    stationarykernel, crossisotropickernel, isotropickernel)
