# lsqfitgp/__init__.py
#
# Copyright (c) 2020, 2022, 2023, 2024, Giacomo Petrillo
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

"""
Module to fit Gaussian processes

See the manual at https://gattocrucco.github.io/lsqfitgp/docs
"""

__version__ = '0.21.1'

# these first because they modify global state
from . import _patch_jax
from . import _patch_gvar

from ._array import (
    StructuredArray,
    broadcast,
    broadcast_to,
    broadcast_arrays,
    asarray,
    unstructured_to_structured,
)
from ._Deriv import Deriv
from ._Kernel import (
    CrossKernel,
    Kernel,
    CrossStationaryKernel,
    StationaryKernel,
    CrossIsotropicKernel,
    IsotropicKernel,
    crosskernel,
    kernel,
    crossstationarykernel,
    stationarykernel,
    crossisotropickernel,
    isotropickernel,
)
from ._kernels import * # safe, _kernels/__init__.py only imports kernels
from ._gvarext import (
    jacobian,
    from_jacobian,
    gvar_gufunc,
    switchgvar,
    uformat,
    fmtspec_kwargs,
    gvar_format,
)
from ._GP import GP
from ._fit import empbayes_fit
from ._fastraniter import raniter, sample

from . import copula
from . import bayestree
