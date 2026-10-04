# lsqfitgp/_Kernel/__init__.py
#
# Copyright (c) 2023, 2026, Giacomo Petrillo
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

from lsqfitgp._Kernel._crosskernel import (  # noqa: F401
    AffineSpan,
    CrossKernel,
    PreservedBySwap,
)
from lsqfitgp._Kernel._util import (  # noqa: F401
    is_numerical_scalar,
    prod_recurse_dtype,
    sum_recurse_dtype,
)

# isort: off
from lsqfitgp._Kernel import _ops  # noqa: F401  # keep first
from lsqfitgp._Kernel import _alg  # noqa: F401  # keep first

# isort: on
from lsqfitgp._Kernel._decorators import (  # noqa: F401
    crossisotropickernel,
    crosskernel,
    crossstationarykernel,
    isotropickernel,
    kernel,
    stationarykernel,
)
from lsqfitgp._Kernel._isotropic import (  # noqa: F401
    CrossIsotropicKernel,
    IsotropicKernel,
    Zero,
)
from lsqfitgp._Kernel._kernel import Kernel  # noqa: F401
from lsqfitgp._Kernel._stationary import (  # noqa: F401
    CrossStationaryKernel,
    StationaryKernel,
)
