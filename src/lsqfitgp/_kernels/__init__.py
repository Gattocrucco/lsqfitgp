# lsqfitgp/_kernels/__init__.py
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

from ._arma import *
from ._bart import *
from ._basic import *
from ._celerite import *
from ._matern import *
from ._randomwalk import *
from ._spectral import *
from ._wendland import *
from ._zeta import *

# TODO instead of adding forcekron by default to all 1D kernels, use maxdim=None
# by default in CrossKernel, add maxdim=1 to all 1D kernels, and let the user
# choose how to deal with nd (add option for sum-separation). Make an example
# about this in `multidimensional input`. Implement tests for separability
# on all kernels.

# TODO maybe I could have a continuity check like derivable, but to be useful
# it would be callable-only and take the derivation order. But I don't think
# any potential user needs it.

# TODO add explicit exponent parameter to infinitely divisible kernels, and
# check the exponent is an int in __pow__.
