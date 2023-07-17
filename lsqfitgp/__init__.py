# lsqfitgp/__init__.py
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

"""
Module to fit Gaussian processes

See the manual at https://gattocrucco.github.io/lsqfitgp/docs
"""

__version__ = '0.18'

from . import _patch_jax # keep first due to pre-import jax configs
from . import _patch_gvar

from ._GP import *
from ._Kernel import *
from ._kernels import *
from ._array import *
from ._fit import *
from ._Deriv import *
from ._fastraniter import *

from . import copula
from . import bayestree
