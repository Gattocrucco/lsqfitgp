# lsqfitgp/_special/__init__.py
#
# Copyright (c) 2022, 2024, Giacomo Petrillo
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

from ._bernoulli import (
    periodic_bernoulli,
    scaled_periodic_bernoulli,
)
from ._bessel import (
    j0,
    j1,
    jv,
    jvp,
    kv,
    kvp,
    iv,
    ivp,
    jvmodx2,
    kvmodx2,
    kvmodx2_hi,
)
from ._exp import (
    expm1x,
)
from ._expint import (
    expn_imag,
    exp1_imag,
    ci,
)
from ._gamma import (
    sgngamma,
    gamma,
    poch,
    gamma_incr,
    gammaln1,
)
from ._sinc import sinc
from ._taylor import taylor
from ._zeta import (
    zeta,
    hurwitz_zeta,
    periodic_zeta,
    zeta_zero,
    zeta_series_power_diff,
)
