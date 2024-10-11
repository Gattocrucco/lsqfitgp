# lsqfitgp/_kernels/__init__.py
#
# Copyright (c) 2023, 2024, Giacomo Petrillo
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

# Keep this file a pure import list.

from ._arma import (
    MA,
    AR,
)
from ._bart import BART
from ._basic import (
    Constant,
    White,
    ExpQuad,
    Linear,
    GammaExp,
    NNKernel,
    Gibbs,
    Periodic,
    Categorical,
    Rescaling,
    Expon,
    BagOfWords,
    HoleEffect,
    Cauchy,
    CausalExpQuad,
    Decaying,
    Log,
    Taylor,
)
from ._celerite import (
    Celerite,
    Harmonic,
)
from ._matern import (
    Maternp,
    Matern,
    Bessel,
)
from ._randomwalk import (
    Wiener,
    FracBrownian,
    WienerIntegral,
    OrnsteinUhlenbeck,
    BrownianBridge,
    StationaryFracBrownian,
)
from ._spectral import (
    Cos,
    Pink,
    Color,
    Sinc,
)
from ._wendland import (
    Wendland,
    Circular,
)
from ._zeta import Zeta

# TODO add explicit exponent parameter to all infinitely divisible kernels

# TODO new kernels LM(formula) and LMER(formula).
