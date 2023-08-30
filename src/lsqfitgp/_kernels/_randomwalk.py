# lsqfitgp/_kernels/_randomwalk.py
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

import jax
from jax import numpy as jnp

from .. import _jaxext
from .._Kernel import kernel, stationarykernel

@kernel(derivable=False, maxdim=1)
def Wiener(x, y):
    """
    Wiener kernel.
    
    .. math::
        k(x, y) = \\min(x, y), \\quad x, y > 0
    
    A kernel representing a non-differentiable random walk starting at 0.
    
    Reference: Rasmussen and Williams (2006, p. 94).
    """
    with _jaxext.skipifabstract():
        assert jnp.all(x >= 0)
        assert jnp.all(y >= 0)
    return jnp.minimum(x, y)

def _fracbrownian_derivable(H=1/2, K=1):
    return H == 1 and K == 1
    # TODO fails under tracing, return None if not concrete

@kernel(derivable=_fracbrownian_derivable, maxdim=1)
def FracBrownian(x, y, H=1/2, K=1):
    """
    Bifractional Brownian motion kernel.
    
    .. math::
        k(x, y) = \\frac 1{2^K} \\big(
            (|x|^{2H} + |y|^{2H})^K - |x-y|^{2HK}
        \\big), \\quad H, K \\in (0, 1]
    
    For :math:`H = 1/2` (default) it is the Wiener kernel. For :math:`H \\in (0, 1/2)` the
    increments are anticorrelated (strong oscillation), for :math:`H \\in (1/2, 1]`
    the increments are correlated (tends to keep a slope).
    
    Reference: Houdré and Villa (2003).
    """
        
    # TODO I think the correlation between successive same step increments
    # is 2^(2H-1) - 1 in (-1/2, 1). Maybe add this to the docstring.
    
    with _jaxext.skipifabstract():
        assert 0 < H <= 1, H
        assert 0 < K <= 1, K
    H2 = 2 * H
    return 1 / 2 ** K * ((jnp.abs(x) ** H2 + jnp.abs(y) ** H2) ** K - jnp.abs(x - y) ** (H2 * K))

# redefine derivatives of min and max because jax default is to yield 1/2
# when x == y, while I need 1 but consistently between min/max

@jax.custom_jvp
def _minimum(x, y):
    return jnp.minimum(x, y)

@_minimum.defjvp
def _minimum_jvp(primals, tangents):
    x, y = primals
    xdot, ydot = tangents
    return _minimum(x, y), jnp.where(x < y, xdot, ydot)

@jax.custom_jvp
def _maximum(x, y):
    return jnp.maximum(x, y)

@_maximum.defjvp
def _maximum_jvp(primals, tangents):
    x, y = primals
    xdot, ydot = tangents
    return _maximum(x, y), jnp.where(x >= y, xdot, ydot)

@kernel(derivable=1, maxdim=1)
def WienerIntegral(x, y):
    """
    Kernel for a process whose derivative is a Wiener process.
    
    .. math::
        k(x, y) = \\frac 12 a^2 \\left(b - \\frac a3 \\right),
        \\quad a = \\min(x, y), b = \\max(x, y)
    
    """
    
    # TODO can I generate this algorithmically for arbitrary integration order?
    # If I don't find a closed formula I can use sympy. =>
    # JuliaGaussianProcesses implements it, copy their code
    
    with _jaxext.skipifabstract():
        assert jnp.all(x >= 0)
        assert jnp.all(y >= 0)
    a = _minimum(x, y)
    b = _maximum(x, y)
    return 1/2 * a ** 2 * (b - a / 3)

@kernel(derivable=False, maxdim=1)
def OrnsteinUhlenbeck(x, y):
    """
    Ornstein-Uhlenbeck process kernel.
    
    .. math::
        k(x, y) = \\exp(-|x - y|) - \\exp(-(x + y)),
        \\quad x, y \\ge 0
    
    It is a random walk plus a negative feedback term that keeps the
    asymptotical variance constant. It is asymptotically stationary; often the
    name "Ornstein-Uhlenbeck" is given to the stationary part only, which here
    is provided as `Expon`.
    
    """
    
    # TODO reference? look on wikipedia
    
    with _jaxext.skipifabstract():
        assert jnp.all(x >= 0)
        assert jnp.all(y >= 0)
    return jnp.exp(-jnp.abs(x - y)) - jnp.exp(-(x + y))

@kernel(derivable=False, maxdim=1)
def BrownianBridge(x, y):
    """
    Brownian bridge kernel.
    
    .. math::
        k(x, y) = \\min(x, y) - xy,
        \\quad x, y \\in [0, 1]
    
    It is a Wiener process conditioned on being zero at x = 1.
    """
    
    # TODO reference? look on wikipedia
    
    # TODO can this have a Hurst index? I think the kernel would be
    # (t^2H(1-s) + s^2H(1-t) + s(1-t)^2H + t(1-s)^2H - (t+s) - |t-s|^2H + 2ts)/2
    # but I have to check if it is correct. (In new kernel FracBrownianBridge.)
    
    with _jaxext.skipifabstract():
        assert jnp.all(x >= 0) and jnp.all(x <= 1)
        assert jnp.all(y >= 0) and jnp.all(y <= 1)
    return jnp.minimum(x, y) - x * y

def _stationaryfracbrownian_derivable(H=1/2):
    return H == 1

@stationarykernel(derivable=_stationaryfracbrownian_derivable, input='signed', maxdim=1)
def StationaryFracBrownian(delta, H=1/2):
    """
    Stationary fractional Brownian motion kernel.
    
    .. math::
        k(\\Delta) = \\frac 12 (|\\Delta+1|^{2H} + |\\Delta-1|^{2H} - 2|\\Delta|^{2H}),
        \\quad H \\in (0, 1]
        
    Reference: Gneiting and Schlather (2006, p. 272).
    """
    
    # TODO older reference, see [29] is GS06.
    
    with _jaxext.skipifabstract():
        assert 0 < H <= 1, H
    H2 = 2 * H
    return 1/2 * (jnp.abs(delta + 1) ** H2 + jnp.abs(delta - 1) ** H2 - 2 * jnp.abs(delta) ** H2)
    
    # TODO is the bifractional version of this valid?
