# lsqfitgp/_kernels/_wendland.py
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

from jax import numpy as jnp

from .. import _jaxext
from .._Kernel import stationarykernel, isotropickernel

def _wendland_derivable(k=0, **_):
    return k

def _wendland_maxdim(k=0, alpha=1):
    with _jaxext.skipifabstract():
        return int(jnp.floor(2 * alpha - 1))

@isotropickernel(input='posabs', derivable=_wendland_derivable, maxdim=_wendland_maxdim)
def Wendland(r, k=0, alpha=1):
    """
    Wendland kernel.
    
    .. math::
        k(r) &= \\frac1{B(2k+1,\\nu)}
        \\int_r^\\infty \\mathrm du\\, (u^2 - r^2)^k (1 - u)_+^{\\nu-1}, \\\\
        \\quad k &\\in \\mathbb N,\\ \\nu = k + \\alpha,\\ \\alpha \\ge 1.
    
    An isotropic kernel with finite support. The covariance is nonzero only
    when the distance between the points is less than 1. Parameter :math:`k \\in \\{0,
    1, 2, 3\\}` sets the differentiability, while the maximum dimensionality the
    kernel can be used in is :math:`\\lfloor 2\\alpha-1 \\rfloor`. Default is
    :math:`k = 0` (non derivable), :math:`\\alpha = 1` (can be used only in
    1D).
    
    Reference: Gneiting (2002), Wendland (2004, p. 128), Rasmussen and Williams
    (2006, p. 87), Porcu, Furrer and Nychka (2020, p. 4).
    
    """
        
    # TODO compute the kernel only on the nonzero points.
    
    # TODO find the nonzero points in O(nlogn) instead of O(n^2) by sorting
    # the inputs, and output a sparse matrix => on second thought this should
    # be a general mechanism implemented in GP that gives sparse x and y to
    # the kernel
    
    with _jaxext.skipifabstract():
        D = _wendland_maxdim(k, alpha)
        assert D >= 1, D
    
    if k == 0:
        poly = [
            [1],
        ]
    elif k == 1:
        poly = [
            [1, 1],
            [1],
        ]
    elif k == 2:
        poly = [
            [1/3, 4/3, 1],
            [1, 2],
            [1],
        ]
    elif k == 3:
        poly = [
            [1/15, 3/5, 23/15, 1],
            [2/5, 12/5, 3],
            [1, 3],
            [1],
        ]
    else:
        raise NotImplementedError
    
    nu = k + alpha
    coeffs = jnp.array([jnp.polyval(jnp.array(pj), nu) for pj in poly])
    poly = jnp.polyval(coeffs, r)
    return jnp.where(r < 1, (1 - r) ** (nu + k) * poly, 0)


# adapted from the "GP-circular" example in the PyMC documentation

# TODO maxdim actually makes sense only for isotropic. I need a way to say
# structured/non structured. Maybe all this should just live in the test suite.

# TODO Any stationary kernel supported on [0, pi] would be fine combined with
# the geodesic distance. Use the generic wendland kernel. Options:
# 1) add here the parameters of Wendland
# 2) add a distance option in stationary to use the angular distance, then
#    let the user apply it to wendland => the problem is that the user would
#    need to be careful with positiveness, while wendland checks it for him

@stationarykernel(derivable=1, maxdim=1, input='posabs')
def Circular(delta, tau=4, c=1/2):
    """
    Circular kernel.
    
    .. math:: k(x, y) &= W_c(d_{\\text{geo}}(x, y)), \\\\
        W_c(t) &= \\left(1 + \\tau\\frac tc\\right)
            \\left(1 - \\frac tc\\right)^\\tau_+,
        \\quad c \\in (0, 1/2], \\tau \\ge 4, \\\\
        d_{\\text{geo}}(x, y) &= \\arccos\\cos(2\\pi(x-y)).
    
    It is a stationary periodic kernel with period 1.
    
    Reference: Padonou and Roustant (2016).
    """
    with _jaxext.skipifabstract():
        assert tau >= 4, tau
        assert 0 < c <= 1/2, c
    x = delta % 1
    t = jnp.minimum(x, 1 - x)
    return (1 + tau * t / c) * jnp.maximum(1 - t / c, 0) ** tau
