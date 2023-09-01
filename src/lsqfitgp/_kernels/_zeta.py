# lsqfitgp/_kernels/_zeta.py
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

import functools

from jax import numpy as jnp

from .. import _special
from .. import _jaxext
from .. import _Kernel

__all__ = ['Zeta']

def zeta_derivable(*, nu):
    with _jaxext.skipifabstract():
        return int(max(0, jnp.ceil(nu) - 1))

@_Kernel.stationarykernel(maxdim=1, derivable=zeta_derivable)
def Zeta(delta, *, nu):
    r"""
    
    Zeta kernel.
    
    .. math::
        k(\Delta)
        &= \frac{\Re F(\Delta, s)}{\zeta(s)} =
        \qquad (s = 1 + 2 \nu, \quad \nu \ge 0) \\
        &= \frac1{\zeta(s)} \sum_{k=1}^\infty
        \frac {\cos(2\pi k\Delta)} {k^s} = \\
        &= -(-1)^{s/2}
        \frac {(2\pi)^s} {2s!}
        \frac {\tilde B_s(\Delta)} {\zeta(s)}
        \quad \text{for even integer $s$.}
    
    It is equivalent to fitting with a Fourier series of period 1 with
    independent priors on the coefficients with mean zero and variance
    :math:`1/(\zeta(s)k^s)` for the :math:`k`-th term. Analogously to
    :class:`Matern`, the process is :math:`\lceil\nu\rceil - 1` times
    derivable, and the highest derivative is continuous iff :math:`\nu\bmod 1
    \ge 1/2`.
    
    The :math:`k = 0` term is not included in the summation, so the mean of the
    process over one period is forced to be zero.

    Reference: Petrillo (2022).
    
    """
    with _jaxext.skipifabstract():
        assert 0 <= nu < jnp.inf, nu
        
    s = 1 + 2 * nu
    nupos = _special.periodic_zeta(delta, s) / _special.zeta(s)
    nuzero = jnp.where(delta % 1, 0, 1)
    return jnp.where(s > 1, nupos, nuzero)
    
    # return -(-1) ** (s // 2) * _special.scaled_periodic_bernoulli(s, delta) / jspecial.zeta(s, 1)
    
    # TODO use the bernoully version for integer even s, based on the type of
    # the input such that it's static, because it is much more accurate

    # TODO ND version. The separable product is not equivalent I think.
    
    # TODO the derivative w.r.t. nu is probably broken
    
@_Kernel.kernel(maxdim=1, derivable=False)
def ZetaFourier(k, q, *, nu):
    s = 1 + 2 * nu
    order = jnp.ceil(k / 2)
    denom = order ** s * _special.zeta(s)
    return jnp.where((k == q) & (k > 0), 1 / denom, 0)

def crosszeta_derivable(*, nu):
    return 0, zeta_derivable(nu=nu)

@_Kernel.crosskernel(maxdim=1, derivable=crosszeta_derivable)
def CrossZetaFourier(k, y, *, nu):
    s = 1 + 2 * nu
    order = jnp.ceil(k / 2)
    denom = order ** s * _special.zeta(s)
    odd = k % 2
    arg = 2 * jnp.pi * order * y
    return jnp.where(k > 0, jnp.where(odd, jnp.sin(arg), jnp.cos(arg)) / denom, 0)

@functools.partial(Zeta.register_linop, argparser=lambda do: do if do else None)
def fourier(_, self, dox, doy):
    r"""

    Compute the Fourier series transform of the function.

    .. math::

        T(f)(k) = \begin{cases}
                \frac2T \int_0^T \mathrm dx\, f(x)
                \cos\left(\frac{2\pi}T \frac k2 x\right)
                & \text{if $k$ is even} \\
                \frac2T \int_0^T \mathrm dx\, f(x)
                \sin\left(\frac{2\pi}T \frac{k+1}2 x\right)
                & \text{if $k$ is odd}
            \end{cases}
        
    The period :math:`T` is 1.

    """
    
    nu = self.initkw['nu']
    
    if dox and doy:
        return ZetaFourier(nu=nu)    
    elif dox:
        return CrossZetaFourier(nu=nu)
    else:
        return CrossZetaFourier(nu=nu)._swap()

# TODO
# - use make_linop_family
# - change the `superclass` param of `kernel` to `bases` to allow passing
#   additional classes, AffineSpan in particular
#     - a more user-friendly alternative would be an option just for
#       affine span, hiding the class as implementation detail.
# - make Zeta an AffineSpan subclass
# - implement the scaling and phases of the fourier series
# - consider renaming fourier to fourier_series
