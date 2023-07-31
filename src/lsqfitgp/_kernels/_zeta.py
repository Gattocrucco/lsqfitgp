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

from jax import numpy as jnp

from .. import _special
from .. import _patch_jax
from .. import _Kernel
from .._Kernel import stationarykernel

def _zeta_derivable(nu=None):
    with _patch_jax.skipifabstract():
        return max(0, jnp.ceil(nu) - 1)

@stationarykernel(maxdim=1, derivable=_zeta_derivable, saveargs=True)
def _ZetaBase(delta, nu=None):
    """
    Zeta kernel.
    
    .. math::
        k(\\Delta)
        &= \\frac{\\Re F(\\Delta, s)}{\\zeta(s)} =
        \\qquad (s = 1 + 2 \\nu, \\quad \\nu \\ge 0) \\\\
        &= \\frac1{\\zeta(s)} \\sum_{k=1}^\\infty
        \\frac {\\cos(2\\pi k\\Delta)} {k^s} = \\\\
        &= -(-1)^{s/2}
        \\frac {(2\\pi)^s} {2s!}
        \\frac {\\tilde B_s(\\Delta)} {\\zeta(s)}
        \\quad \\text{for even integer $s$.}
    
    It is equivalent to fitting with a Fourier series of period 1 with
    independent priors on the coefficients with mean zero and variance
    :math:`1/(\\zeta(s)k^s)` for the :math:`k`-th term. Analogously to
    :class:`Matern`, the process is :math:`\\lceil\\nu\\rceil - 1` times
    derivable, and the highest derivative is continuous iff :math:`\\nu\\bmod 1
    \\ge 1/2`.
    
    Note that the :math:`k = 0` term is not included in the summation, so the
    mean of the process over one period is forced to be zero.
    
    Reference: Petrillo (2022).
    
    """
    
    # TODO reference as covariance function? I had found something with
    # fourier and bernoulli but lost it. Maybe known as zeta?
    
    # TODO add constant as option, otherwise I can't compute the Fourier
    # series when I add a constant. => Maybe this will be solved when I
    # improve the transformations system.
    
    # TODO ND version. The separable product is not equivalent I think.
    
    # TODO the derivative w.r.t. nu is probably broken
    
    with _patch_jax.skipifabstract():
        assert 0 <= nu < jnp.inf, nu
        
    s = 1 + 2 * nu
    nupos = _special.periodic_zeta(delta, s) / _special.zeta(s)
    nuzero = jnp.where(delta % 1, 0, 1)
    return jnp.where(s > 1, nupos, nuzero)
    
    # return -(-1) ** (s // 2) * _special.scaled_periodic_bernoulli(s, delta) / jspecial.zeta(s, 1)
    
    # TODO use the bernoully version for integer even s, based on the type of
    # the input so that it's static, because it is much more accurate
    
class Zeta(_ZetaBase):
    
    __doc__ = _ZetaBase.__doc__
    
    # TODO write a method of _KernelBase that makes a new kernel from the
    # current one to be used as starting point by all the transformation
    # methods. It should have an option on how much the subclass should be
    # preserved, for example this implementation of `fourier` is broken as soon
    # as a transformation is applied to the kernel. Linear transformations
    # should transform not only the kernel but also its transformations. I
    # have to think how to make this work in full generality. => Tentative
    # design: loc, scale, dim, etc. must become standalone methods and
    # appropriately transform the other transformation methods. Alternative:
    # interface for providing the internals of the transformations, separately
    # for cross and symmetric cases, and it's always the Kernel method that
    # manages everything like it is for __call__.
    
    def fourier(self, dox, doy):
        
        # TODO problem: this ignores completely loc and scale. Write a static
        # _KernelBase method that applies loc and scale to a kernel core, use
        # it in __init__ and here.
        
        if not dox and not doy:
            return self
        
        nu = self.initargs['nu']
        s = 1 + 2 * nu
        
        if dox and doy:
            def kernel(k, q):
                order = jnp.ceil(k / 2)
                denom = order ** s * _special.zeta(s)
                return jnp.where((k == q) & (k > 0), 1 / denom, 0)
        
        else:
            def kernel(k, y):
                order = jnp.ceil(k / 2)
                denom = order ** s * _special.zeta(s)
                odd = k % 2
                arg = 2 * jnp.pi * order * y
                return jnp.where(k > 0, jnp.where(odd, jnp.sin(arg), jnp.cos(arg)) / denom, 0)
        
            if doy:
                kernel = lambda x, q, kernel=kernel: kernel(q, x)
        
        cls = _Kernel.Kernel if dox == doy and isinstance(self, _Kernel.Kernel) else _Kernel.CrossKernel
        obj = cls(kernel)
        obj.initargs = self.initargs
        obj._maxderivable = self._maxderivable
        return obj
