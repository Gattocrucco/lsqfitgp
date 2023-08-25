# lsqfitgp/_kernels/_matern.py
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
from .. import _special
from .._Kernel import isotropickernel

def _maternp_derivable(p=None):
    return p

@isotropickernel(derivable=_maternp_derivable)
def Maternp(r2, p=None):
    """
    Matérn kernel of half-integer order. 
    
    .. math::
        k(r) &= \\frac {2^{1-\\nu}} {\\Gamma(\\nu)} x^\\nu K_\\nu(x) = \\\\
        &= \\exp(-x) \\frac{p!}{(2p)!}
        \\sum_{i=0}^p \\frac{(p+i)!}{i!(p-i)!} (2x)^{p-i} \\\\
        \\nu &= p + 1/2,
        p \\in \\mathbb N,
        x = \\sqrt{2\\nu} r
    
    The degree of derivability is :math:`p`.

    Reference: Rasmussen and Williams (2006, p. 85).
    """
    with _jaxext.skipifabstract():
        assert int(p) == p and p >= 0, p
    r2 = (2 * p + 1) * r2
    return _special.kvmodx2_hi(r2 + 1e-30, p)
    # TODO see if I can remove the 1e-30 improving kvmodx2_hi_jvp

def _matern_derivable(nu=None):
    with _jaxext.skipifabstract():
        return int(max(0, jnp.ceil(nu) - 1))

@isotropickernel(derivable=_matern_derivable)
def Matern(r2, nu=None):
    """
    Matérn kernel of real order. 
    
    .. math::
        k(r) = \\frac {2^{1-\\nu}} {\\Gamma(\\nu)} x^\\nu K_\\nu(x),
        \\quad \\nu \\ge 0,
        \\quad x = \\sqrt{2\\nu} r
    
    The process is :math:`\\lceil\\nu\\rceil-1` times derivable: so for
    :math:`0 \\le \\nu \\le 1` it is not derivable, for :math:`1 < \\nu \\le 2`
    it is derivable but has not a second derivative, etc. The highest
    derivative is (Lipschitz) continuous iff :math:`\\nu\\bmod 1 \\ge 1/2`.

    Reference: Rasmussen and Williams (2006, p. 84).
    """
    with _jaxext.skipifabstract():
        assert 0 <= nu < jnp.inf, nu
    r2 = 2 * jnp.where(nu, nu, 1) * r2  # for v = 0 the correct limit is white
                                        # noise, so I avoid doing r2 * 0
    return _special.kvmodx2(nu, r2)
    
    # TODO broken for high nu. However the convergence to ExpQuad is extremely
    # slow. Tentative temporary patch:
    # - for large x, when x^v=inf, use https://dlmf.nist.gov/10.25.E3
    # - for small x, when Kv(x)=inf, return 1
    # - for very large v, use expquad even if it's not good enough
    
    # The GSL has log K_nu
    # https://www.gnu.org/software/gsl/doc/html/specfunc.html#irregular-modified-bessel-functions-fractional-order
    
# def _bessel_scale(nu):
#     lnu = numpy.floor(nu)
#     rnu = numpy.ceil(nu)
#     zl, = special.jn_zeros(lnu, 1)
#     if lnu == rnu:
#         return zl
#     else:
#         zr, = special.jn_zeros(rnu, 1)
#         return zl + (nu - lnu) * (zr - zl) / (rnu - lnu)

def _bessel_derivable(nu=0):
    with _jaxext.skipifabstract():
        return int(nu // 2)

# TODO looking at the plot in the reference, it seems derivable also for nu = 0.
# what's up? investigate numerically by overwriting the derivability. rasmussen
# does not say anything about it. Problem: my custom derivatives may not work
# properly in this case.

def _bessel_maxdim(nu=0):
    with _jaxext.skipifabstract():
        return 2 * int(jnp.floor(nu) + 1)

@isotropickernel(derivable=_bessel_derivable, maxdim=_bessel_maxdim)
def Bessel(r2, nu=0):
    """
    Bessel kernel.
    
    .. math:: k(r) = \\Gamma(\\nu + 1) 2^\\nu (sr)^{-\\nu} J_{\\nu}(sr),
        \\quad s = 2 + \\nu / 2,
        \\quad \\nu \\ge 0,
    
    where :math:`s` is a crude estimate of the half width at half maximum of
    :math:`J_\\nu`. Can be used in up to :math:`2(\\lfloor\\nu\\rfloor + 1)`
    dimensions and derived up to :math:`\\lfloor\\nu/2\\rfloor` times.
    
    Reference: Rasmussen and Williams (2006, p. 89).
    """
    with _jaxext.skipifabstract():
        assert 0 <= nu < jnp.inf, nu
    r2 = r2 * (2 + nu / 2) ** 2
    return _special.gamma(nu + 1) * _special.jvmodx2(nu, r2)

    # nu >= (D-2)/2
    # 2 nu >= D - 2
    # 2 nu + 2 >= D
    # D <= 2 (nu + 1)
