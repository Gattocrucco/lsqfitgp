# lsqfitgp/_kernels/_spectral.py
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
from .._Kernel import stationarykernel

@stationarykernel(derivable=True, maxdim=1)
def Cos(delta):
    """
    Cosine kernel.
    
    .. math::
        k(\\Delta) = \\cos(\\Delta)
        = \\cos x \\cos y + \\sin x \\sin y
    
    Samples from this kernel are harmonic functions. It can be multiplied with
    another kernel to introduce anticorrelations.
    
    """
    # TODO reference?
    return jnp.cos(delta)

    
@stationarykernel(maxdim=1, derivable=1, input='abs')
def Pink(delta, dw=1):
    """
    Pink noise kernel.
    
    .. math::
        k(\\Delta) &= \\frac 1 {\\log(1 + \\delta\\omega)}
        \\int_1^{1+\\delta\\omega} \\mathrm d\\omega
        \\frac{\\cos(\\omega\\Delta)}\\omega = \\\\
        &= \\frac {     \\operatorname{Ci}(\\Delta (1 + \\delta\\omega))
                        - \\operatorname{Ci}(\\Delta)                   }
        {\\log(1 + \\delta\\omega)}
    
    A process with power spectrum :math:`1/\\omega` truncated between 1 and
    :math:`1 + \\delta\\omega`. :math:`\\omega` is the angular frequency
    :math:`\\omega = 2\\pi f`. In the limit :math:`\\delta\\omega\\to\\infty`
    it becomes white noise. Derivable one time.
    
    """
    # TODO reference?

    l = _special.ci(delta)
    r = _special.ci(delta * (1 + dw))
    mean = delta * (1 + dw / 2)
    norm = jnp.log1p(dw)
    tol = jnp.sqrt(jnp.finfo(jnp.empty(0).dtype).eps)
    
    # TODO choose better this tolerance by estimating error terms
    
    return jnp.where(delta * dw < tol, jnp.cos(mean), (r - l) / norm)

def _color_derivable(n=2):
    return n // 2 - 1

@stationarykernel(maxdim=1, derivable=_color_derivable, input='abs')
def Color(delta, n=2):
    """
    Colored noise kernel.
    
    .. math::
        k(\\Delta) &= (n-1) \\Re E_n(-i\\Delta) = \\\\
        &= (n-1) \\int_1^\\infty \\mathrm d\\omega
        \\frac{\\cos(\\omega\\Delta)}{\\omega^n},
        \\quad n \\in \\mathbb N, n \\ge 2.
    
    A process with power spectrum :math:`1/\\omega^n` truncated below
    :math:`\\omega = 1`. :math:`\\omega` is the angular frequency
    :math:`\\omega = 2\\pi f`. Derivable :math:`\\lfloor n/2 \\rfloor - 1`
    times.
     
    """
    # TODO reference?
    
    # TODO real n > 1 => use usual series for small z and cont frac for large?
    
    # TODO integration limit dw like Pink, use None to mean inf since the
    # derivability changes and I can not use values for conditions due to the
    # jit, delete Pink
    
    # TODO The most general possible version I can imagine is int_a^b dw e^iwx
    # w^n with n in R. With n < -1 b=None is allowed, and a=None with n < 1,
    # meaning inf and 0. Since the general implementation would be slower, I can
    # detect n is an integer (at compile time) to use expn_imag. Afterward
    # delete Sinc and call this Power.
    #
    # This is the generalized incomplete gamma function with real argument and
    # imaginary parameters. DLMF says there is no known fixed-precision
    # implementation!
    #
    # int_l^h dw w^n e^-ixw =                               t = ixw
    #   = (ix)^-(n+1) int_ixl^ixh dt t^n e^-t =
    #   = (ix)^-(n+1) gammainc(n + 1, ixl, ixh) =           z = ix, a = n+1
    #   = z^-a gammainc(a, lz, hz) =
    #   = z^-a (gamma(a, hz) - gamma(a, lz)) =
    #   = z^-a (Gamma(a, lz) - Gamma(a, hz))
    #
    # int_0^1 dw w^n e^-ixw =
    #   = Gamma(n + 1) gamma*(n + 1, ix)
    # https://www.boost.org/doc/libs/1_79_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html boost impl of gamma(a,x) and Gamma(a,x) for real x
    # https://dlmf.nist.gov/8.2.E7 gamma*(a, z)
    # https://dlmf.nist.gov/8.8.E14 d/da gamma*(a, z)
    # https://dlmf.nist.gov/8.8.E15 d^n/dz^n z^-a gamma(a, z)
    # https://dlmf.nist.gov/8.9 continued fractions for gamma* and Gamma
    # https://dlmf.nist.gov/8.11.E2 asymp series for Gamma(a, z)
    # https://dlmf.nist.gov/8.12 series for a->∞ in terms of erfc (do I have complex erfc? => yes scipy, no jax)
    # https://dlmf.nist.gov/8.19.E17 continued fraction for E_n(z)
    # https://dlmf.nist.gov/8.20.E6 series of E_n(z) for large n
    # https://dlmf.nist.gov/8.27.i reference to Padé of gammainc for complex z
    # 683.f impl of E_n(z) for complex z (license?) (amos1990)
    # https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.expint implementation in julia for arbitrary complex arguments
    # https://github.com/JuliaMath/SpecialFunctions.jl/blob/36c547b4a270b6089b1baf7bec05707e9fb8c7f9/src/expint.jl#L446-L503
    
    # E_p(z)
    # following scipy's implementation, generalized to noninteger arguments:
    # if p > 50, use https://dlmf.nist.gov/8.20.ii (the dlmf reports it only
    #                for real z, need to follow the references)
    # elif |z| < 1 and p integer, use https://dlmf.nist.gov/8.19.E8
    #                p noninteger, use https://dlmf.nist.gov/8.19.E10 (need to
    #                                  sum the external term accurately)
    # else (|z| >= 1), use https://dlmf.nist.gov/8.19.E17
    
    # https://github.com/Radonirinaunimi/cmpx-spfunc incomplete gamma for
    # complex arguments (not well tested yet)
    
    # Parametrize with n = 1 + 2 nu like Matérn.
    
    # Bartosch, L. (2001). "Generation of colored noise". International Journal of Modern Physics C. 12 (6): 851–855. Bibcode:2001IJMPC..12..851B. doi:10.1142/S0129183101002012. S2CID 54500670.
    
    assert int(n) == n and n >= 2, n
    return (n - 1) * _special.expn_imag(n, delta).real
    
@stationarykernel(derivable=True, input='posabs', maxdim=1)
def Sinc(delta):
    """
    Sinc kernel.
    
    .. math:: k(\\Delta) = \\operatorname{sinc}(\\Delta) =
        \\frac{\\sin(\\pi\\Delta)}{\\pi\\Delta}.
    
    Reference: Tobar (2019).
    """
    return _special.sinc(delta)
    
    # TODO is this isotropic? My current guess is that it works up to some
    # dimension due to a coincidence but is not in general isotropic.
