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

    l = _special.ci(delta)
    r = _special.ci(delta * (1 + dw))
    mean = delta * (1 + dw / 2)
    norm = jnp.log1p(dw)
    tol = jnp.sqrt(jnp.finfo(jnp.empty(0).dtype).eps)
    
    
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
    
