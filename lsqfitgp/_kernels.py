# lsqfitgp/_kernels.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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

import re
import collections
import functools
import sys

import jax
import numpy
from jax import numpy as jnp
from jax.scipy import special as jspecial
from jax import tree_util, lax
from scipy import special

from . import _array
from . import _Kernel
from . import _linalg
from . import _patch_jax
from ._Kernel import kernel, stationarykernel, isotropickernel

__all__ = [
    'AR',
    'BART',
    'BagOfWords',
    'Bessel',
    'BrownianBridge',
    'Categorical',
    'Cauchy',
    'CausalExpQuad',
    'Celerite',
    'Circular',
    'Color',
    'Constant',
    'Cos',
    'Decaying',
    'Expon',
    'ExpQuad',
    'Fourier',
    'FracBrownian',
    'GammaExp',
    'Gibbs',
    'Harmonic',
    'HoleEffect',
    'Linear',
    'Log',
    'MA',
    'Matern',
    'Maternp',
    'NNKernel',
    'OrnsteinUhlenbeck',
    'Periodic',
    'Pink',
    'Rescaling',
    'Sinc',
    'StationaryFracBrownian',
    'Taylor',
    'Wendland',
    'White',
    'Wiener',
    'WienerIntegral',
]

# TODO instead of adding forcekron by default to all 1D kernels, use maxdim=None
# by default in CrossKernel, add maxdim=1 to all 1D kernels, and let the user
# choose how to deal with nd (add option for sum-separation). Make an example
# about this in `multidimensional input`. Implement tests for separability
# on all kernels.

# TODO maybe I could have a continuity check like derivable, but to be useful
# it would be callable-only and take the derivation order. But I don't think
# any potential user needs it.

# TODO add explicit exponent parameter to infinitely divisible kernels, and
# check the exponent is an int in __pow__.

def _dot(x, y):
    return _Kernel.sum_recurse_dtype(lambda x, y: x * y, x, y)
    
@isotropickernel(derivable=True, input='raw')
def Constant(x, y):
    """
    Constant kernel.
    
    .. math::
        k(x, y) = 1
    
    This means that all points are completely correlated, thus it is equivalent
    to fitting with a horizontal line. This can be seen also by observing that
    1 = 1 x 1.
    """
    return jnp.ones(jnp.broadcast_shapes(x.shape, y.shape))
    
@isotropickernel(derivable=False, input='raw')
def White(x, y):
    """
    White noise kernel.
    
    .. math::
        k(x, y) = \\begin{cases}
            1 & x = y     \\\\
            0 & x \\neq y
        \\end{cases}
    """
    return _Kernel.prod_recurse_dtype(lambda x, y: x == y, x, y).astype(int)
    # TODO maybe StructuredArray should support equality and other operations

@isotropickernel(derivable=True)
def ExpQuad(r2):
    """
    Exponential quadratic kernel.
    
    .. math::
        k(r) = \\exp \\left( -\\frac 12 r^2 \\right)
    
    It is smooth and has a strict typical lengthscale, i.e., oscillations are
    strongly suppressed under a certain wavelength, and correlations are
    strongly suppressed over a certain distance.
    
    Reference: Rasmussen and Williams (2006, p. 83).
    """
    return jnp.exp(-1/2 * r2)

@kernel(derivable=True)
def Linear(x, y):
    """
    Dot product kernel.
    
    .. math::
        k(x, y) = x \\cdot y = \\sum_i x_i y_i
    
    In 1D it is equivalent to fitting with a line passing by the origin.

    Reference: Rasmussen and Williams (2006, p. 89).
    """
    return _dot(x, y)

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
    
    The degree of derivability is `p`.

    Reference: Rasmussen and Williams (2006, p. 85).
    """
    if _patch_jax.isconcrete(p):
        assert int(p) == p and p >= 0, p
    r2 = (2 * p + 1) * r2
    return _patch_jax.kvmodx2_hi(r2 + 1e-30, p)
    # TODO see if I can remove the 1e-30 improving kvmodx2_hi_jvp

def _matern_derivable(nu=None):
    return max(0, numpy.ceil(nu) - 1)

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
    derivative is continuous iff :math:`\\nu\\bmod 1 \\ge 1/2`.

    Reference: Rasmussen and Williams (2006, p. 84).
    """
    if _patch_jax.isconcrete(nu):
        assert 0 <= nu < jnp.inf, nu
    r2 = 2 * jnp.where(nu, nu, 1) * r2  # for v = 0 the correct limit is white
                                        # noise, so I avoid doing r2 * 0
    return _patch_jax.kvmodx2(nu, r2)
    
def _gammaexp_derivable(gamma=1):
    return gamma == 2

@isotropickernel(derivable=_gammaexp_derivable)
def GammaExp(r2, gamma=1):
    """
    Gamma exponential kernel.
    
    .. math::
        k(r) = \\exp(-r^\\gamma), \\quad
        \\gamma \\in (0, 2]
    
    For :math:`\\gamma = 2` it is the squared exponential kernel, for
    :math:`\\gamma = 1` (default) it is the Matérn 1/2 kernel, for
    :math:`\\gamma \\to 0` it tends to white noise plus a constant. The process
    is differentiable only for :math:`\\gamma = 2`, however as :math:`\\gamma`
    gets closer to 2 the variance of the non-derivable component goes to zero.

    Reference: Rasmussen and Williams (2006, p. 86).
    """
    if _patch_jax.isconcrete(gamma):
        assert 0 < gamma <= 2, gamma
    return jnp.exp(-(r2 ** (gamma / 2)))

@kernel(derivable=True)
def NNKernel(x, y, sigma0=1):
    """
    Neural network kernel.
    
    .. math::
        k(x, y) = \\frac 2 \\pi
        \\arcsin \\left( \\frac
        {
            2 (q + x \\cdot y)
        }{
            (1 + 2 (q + x \\cdot x))
            (1 + 2 (q + y \\cdot y))
        }
        \\right),
        \\quad q = \\texttt{sigma0}^2
    
    Kernel which is equivalent to a neural network with one infinite hidden
    layer with Gaussian priors on the weights and error function response. In
    other words, you can think of the process as a superposition of sigmoids
    where `sigma0` sets the dispersion of the centers of the sigmoids.
    
    Reference: Rasmussen and Williams (2006, p. 90).
    """
    
    # TODO the `2`s in the formula are a bit arbitrary. Remove them or give
    # motivation relative to the precise formulation of the neural network.
    if _patch_jax.isconcrete(sigma0):
        assert 0 < sigma0 < jnp.inf
    q = sigma0 ** 2
    denom = (1 + 2 * (q + _dot(x, x))) * (1 + 2 * (q + _dot(y, y)))
    return 2/jnp.pi * jnp.arcsin(2 * (q + _dot(x, y)) / denom)
    
    # TODO this is not fully equivalent to an arbitrary transformation on the
    # augmented vector even if x and y are transformed, unless I support q
    # being a vector or an additional parameter.

@kernel(forcekron=True, derivable=False)
def Wiener(x, y):
    """
    Wiener kernel.
    
    .. math::
        k(x, y) = \\min(x, y), \\quad x, y > 0
    
    A kernel representing a non-differentiable random walk starting at 0.
    
    Reference: Rasmussen and Williams (2006, p. 94).
    """
    if _patch_jax.isconcrete(x, y):
        assert numpy.all(_patch_jax.concrete(x) >= 0)
        assert numpy.all(_patch_jax.concrete(y) >= 0)
    return jnp.minimum(x, y)

@kernel(maxdim=sys.maxsize)
def Gibbs(x, y, scalefun=lambda x: 1):
    """
    Gibbs kernel.
    
    .. math::
        k(x, y) = \\sqrt{ \\frac {2 s(x) s(y)} {s(x)^2 + s(y)^2} }
        \\exp \\left( -\\frac {(x - y)^2} {s(x)^2 + s(y)^2} \\right),
        \\quad s = \\texttt{scalefun}.
    
    Kernel which in some sense is like a Gaussian kernel where the scale
    changes at every point. The scale is computed by the parameter `scalefun`
    which must be a callable taking the x array and returning a scale for each
    point. By default `scalefun` returns 1 so it is a Gaussian kernel.
    
    Consider that the default parameter `scale` acts before `scalefun`, so
    for example if `scalefun(x) = x` then `scale` has no effect. You should
    include all rescalings in `scalefun` to avoid surprises.
    
    Reference: Rasmussen and Williams (2006, p. 93).
    """
    sx = scalefun(x)
    sy = scalefun(y)
    if _patch_jax.isconcrete(sx, sy):
        assert numpy.all(_patch_jax.concrete(sx) > 0)
        assert numpy.all(_patch_jax.concrete(sy) > 0)
    denom = sx ** 2 + sy ** 2
    factor = jnp.sqrt(2 * sx * sy / denom)
    distsq = _Kernel.sum_recurse_dtype(lambda x, y: (x - y) ** 2, x, y)
    return factor * jnp.exp(-distsq / denom)

@stationarykernel(derivable=True, forcekron=True)
def Periodic(delta, outerscale=1):
    """
    Periodic Gaussian kernel.
    
    .. math::
        k(\\Delta) = \\exp \\left(
        -2 \\left(
        \\frac {\\sin(\\Delta / 2)} {\\texttt{outerscale}}
        \\right)^2
        \\right)
    
    A Gaussian kernel over a transformed periodic space. It represents a
    periodic process. The usual `scale` parameter sets the period, with the
    default `scale` = 1 giving a period of 2π, while the `outerscale` parameter
    sets the length scale of the correlations.
    
    Reference: Rasmussen and Williams (2006, p. 92).
    """
    if _patch_jax.isconcrete(outerscale):
        assert 0 < outerscale < jnp.inf
    return jnp.exp(-2 * (jnp.sin(delta / 2) / outerscale) ** 2)

@kernel(forcekron=True, derivable=False)
def Categorical(x, y, cov=None):
    """
    Categorical kernel.
    
    .. math::
        k(x, y) = \\texttt{cov}[x, y]
    
    A kernel over integers from 0 to N-1. The parameter `cov` is the covariance
    matrix of the values.
    """
        
    # TODO support sparse matrix for cov (replace jnp.asarray and numpy
    # check)
    
    assert jnp.issubdtype(x.dtype, jnp.integer)
    cov = jnp.asarray(cov)
    assert len(cov.shape) == 2
    assert cov.shape[0] == cov.shape[1]
    if _patch_jax.isconcrete(cov):
        C = _patch_jax.concrete(cov)
        assert numpy.allclose(C, C.T)
    return cov[x, y]

@kernel
def Rescaling(x, y, stdfun=None):
    """
    Outer product kernel.
    
    .. math::
        k(x, y) = \\texttt{stdfun}(x) \\texttt{stdfun}(y)
    
    A totally correlated kernel with arbitrary variance. Parameter `stdfun`
    must be a function that takes `x` or `y` and computes the standard
    deviation at the point. It can yield negative values; points with the same
    sign of `fun` will be totally correlated, points with different sign will
    be totally anticorrelated. Use this kernel to modulate the variance of
    other kernels. By default `stdfun` returns a constant, so it is equivalent
    to :class:`Constant`.
    
    """
    if stdfun is None:
        stdfun = lambda x: jnp.ones(x.shape)
        # do not use np.ones_like because it does not recognize StructuredArray
        # do not use x.dtype because it could be structured
    return stdfun(x) * stdfun(y)

@stationarykernel(derivable=True, forcekron=True)
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

def _fracbrownian_derivable(H=1/2, K=1):
    return H == 1 and K == 1
    # TODO fails under tracing, return None if not concrete, maybe silence
    # derivability warnings under tracing

@kernel(forcekron=True, derivable=_fracbrownian_derivable)
def FracBrownian(x, y, H=1/2, K=1):
    """
    Bifractional Brownian motion kernel.
    
    .. math::
        k(x, y) = \\frac 1{2^K} \\big(
            (|x|^{2H} + |y|^{2H})^K - |x-y|^{2HK}
        \\big), \\quad H, K \\in (0, 1]
    
    For `H` = 1/2 (default) it is the Wiener kernel. For `H` in (0, 1/2) the
    increments are anticorrelated (strong oscillation), for `H` in (1/2, 1]
    the increments are correlated (tends to keep a slope).
    
    Reference: Houdré and Villa (2003).
    """
        
    # TODO I think the correlation between successive same step increments
    # is 2^(2H-1) - 1 in (-1/2, 1). Maybe add this to the docstring.
    
    if _patch_jax.isconcrete(H, K):
        assert 0 < H <= 1, H
        assert 0 < K <= 1, K
    H2 = 2 * H
    return 1 / 2 ** K * ((jnp.abs(x) ** H2 + jnp.abs(y) ** H2) ** K - jnp.abs(x - y) ** (H2 * K))

def _wendland_derivable(k=0, alpha=1):
    return k

def _wendland_maxdim(k=0, alpha=1):
    return numpy.floor(2 * alpha - 1)

@isotropickernel(input='soft', derivable=_wendland_derivable, maxdim=_wendland_maxdim)
def Wendland(r, k=0, alpha=1):
    """
    Wendland kernel.
    
    .. math::
        k(r) &= \\frac1{B(2k+1,\\nu)}
        \\int_r^\\infty \\mathrm du\\, (u^2 - r^2)^k (1 - u)_+^{\\nu-1}, \\\\
        \\quad k &\\in \\mathbb N,\\ \\nu = k + \\alpha,\\ \\alpha \\ge 1.
    
    An isotropic kernel with finite support. The covariance is nonzero only
    when the distance between the points is less than 1. Parameter `k` in (0,
    1, 2, 3) sets the differentiability, while the maximum dimensionality the
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
    
    if _patch_jax.isconcrete(k, alpha):
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

@kernel(derivable=1, forcekron=True)
def WienerIntegral(x, y):
    """
    Kernel for a process whose derivative is a Wiener process.
    
    .. math::
        k(x, y) = \\frac 12 a^2 \\left(b - \\frac a3 \\right),
        \\quad a = \\min(x, y), b = \\max(x, y)
    
    """
    
    # TODO can I generate this algorithmically for arbitrary integration order?
    # If I don't find a closed formula I can use sympy.
    
    if _patch_jax.isconcrete(x, y):
        assert numpy.all(_patch_jax.concrete(x) >= 0)
        assert numpy.all(_patch_jax.concrete(y) >= 0)
    a = _minimum(x, y)
    b = _maximum(x, y)
    return 1/2 * a ** 2 * (b - a / 3)

@kernel(forcekron=True, derivable=True)
def Taylor(x, y):
    """
    Exponential-like power series kernel.
    
    .. math::
        k(x, y) = \\sum_{k=0}^\\infty \\frac {x^k}{k!} \\frac {y^k}{k!}
        = I_0(2 \\sqrt{xy})
    
    It is equivalent to fitting with a Taylor series expansion in zero with
    independent priors on the coefficients k with mean zero and standard
    deviation 1/k!.
    """
    # TODO reference? Maybe it's called bessel kernel in the literature?
        
    # TODO what is the "natural" extension of this to multidim? Is forcekron
    # appropriate?
    
    mul = x * y
    val = 2 * jnp.sqrt(jnp.abs(mul))
    return jnp.where(mul >= 0, jspecial.i0(val), _patch_jax.j0(val))

def _bernoulli_poly(n, x):
    # takes x mod 1
    # TODO to make this jittable, hardcode size to 60 and truncate by writing
    # zeros
    n = int(n)
    bernoulli = special.bernoulli(n)
    k = numpy.arange(n + 1)
    binom = special.binom(n, k)
    coeffs = binom[::-1] * bernoulli
    x = x % 1
    cond = x < 0.5
    x = jnp.where(cond, x, 1 - x)
    out = jnp.polyval(coeffs, x)
    if n % 2 == 1:
        out = out * jnp.where(cond, 1, -1)
    return out

@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _fourier(s, x):
    tau = 2 * jnp.pi
    lognorm = s * jnp.log(tau) - jspecial.gammaln(s + 1)
    norm = jnp.exp(lognorm) / 2
    cond = s < 60
    smalls = norm * _bernoulli_poly(s if cond else 1, x)
    # s -> ∞: -Re e^2πix / i^s
    # don't use 1j ** s because it is very inaccurate
    arg = tau * x
    sign = jnp.where((s // 2) % 2, 1, -1)
    larges = sign * jnp.where(s % 2, jnp.sin(arg), jnp.cos(arg))
    return jnp.where(cond, smalls, larges.real)

@_fourier.defjvp
def _fourier_jvp(s, primals, tangents):
    x, = primals
    xt, = tangents
    return _fourier(s, x), 2 * jnp.pi * _fourier(s - 1, x) * xt

def _fourier_derivable(n=2):
    return n - 1

@stationarykernel(forcekron=True, derivable=_fourier_derivable, saveargs=True)
def _FourierBase(delta, n=2):
    """
    Fourier kernel.
    
    .. math::
        k(\\Delta) &= \\frac1{\\zeta(2n)} \\sum_{k=1}^\\infty
        \\frac {\\cos(2\\pi kx)}{k^n} \\frac {\\cos(2\\pi ky)}{k^n}
        + \\frac1{\\zeta(2n)} \\sum_{k=1}^\\infty
        \\frac {\\sin(2\\pi kx)}{k^n} \\frac {\\sin(2\\pi ky)}{k^n} = \\\\
        &= \\frac1{\\zeta(2n)} \\sum_{k=1}^\\infty
        \\frac {\\cos(2\\pi k\\Delta)} {k^{2n}} = \\\\
        &= (-1)^n \\frac {(2\\pi)^{2n}} {2\\Gamma(2n)}
        \\frac{\\zeta(1 - 2n, \\Delta \\bmod 1)}{\\zeta(2n)} = \\\\
        &= (-1)^{n+1}
        \\frac1{\\zeta(2n)} \\frac {(2\\pi)^{2n}} {2(2n)!}
        B_{2n}(\\Delta \\bmod 1).
    
    It is equivalent to fitting with a Fourier series of period 1 with
    independent priors on the coefficients with mean zero and variance
    :math:`1/(\\zeta(2n)k^{2n})`. The process is :math:`n - 1` times derivable.
    
    Note that the :math:`k = 0` term is not included in the summation, so the
    mean of the process over one period is forced to be zero.
    
    Reference: https://dlmf.nist.gov/25.11.E14, https://dlmf.nist.gov/25.11.E9.
    
    """
    
    # TODO reference as covariance function? I had found something with
    # fourier and bernoulli but lost it. Maybe known as zeta?
    
    # TODO maxk parameter to truncate the series. I have to manually sum the
    # components? => Bad idea then. => Can I sum analitically the residual?
    
    # TODO add constant as option, otherwise I can't compute the Fourier
    # series when I add a constant. => Maybe this will be solved when I
    # improve the transformations system.
    
    # TODO ND version. The separable product is not equivalent I think.
    
    assert int(n) == n and n >= 1, n
    # TODO I could allow n == 0 to be a constant kernel
        
    s = 2 * n
    return -(-1) ** n * _fourier(s, delta) / jspecial.zeta(s, 1)

    # implementation for integer n with zeta:
    # s = 2 * n
    # a = delta % 1
    # lognorm = s * jnp.log(2 * jnp.pi) - jspecial.gammaln(s)
    # norm = (-1) ** n * jnp.exp(lognorm) / (2 * jspecial.zeta(s, 1))
    # return norm * jspecial.zeta(1 - s, a)

    # TODO real n
    # zeta(1-s, a) diverges for s -> oo
    
class Fourier(_FourierBase):
    
    __doc__ = _FourierBase.__doc__
    
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
        
        n = self.initargs.get('n', 2)
        s = 2 * n
        
        if dox and doy:
            def kernel(k, q):
                order = jnp.ceil(k / 2)
                denom = order ** s * special.zeta(s)
                return jnp.where((k == q) & (k > 0), 1 / denom, 0)
        
        else:
            def kernel(k, y):
                order = jnp.ceil(k / 2)
                denom = order ** s * special.zeta(s)
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

@kernel(forcekron=True, derivable=False)
def OrnsteinUhlenbeck(x, y):
    """
    Ornstein-Uhlenbeck process kernel.
    
    .. math::
        k(x, y) = \\exp(-|x - y|) - \\exp(-(x + y)),
        \\quad x, y \\ge 0
    
    It is a random walk plus a negative feedback term that keeps the
    asymptotical variance constant. It is asymptotically stationary; often the
    name "Ornstein-Uhlenbeck" is given to the stationary part only, which here
    is provided as :class:`Expon`.
    
    """
    
    # TODO reference? look on wikipedia
    
    if _patch_jax.isconcrete(x, y):
        assert numpy.all(_patch_jax.concrete(x) >= 0)
        assert numpy.all(_patch_jax.concrete(y) >= 0)
    return jnp.exp(-jnp.abs(x - y)) - jnp.exp(-(x + y))
    
def _Celerite_derivable(**kw):
    gamma = kw.get('gamma', 1)
    B = kw.get('B', 0)
    if jnp.isscalar(gamma) and jnp.isscalar(B) and B == gamma:
        return 1
    else:
        return False

@stationarykernel(forcekron=True, derivable=_Celerite_derivable, input='hard')
def Celerite(delta, gamma=1, B=0):
    """
    Celerite kernel.
    
    .. math::
        k(\\Delta) = \\exp(-\\gamma|\\Delta|)
        \\big( \\cos(\\Delta) + B \\sin(|\\Delta|) \\big)
    
    This is the covariance function of an AR(2) process with complex roots. The
    parameters must satisfy the condition :math:`|B| \\le \\gamma`. For
    :math:`B = \\gamma` it is equivalent to the :class:`Harmonic` kernel with
    :math:`\\eta Q = 1/B, Q > 1`, and it is derivable.
    
    Reference: Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, and Ruth
    Angus: *Fast and Scalable Gaussian Process Modeling With Applications To
    Astronomical Time Series*.
    """
    if _patch_jax.isconcrete(gamma, B):
        assert 0 <= gamma < jnp.inf, gamma
        assert abs(B) <= gamma, (B, gamma)
    return jnp.exp(-gamma * delta) * (jnp.cos(delta) + B * jnp.sin(delta))

@kernel(forcekron=True, derivable=False)
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
    
    if _patch_jax.isconcrete(x, y):
        X = _patch_jax.concrete(x)
        Y = _patch_jax.concrete(y)
        assert numpy.all(0 <= X) and numpy.all(X <= 1)
        assert numpy.all(0 <= Y) and numpy.all(Y <= 1)
    return jnp.minimum(x, y) - x * y

@stationarykernel(forcekron=True, derivable=1)
def Harmonic(delta, Q=1):
    """
    Damped stochastically driven harmonic oscillator kernel.
    
    .. math::
        k(\\Delta) =
        \\exp\\left( -\\frac {|\\Delta|} {Q} \\right)
        \\begin{cases}
            \\cosh(\\eta\\Delta) + \\sinh(\\eta|\\Delta|) / (\\eta Q)
            & 0 < Q < 1 \\\\
            1 + |\\Delta| & Q = 1 \\\\
            \\cos(\\eta\\Delta) + \\sin(\\eta|\\Delta|) / (\\eta Q)
            & Q > 1,
        \\end{cases}
    
    where :math:`\\eta = \\sqrt{|1 - 1/Q^2|}`.
    
    The process is the solution to the stochastic differential equation
    
    .. math::   f''(x) + 2/Q f'(x) + f(x) = w(x),
    
    where `w` is white noise.
    
    The parameter `Q` is the quality factor, i.e., the ratio between the energy
    stored in the oscillator and the energy lost in each cycle due to damping.
    The angular frequency is 1, i.e., the period is 2π. The process is derivable
    one time.
    
    In 1D, for `Q` = 1 (default) and `scale` = sqrt(1/3), it is the Matérn 3/2
    kernel.
    
    Reference: Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, and Ruth
    Angus: *Fast and Scalable Gaussian Process Modeling With Applications To
    Astronomical Time Series*.
    """
    
    # TODO improve and test the numerical accuracy for derivatives near x=0
    # and Q=1. I don't know if the derivatives have problems away from Q=1.
    
    # TODO probably second derivatives w.r.t. Q at Q=1 are wrong.
    
    # TODO will fail if Q is traced.
    
    if _patch_jax.isconcrete(Q):
        assert 0 < Q < jnp.inf, Q
    
    tau = jnp.abs(delta)
    
    if Q < 1/2:
        etaQ = jnp.sqrt((1 - Q) * (1 + Q))
        tauQ = tau / Q
        pexp = jnp.exp(_sqrt1pm1(-jnp.square(Q)) * tauQ)
        mexp = jnp.exp(-(1 + etaQ) * tauQ)
        return (pexp + mexp + (pexp - mexp) / etaQ) / 2
    
    elif 1/2 <= Q < 1:
        etaQ = jnp.sqrt(1 - jnp.square(Q))
        tauQ = tau / Q
        etatau = etaQ * tauQ
        return jnp.exp(-tauQ) * (jnp.cosh(etatau) + jnp.sinh(etatau) / etaQ)
        
    elif Q == 1:
        return _harmonic(tau, Q)
    
    else: # Q > 1
        etaQ = jnp.sqrt(jnp.square(Q) - 1)
        tauQ = tau / Q
        etatau = etaQ * tauQ
        return jnp.exp(-tauQ) * (jnp.cos(etatau) + jnp.sin(etatau) / etaQ)

def _sqrt1pm1(x):
    """sqrt(1 + x) - 1, numerically stable for small x"""
    return jnp.expm1(1/2 * jnp.log1p(x))

@jax.custom_jvp
def _matern32(x):
    return (1 + x) * jnp.exp(-x)

_matern32.defjvps(lambda g, ans, x: g * -x * jnp.exp(-x))

def _harmonic(x, Q):
    return _matern32(x / Q) + jnp.exp(-x/Q) * (1 - Q) * jnp.square(x) * (1 + x/3)

# def _harmonic(x, Q):
#     return np.exp(-x/Q) * (1 + x + (1 - Q) * x * (1 + x * (1 + x/3)))

# @autograd.extend.primitive
# def _harmonic(x, Q):
#     return (1 + x) * np.exp(-x)
#
# autograd.extend.defvjp(
#     _harmonic,
#     lambda ans, x, Q: lambda g: g * -np.exp(-x/Q) * x * (1 + (Q-1) * (1+x)),
#     lambda ans, x, Q: lambda g: g * -np.exp(-x) * x ** 3 / 3
# ) # d/dQ: -np.exp(-x/Q) * (3/Q**2 - 1) * x**3 / (6 * Q**2)
#
# autograd.extend.defjvp(
#     _harmonic,
#     lambda g, ans, x, Q: (g.T * (-np.exp(-x) * x).T).T,
#     lambda g, ans, x, Q: (g.T * (-np.exp(-x) * x ** 3 / 3).T).T
# )

@stationarykernel(forcekron=True, derivable=False, input='hard')
def Expon(delta):
    """
    Exponential kernel.
    
    .. math::
        k(\\Delta) = \\exp(-|\\Delta|)
    
    In 1D it is equivalent to the Matérn 1/2 kernel, however in more dimensions
    it acts separately while the Matérn kernel is isotropic.

    Reference: Rasmussen and Williams (2006, p. 85).
    """
    return jnp.exp(-delta)

_bow_regexp = re.compile(r'\s|[!«»"“”‘’/()\'?¡¿„‚<>,;.:-–—]')

@kernel(forcekron=True, derivable=False)
@numpy.vectorize
def BagOfWords(x, y):
    """
    Bag of words kernel.
    
    .. math::
        k(x, y) &= \\sum_{w \\in \\text{words}} c_w(x) c_w(y), \\\\
        c_w(x) &= \\text{number of times word $w$ appears in $x$}
    
    The words are defined as non-empty substrings delimited by spaces or one of
    the following punctuation characters: ! « » " “ ” ‘ ’ / ( ) ' ? ¡ ¿ „ ‚ < >
    , ; . : - – —.

    Reference: Rasmussen and Williams (2006, p. 100).
    """
    
    # TODO precompute the bags for x and y, then call a vectorized private
    # function.
    
    # TODO iterate on the shorter bag and use get on the other instead of
    # computing set intersection? Or: convert words to integers and then do
    # set intersection with sorted arrays?
    
    xbag = collections.Counter(_bow_regexp.split(x))
    ybag = collections.Counter(_bow_regexp.split(y))
    xbag[''] = 0 # why this? I can't recall
    ybag[''] = 0
    common = set(xbag) & set(ybag)
    return sum(xbag[k] * ybag[k] for k in common)

# TODO add bag of characters and maybe other text kernels

@stationarykernel(derivable=False, input='hard', forcekron=True)
def HoleEffect(delta):
    """
    
    Hole effect kernel.
    
    .. math:: k(\\Delta) = (1 - \\Delta) \\exp(-\\Delta)
    
    Reference: Dietrich and Newsam (1997, p. 1096).
    
    """
    return (1 - delta) * jnp.exp(-delta)

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
    return nu // 2

def _bessel_maxdim(nu=0):
    return 2 * (numpy.floor(nu) + 1)

@isotropickernel(derivable=_bessel_derivable, maxdim=_bessel_maxdim)
def Bessel(r2, nu=0):
    """
    Bessel kernel.
    
    .. math:: k(r) = \\Gamma(\\nu + 1) 2^\\nu (sr)^{-\\nu} J_{\\nu}(sr),
        \\quad s = 2 + \\nu / 2, \\nu > 0,
    
    where `s` is a crude estimate of the half width at half maximum of
    :math:`J_\\nu`. Can be used in up to :math:`2(\\lfloor\\nu\\rfloor + 1)`
    dimensions and derived up to :math:`\\lfloor\\nu/2\\rfloor` times.
    
    Reference: Rasmussen and Williams (2006, p. 89).
    """
    if _patch_jax.isconcrete(nu):
        assert 0 <= nu < jnp.inf, nu
    r2 = r2 * (2 + nu / 2) ** 2
    return special.gamma(nu + 1) * _patch_jax.jvmodx2(nu, r2)

    # nu >= (D-2)/2
    # 2 nu >= D - 2
    # 2 nu + 2 >= D
    # D <= 2 (nu + 1)
    
@stationarykernel(maxdim=1, derivable=1, input='hard')
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

    l = _patch_jax.ci(delta)
    r = _patch_jax.ci(delta * (1 + dw))
    mean = delta * (1 + dw / 2)
    norm = jnp.log1p(dw)
    tol = jnp.sqrt(jnp.finfo(jnp.empty(0).dtype).eps)
    
    # TODO choose better this tolerance by estimating error terms
    
    return jnp.where(delta * dw < tol, jnp.cos(mean), (r - l) / norm)

def _color_derivable(n=2):
    return n // 2 - 1

@stationarykernel(maxdim=1, derivable=_color_derivable, input='hard')
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
    # 683.f impl of E_n(z) for complex z (license?)
    
    assert int(n) == n and n >= 2, n
    return (n - 1) * _patch_jax.expn_imag(n, delta).real
    
@stationarykernel(forcekron=True, derivable=True, input='soft')
def Sinc(delta):
    """
    Sinc kernel.
    
    .. math:: k(\\Delta) = \\operatorname{sinc}(\\Delta) =
        \\frac{\\sin(\\pi\\Delta)}{\\pi\\Delta}.
    
    Reference: Tobar (2019).
    """
    return _patch_jax.sinc(delta)
    
    # TODO is this isotropic? My current guess is that it works up to some
    # dimension due to a coincidence but is not in general isotropic.

def _stationaryfracbrownian_derivable(H=1/2):
    return H == 1

@stationarykernel(forcekron=True, derivable=_stationaryfracbrownian_derivable, input='signed')
def StationaryFracBrownian(delta, H=1/2):
    """
    Stationary fractional brownian motion kernel.
    
    .. math::
        k(\\Delta) = \\frac 12 (|\\Delta+1|^{2H} + |\\Delta-1|^{2H} - 2|\\Delta|^{2H}),
        \\quad H \\in (0, 1]
        
    Reference: Gneiting and Schlather (2006, p. 272).
    """
    
    # TODO older reference, see [29] is GS06.
    
    if _patch_jax.isconcrete(H):
        assert 0 < H <= 1, H
    H2 = 2 * H
    return 1/2 * (jnp.abs(delta + 1) ** H2 + jnp.abs(delta - 1) ** H2 - 2 * jnp.abs(delta) ** H2)
    
    # TODO is the bifractional version of this valid?

def _cauchy_derivable(alpha=2, **_):
    return alpha == 2

@isotropickernel(derivable=_cauchy_derivable)
def Cauchy(r2, alpha=2, beta=2):
    """
    Generalized Cauchy kernel.
    
    .. math::
        k(r) = \\left(1 + \\frac{r^\\alpha}{\\beta} \\right)^{-\\beta/\\alpha},
        \\quad \\alpha \\in (0, 2], \\beta > 0.
    
    In the geostatistics literature, the case :math:`\\alpha=2` and
    :math:`\\beta=2` (default) is known as the Cauchy kernel. In the machine
    learning literature, the case :math:`\\alpha=2` (for any :math:`\\beta`) is
    known as the rational quadratic kernel. For :math:`\\beta\\to\\infty` it is
    equivalent to ``GammaExp(gamma=alpha, scale=alpha ** (1/alpha))``, while
    for :math:`\\beta\\to 0` to ``Constant``. It is smooth only for
    :math:`\\alpha=2`.
    
    References: Gneiting and Schlather (2004, p. 273), Rasmussen and Williams
    (2006, p. 86).
    
    """
    if _patch_jax.isconcrete(alpha, beta):
        assert 0 < alpha <= 2, alpha
        assert 0 < beta, beta
    return (1 + r2 ** (alpha / 2) / beta) ** (-beta / alpha)
    
def _causalexpquad_derivable(alpha=1):
    return alpha == 0

@isotropickernel(derivable=_causalexpquad_derivable, input='soft')
def CausalExpQuad(r, alpha=1):
    """
    Causal exponential quadratic kernel.
    
    .. math::
        k(r) = \\big(1 - \\operatorname{erf}(\\alpha r/4)\\big)
        \\exp\\left(-\\frac12 r^2 \\right)
        
    From https://github.com/wesselb/mlkernels.
    """
    if _patch_jax.isconcrete(alpha):
        assert alpha >= 0, alpha
    return jspecial.erfc(alpha / 4 * r) * jnp.exp(-1/2 * jnp.square(r))
    # TODO taylor-expand erfc near 0 and use r2

@kernel(derivable=True, maxdim=1)
def Decaying(x, y):
    """
    Decaying kernel.
    
    .. math::
        k(x, y) =
        \\frac{1}{1 + x + y},
        \\quad x, y \\ge 0
    
    It is infinitely divisible.
    
    Reference: Swersky, Snoek and Adams (2014).
    """
    # TODO high dimensional version of this, see mlkernels issue #3
    if _patch_jax.isconcrete(x, y):
        X, Y = _patch_jax.concrete(x, y)
        assert numpy.all(X >= 0)
        assert numpy.all(Y >= 0)
    return 1 / (x + y + 1)
    # use x + y + 1 instead of 1 + x + y because the latter is less numerically
    # symmetric for small x and y
    
    # TODO infinite divisibility checking system like maxdim and derivable

@isotropickernel(derivable=False, input='soft')
def Log(r):
    """
    Log kernel.
    
    .. math::
        k(r) = \\log(1 + r) / r
    
    From https://github.com/wesselb/mlkernels.
    """
    return jnp.log1p(r) / r

# adapted from the "GP-circular" example in the PyMC documentation

# TODO maxdim actually makes sense only for isotropic. I need a way to say
# structured/non structured. Maybe all this should just live in the test suite.

# TODO Any stationary kernel supported on [0, pi] would be fine combined with
# the geodesic distance. Use the generic wendland kernel. Options:
# 1) add here the parameters of Wendland
# 2) add a distance option in stationary to use the angular distance, then
#    let the user apply it to wendland => the problem is that the user would
#    need to be careful with positiveness, while wendland checks it for him

@stationarykernel(derivable=1, maxdim=1, input='soft')
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
    if _patch_jax.isconcrete(tau, c):
        assert tau >= 4, tau
        assert 0 < c <= 1/2, c
    x = delta % 1
    t = jnp.minimum(x, 1 - x)
    return (1 + tau * t / c) * jnp.maximum(1 - t / c, 0) ** tau

# use positive delta because negative indices wrap around
@stationarykernel(derivable=False, maxdim=1, input='hard')
def MA(delta, w=None):
    """
    Discrete moving average kernel.
    
    .. math::
        k(\\Delta) = \\sum_{k=|\\Delta|}^{n-1} w_k w_{k-|\\Delta|},
        \\quad \\mathbf w = (w_0, \\ldots, w_{n-1}).
    
    The inputs must be integers. It is the autocovariance function of a moving
    average with weights :math:`\\mathbf w` applied to white noise:
    
    .. math::
        k(i, j) &= \\operatorname{Cov}[y_i, y_j], \\\\
        y_i &= \\sum_{k=0}^{n-1} w_k \\epsilon_{i-k}, \\\\
        \\operatorname{Cov}[\\epsilon_i,\\epsilon_j] &= \\delta_{ij}.
    
    """
    
    # TODO reference? must find some standard book with a treatment which is
    # not too formal yet writes clearly about the covariance function
    
    # TODO nd version with w.ndim == n, it's a nd convolution
    
    w = jnp.asarray(w)
    assert w.ndim == 1
    if len(w):
        corr = jnp.convolve(w, w[::-1])
        return corr.at[delta + len(w) - 1].get(mode='fill', fill_value=0)
    else:
        return jnp.zeros(delta.shape)

@stationarykernel(derivable=False, maxdim=1, input='hard')
def _ARBase(delta, phi=None, gamma=None, maxlag=None, slnr=None, lnc=None, norm=False):
    """
    Discrete autoregressive kernel.
        
    You have to specify one and only one of the sets of parameters
    `phi+maxlag`, `gamma+maxlag`, `slnr+lnc`.

    Parameters
    ----------
    phi : (p,) real
        The autoregressive coefficients at lag 1...p.
    gamma : (p + 1,) real
        The autocovariance at lag 0...p.
    maxlag : int
        The maximum lag that the kernel will be evaluated on. If the actual
        inputs produce higher lags, the missing values are filled with ``nan``.
    slnr : (nr,) real
        The real roots of the characteristic polynomial, expressed in the
        following way: ``sign(slnr)`` is the sign of the root, and
        ``abs(snlr)`` is the natural logarithm of the absolute value.
    lnc : (nc,) complex
        The natural logarithm of the complex roots of the characteristic
        polynomial (:math:`\\log z = \\log|z| + i\\arg z`), where each root
        also stands for its paired conjugate.
    
        In `slnr` and `lnc`, the multiplicity of a root is expressed by
        repeating the root in the array (not necessarily next to each other).
        Only exact repetition counts; very close yet distinct roots are treated
        as separate and lead to numerical instability, in particular complex
        roots very close to the real line. An exactly real complex root behaves
        like a pair of identical real roots. Two complex roots also count as
        equal if conjugate, and the argument is standardized to :math:`[0,
        2\\pi)`.
    norm : bool
        If True, normalize the autocovariance to be 1 at lag 0. If False
        (default), normalize such that the variance of the generating noise is
        1, or use the user-provided normalization if `gamma` is specified.
    
    Notes
    -----
    This is the covariance function of a stationary autoregressive process,
    which is defined recursively as
    
    .. math::
        y_i = \\sum_{k=1}^p \\phi_k y_{i-k} + \\epsilon_i,
    
    where :math:`\\epsilon_i` is white noise, i.e.,
    :math:`\\operatorname{Cov}[\\epsilon_i, \\epsilon_j] = \\delta_{ij}`. The
    length :math:`p` of the vector of coefficients :math:`\\boldsymbol\\phi`
    is the `order` of the process.
    
    The covariance function can be expressed in two ways. First as the same
    recursion defining the process:
    
    .. math::
        \\gamma_m = \\sum_{k=1}^p \\phi_k \\gamma_{m-k} + \\delta_{m0},
    
    where :math:`\\gamma_m \\equiv \\operatorname{Cov}[y_i, y_{i+m}]`. This is
    called `Yule-Walker equation`. Second, as a linear combination of mixed
    power-exponentials:
    
    .. math::
        \\gamma_m = \\sum_{j=1}^n
                    \\sum_{l=1}^{\\mu_j}
                    a_{jl} |m|^{l-1} x_j^{-|m|},
    
    where :math:`x_j` and :math:`\\mu_j` are the (complex) roots and
    corresponding multiplicities of the `characteristic polynomial`
    
    .. math::
        P(x) = 1 - \\sum_{k=1}^p \\phi_k x^k,
    
    and the :math:`a_{jl}` are uniquely determined complex coefficients. The
    :math:`\\boldsymbol\\phi` vector is valid iff :math:`|x_j|>1, \\forall j`.
    
    There are three alternative parametrization for this kernel.
    
    If you specify `phi`, the first terms of the covariance are computed
    solving the Yule-Walker equation, and then evolved up to `maxlag`. It
    is necessary to specify `maxlag` instead of letting the code figure it out
    from the actual inputs for technical reasons.
    
    Likewise, if you specify `gamma`, the coefficients are obtained with
    Yule-Walker and then used to evolve the covariance. The only difference is
    that the normalization can be different: starting from `phi`, the variance
    of the generating noise :math:`\\epsilon` is fixed to 1, while giving
    `gamma` directly implies an arbitrary value.
    
    Instead, if you specify the roots with `slnr` and `lnc`, the coefficients
    are obtained from the polynomial defined in terms of the roots, and then
    the amplitudes :math:`a_{jl}` are computed by solving a linear system with
    the covariance (from YW) as RHS. Finally, the full covariance function is
    evaluated with the analytical expression.
    
    The reasons for using the logarithm are that 1) in practice the roots are
    tipically close to 1, so the logarithm is numerically more accurate, and 2)
    the logarithm is readily interpretable as the inverse of the correlation
    length.
    
    """
    cond = (
        (phi is not None and maxlag is not None and gamma is None and slnr is None and lnc is None) or
        (phi is None and maxlag is not None and gamma is not None and slnr is None and lnc is None) or
        (phi is None and maxlag is None and gamma is None and slnr is not None and lnc is not None)
    )
    if not cond:
        raise ValueError('invalid set of specified parameters')
    
    # TODO maybe I could allow redundantly specifying gamma and phi, e.g., for
    # numerical accuracy reasons if they are determined from an analytical
    # expression.
    
    if phi is None and gamma is None:
        return _ar_with_roots(delta, slnr, lnc, norm)
    else:
        return _ar_with_phigamma(delta, phi, gamma, maxlag, norm)

def _ar_with_phigamma(delta, phi, gamma, maxlag, norm):
    if phi is None:
        phi = AR.phi_from_gamma(gamma)
    if gamma is None:
        gamma = AR.gamma_from_phi(phi)
    if norm:
        gamma = gamma / gamma[0]
    acf = AR.extend_gamma(gamma, phi, maxlag + 1 - len(gamma))
    return acf.at[delta].get(mode='fill', fill_value=jnp.nan)

def _yule_walker(gamma):
    """
    gamma = autocovariance at lag 0...p
    output: autoregressive coefficients at lag 1...p
    """
    gamma = jnp.asarray(gamma)
    assert gamma.ndim == 1
    t = gamma[:-1]
    b = gamma[1:]
    if t.size:
        return _linalg._toeplitz.solve(t, b)
    else:
        return jnp.empty(0)

def _yule_walker_inv_mat(phi):
    phi = jnp.asarray(phi)
    assert phi.ndim == 1
    p = len(phi)
    m = jnp.arange(p + 1)[:, None] # rows
    n = m.T # columns
    phi = jnp.pad(phi, (1, 1))
    kp = jnp.clip(m + n, 0, p + 1)
    km = jnp.clip(m - n, 0, p + 1)
    return jnp.eye(p + 1) - (phi[kp] + phi[km]) / jnp.where(n, 1, 2)
    
def _yule_walker_inv(phi):
    """
    phi = autoregressive coefficients at lag 1...p
    output: autocovariance at lag 0...p, assuming driving noise has sdev 1
    """
    a = _yule_walker_inv_mat(phi)
    b = jnp.zeros(len(a)).at[0].set(1)
    # gamma = _pseudo_solve(a, b)
    gamma = jnp.linalg.solve(a, b)
    return gamma

def _ar_evolve(phi, start, noise):
    """
    phi = autoregressive coefficients at lag 1...p
    start = first p values of the process (increasing time)
    noise = n noise values added at each step
    output: n new process values
    """
    phi = jnp.asarray(phi)
    start = jnp.asarray(start)
    noise = jnp.asarray(noise)
    assert phi.ndim == 1 and phi.shape == start.shape and noise.ndim == 1
    return _ar_evolve_jit(phi, start, noise)

@jax.jit
def _ar_evolve_jit(phi, start, noise):
    
    def f(carry, eps):
        vals, cc, roll = carry
        phi = lax.dynamic_slice(cc, [vals.size - roll], [vals.size])
        nextval = phi @ vals + eps
        if vals.size:
            vals = vals.at[roll].set(nextval)
        # maybe for some weird reason like alignment, actual rolling would
        # be faster. whatever
        roll = (roll + 1) % vals.size
        return (vals, cc, roll), nextval
    
    cc = jnp.concatenate([phi, phi])[::-1]
    _, ev = lax.scan(f, (start, cc, 0), noise, unroll=16)
    return ev

def _ar_with_roots(delta, slnr, lnc, norm):
    slnr = jnp.asarray(slnr).astype(float).sort()
    lnc = jnp.asarray(lnc).astype(complex)
    imag = jnp.abs(lnc.imag) % (2 * jnp.pi)
    imag = jnp.where(imag > jnp.pi, 2 * jnp.pi - imag, imag)
    lnc = lnc.real + 1j * imag
    lnc = lnc.sort()
    phi = AR.phi_from_roots(slnr, lnc) # <---- weak
    gamma = AR.gamma_from_phi(phi) # <---- point
    
    # TODO Currently gamma is not even pos def for high multiplicity/roots close
    # to 1. Raw patch: make a function _fix_bad_gamma. The badness condition is
    # gamma[0] < 0 or any(abs(gamma) > gamma[0]). The fix takes the smallest
    # log|root| and assumes it alone determines gamma. Question: if then I
    # iterate with the solution found, does it converge to the correct solution?
    # => no, it just gives 0 amplitude to other roots. Maybe I should determine
    # some necessary constraints on the amplitudes for positivity.
    
    p = phi.size

    if norm:
        gamma /= gamma[0]
    lag = jnp.arange(p + 1)
    mat = _gamma_from_ampl_matmul(slnr, lnc, lag, jnp.eye(1 + p))
    ampl = jnp.linalg.solve(mat, gamma)
    lag = delta if delta.ndim else delta[None]
    acf = _gamma_from_ampl_matmul(slnr, lnc, lag, ampl)
    return acf if delta.ndim else acf.squeeze(0)

    # yw = _yule_walker_inv_mat(phi)
    # b = jnp.zeros(p + 1).at[0].set(1)
    # ampl = jnp.linalg.solve(yw @ mat, b)
    # lag = delta if delta.ndim else delta[None]
    # acf = _gamma_from_ampl_matmul(slnr, lnc, lag, ampl)
    # if norm:
    #     acf0 = _gamma_from_ampl_matmul(slnr, lnc, jnp.array([0]), ampl)
    #     acf /= acf0
    # return acf if delta.ndim else acf.squeeze(0)

# def _pseudo_solve(a, b):
#     # this is less accurate than jnp.linalg.solve
#     u, s, vh = jnp.linalg.svd(a)
#     eps = jnp.finfo(a.dtype).eps
#     s0 = s[0] if s.size else 0
#     invs = jnp.where(s < s0 * eps * len(a), 0, 1 / s)
#     return jnp.einsum('ij,j,jk,k', vh.conj().T, invs, u.conj().T, b)

@jax.jit
def _gamma_from_ampl_matmul(slnr, lnc, lag, ampl, lagnorm=None):

    vec = ampl.ndim == 1
    if vec:
        ampl = ampl[:, None]
    p = slnr.size + 2 * lnc.size
    assert ampl.shape[-2] == p + 1
    if lagnorm is None:
        lagnorm = p
    
    def logcol(root, lag, llag, repeat):
        return -root * lag + jnp.where(repeat, repeat * llag, 0)
    
    def lognorm(root, repeat, lagnorm):
        maxnorm = jnp.where(repeat, repeat * (-1 + jnp.log(repeat / root)), 0)
        defnorm = logcol(root, lagnorm, jnp.log(lagnorm), repeat)
        maxloc = repeat / root
        return jnp.where(maxloc <= lagnorm, maxnorm, defnorm)
    
    # roots at infinity
    col = jnp.where(lag, 0, 1)
    out = col[..., :, None] * ampl[..., 0, :]
    
    # real roots
    llag = jnp.log(lag)
    val = (jnp.nan, 0, out, slnr, lag, llag, lagnorm)
    def loop(i, val):
        prevroot, repeat, out, slnr, lag, llag, lagnorm = val
        root = slnr[i]
        repeat = jnp.where(root == prevroot, repeat + 1, 0)
        prevroot = root
        sign = jnp.sign(root) ** lag
        aroot = jnp.abs(root)
        lcol = logcol(aroot, lag, llag, repeat)
        norm = lognorm(aroot, repeat, lagnorm)
        col = sign * jnp.exp(lcol - norm)
        out += col[..., :, None] * ampl[..., 1 + i, :]
        return prevroot, repeat, out, slnr, lag, llag, lagnorm
    if slnr.size:
        _, _, out, _, _, _, _ = lax.fori_loop(0, slnr.size, loop, val)
    
    # complex roots
    val = (jnp.nan, 0, out, lnc, lag, llag, lagnorm)
    def loop(i, val):
        prevroot, repeat, out, lnc, lag, llag, lagnorm = val
        root = lnc[i]
        repeat = jnp.where(root == prevroot, repeat + 1, 0)
        prevroot = root
        lcol = logcol(root, lag, llag, repeat)
        norm = lognorm(root.real, repeat, lagnorm)
        col = jnp.exp(lcol - norm)
        idx = 1 + slnr.size + 2 * i
        out += col.real[..., :, None] * ampl[..., idx, :]
        
        # real complex root = a pair of identical real roots
        repeat = jnp.where(root.imag, repeat, repeat + 1)
        col1 = jnp.where(root.imag, -col.imag, col.real * lag)
        out += col1[..., :, None] * ampl[..., idx + 1, :]
        
        return prevroot, repeat, out, lnc, lag, llag, lagnorm
    if lnc.size:
        _, _, out, _, _, _, _ = lax.fori_loop(0, lnc.size, loop, val)
    
    if vec:
        out = out.squeeze(-1)
    
    return out

class AR(_ARBase):
    
    __doc__ = _ARBase.__doc__
    
    @staticmethod
    def phi_from_gamma(gamma):
        """
        Determine the autoregressive coefficients from the covariance.
        
        Parameters
        ----------
        gamma : (p + 1,) array
            The autocovariance at lag 0...p.
        
        Returns
        -------
        phi : (p,) array
            The autoregressive coefficients at lag 1...p.
        """
        return _yule_walker(gamma)
    
    @staticmethod
    def gamma_from_phi(phi):
        """
        Determine the covariance from the autoregressive coefficients.

        Parameters
        ----------
        phi : (p,) array
            The autoregressive coefficients at lag 1...p.
        
        Returns
        -------
        gamma : (p + 1,) array
            The autocovariance at lag 0...p. The normalization is
            with noise variance 1.
        
        Notes
        -----
        The result is wildly inaccurate for roots with high multiplicity and/or
        close to 1.
        """
        return _yule_walker_inv(phi)
    
    @staticmethod
    def extend_gamma(gamma, phi, n):
        """
        Extends values of the covariance function to higher lags.
        
        Parameters
        ----------
        gamma : (m,) array
            The autocovariance at lag q-m+1...q, with q >= 0.
        phi : (p,) array
            The autoregressive coefficients at lag 1...p.
        n : int
            The number of new values to generate.
        
        Returns
        -------
        ext : (m + n,) array
            The autocovariance at lag q-m+1...q+n.
        """
        gamma = jnp.asarray(gamma)
        ext = _ar_evolve(phi, gamma[len(gamma) - len(phi):], jnp.broadcast_to(0., (n,)))
        return jnp.concatenate([gamma, ext])
    
    @staticmethod
    def phi_from_roots(slnr, lnc):
        """
        Determine the autoregressive coefficients from the roots of the
        characteristic polynomial.
        
        Parameters
        ----------
        slnr : (nr,) real
            The real roots of the characteristic polynomial, expressed in the
            following way: ``sign(slnr)`` is the sign of the root, and
            ``abs(snlr)`` is the natural logarithm of the absolute value.
        lnc : (nc,) complex
            The natural logarithm of the complex roots of the characteristic
            polynomial (:math:`\\log z = \\log|z| + i\\arg z`), where each root
            also stands for its paired conjugate.
        
        Returns
        -------
        phi : (p,) real
            The autoregressive coefficients at lag 1...p, with p = nr + 2 nc.
        """
        slnr = jnp.asarray(slnr)
        lnc = jnp.asarray(lnc)
        assert slnr.ndim == lnc.ndim == 1
        r = jnp.sign(slnr) * jnp.exp(jnp.abs(slnr))
        c = jnp.exp(lnc)
        roots = jnp.concatenate([r, c, jnp.conj(c)])
        coef = numpy.polynomial.polynomial.polyfromroots(roots)
        # TODO port polyfromroots to jax
        numpy.testing.assert_allclose(coef.imag, 0, rtol=0, atol=1e-4)
        coef = coef.real
        return -coef[1:] / coef[0]

def _bart_maxdim(splits=None, **_):
    splits = _check_splits(splits)
    return splits[0].size

# TODO consider letting derivable=True and returning Zero

@kernel(maxdim=_bart_maxdim, derivable=False)
def _BARTBase(x, y, alpha=0.95, beta=2, maxd=2, splits=None):
    """
    BART kernel.
    
    Parameters
    ----------
    alpha, beta : scalar
        The parameters of the branching probability.
    maxd : int
        The maximum depth of the trees.
    splits : pair of arrays
        The first is an int (p,) array containing the number of splitting
        points along each dimension, the second has shape (n, p) and contains
        the sorted splitting points in each column, filled with high values
        after the length.
    
    Methods
    -------
    splits_from_coord
    indices_from_coord
    
    Notes
    -----
    This is the covariance function of the latent mean prior of BART (Bayesian
    Additive Regression Trees) [1]_ in the limit of an infinite number of
    trees, and with an upper bound on the depth of the trees. This prior is the
    distribution of the function
    
    .. math::
        f(\\mathbf x) = \\lim_{m\\to\\infty}
        \\sum_{j=1}^m g(\\mathbf x; T_j, M_j),
    
    where each :math:`g(\\mathbf x; T_j, M_j)` is a decision tree evaluated at
    :math:`\\mathbf x`, with structure :math:`T_j` and leaf values :math:`M_j`.
    The trees are i.i.d., with the following distribution for :math:`T_j`: for
    a node at depth :math:`d`, with :math:`d = 0` for the root, the probability
    of not being a leaf, conditional on its existence and its ancestors only, is
    
    .. math::
        P_d = \\alpha (1+d)^{-\\beta}, \\quad
        \\alpha \\in [0, 1], \\quad \\beta \\ge 0.
    
    For a non-leaf node, conditional on existence and ancestors, the splitting
    variable has uniform distribution amongst the variables with any splitting
    points not used by ancestors, and the splitting point has uniform
    distribution amongst the available ones. The splitting points are fixed,
    tipically from the data.
    
    The distribution of leaves :math:`M_j` is i.i.d. Normal with variance
    :math:`1/m`, such that :math:`f(x)` has variance 1. In the limit
    :math:`m\\to\\infty`, the distribution of :math:`f(x)` becomes a Gaussian
    process.
    
    Since the trees are independent, the covariance function can be computed
    for a single tree. Consider two coordinates :math:`x` and :math:`y`, with
    :math:`x \\le y`. Let :math:`n^-`, :math:`n^0` and :math:`n^+` be the
    number of splitting points respectively before :math:`x`, between
    :math:`x`, :math:`y` and after :math:`y`. Next, define :math:`\\mathbf
    n^-`, :math:`\\mathbf n^0` and :math:`\\mathbf n^+` as the vectors of such
    quantities for each dimension, with a total of :math:`p` dimensions. Then
    the covariance function can be written recursively as
    
    .. math::
        \\newcommand{\\nvecs}{\\mathbf n^-, \\mathbf n^0, \\mathbf n^+}
        k(\\mathbf x, \\mathbf y) &= k_0(\\nvecs), \\\\
        k_D(\\nvecs) &= 1, \\\\
        k_d(\\nvecs) &= 1 - P_d \\Bigg(1 - \\frac1p
            \\sum_{i=1}^p \\frac1{n^-_i + n^0_i + n^+_i} \\Bigg( \\\\
                &\\qquad \\sum_{k=0}^{n^-_i - 1}
                k_{d+1}(\\mathbf n^-_{n^-_i=k}, \\mathbf n^0, \\mathbf n^+)
                + {} \\\\
                &\\qquad \\sum_{k=0}^{n^+_i - 1}
                k_{d+1}(\\mathbf n^-, \\mathbf n^0, \\mathbf n^+_{n^+_i=k})
            \\Bigg)
        \\Bigg), \\quad d < D,
    
    where it is intended that when :math:`n^-_i = n^0_i = n^+_i = 0`, the term
    of the summation yields 0.
    
    The introduction of a maximum depth :math:`D` is necessary for
    computational feasibility. As :math:`D` increases, the result converges to
    the one without depth limit. In setting :math:`D`, consider that
    :math:`\\beta` regulates the distribution of the depth, with high values
    strongly favoring shallow trees.
    
    In the maximum allowed depth is 1, i.e., either :math:`D = 1` or
    :math:`\\beta\\to\\infty`, the kernel assumes the simple form
    
    .. math::
        k(\\mathbf x, \\mathbf y) = 1 - \\frac\\alpha p \\sum_{i=1}^p
        \\frac{n^0_i}{n^-_i + n^0_i + n^+_i},
    
    which is separable along dimensions.
    
    References
    ----------
    .. [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch "BART:
        Bayesian additive regression trees," The Annals of Applied Statistics,
        Ann. Appl. Stat. 4(1), 266-298, (March 2010).
    """
    if _patch_jax.isconcrete(alpha, beta):
        assert 0 <= alpha <= 1, alpha
        assert beta >= 0, beta
    assert maxd == int(maxd) and maxd >= 0, maxd
    splits = _check_splits(splits)
    if not x.dtype.names:
        x = x[..., None]
    if not y.dtype.names:
        y = y[..., None]
    ix = BART.indices_from_coord(x, splits)
    iy = BART.indices_from_coord(y, splits)
    l = jnp.minimum(ix, iy)
    r = jnp.maximum(ix, iy)
    before = l
    between = r - l
    after = splits[0] - r
    return bart_correlation_maxd(before, between, after, alpha, beta, maxd)
    
    # TODO options to:
    # - use lower/upper bound (is lower pos def?)
    # - interpolate (is linear interp pos def?)
    # - approximate as stationary w.r.t. indices (is it pos def?)
    # - allow index input

class BART(_BARTBase):
    
    __doc__ = _BARTBase.__doc__
    
    @staticmethod
    def splits_from_coord(x):
        """
        Generate splitting points from data.
        
        Parameters
        ----------
        x : array of numbers
            The data. Can be passed in two formats: 1) a structured array where
            each leaf field represents a dimension, 2) a normal array where the
            last axis runs over dimensions. In the structured case, each
            index in any shaped field is a different dimension.
        
        Returns
        -------
        length : int (p,) array
            The number of splitting points along each of `p` dimensions.
        splits : (n, p) array
            Each column contains the sorted splitting points along a dimension.
            The splitting points are the midpoints between consecutive
            coordinates appearing in `x` for that dimension. Column
            ``splits[:, i]`` contains splitting points only up to ``length[i]``,
            while afterward it is filled with a very large value.
        
        """
        x = _check_x(x)
        x = x.reshape(-1, x.shape[-1]) if x.size else x.reshape(1, x.shape[-1])
        x = jnp.sort(x, axis=0)
        midpoints = (x[:-1, :] + x[1:, :]) / 2
        return _unique_vectorized(midpoints)
    
    @staticmethod
    def indices_from_coord(x, splits):
        """
        Convert coordinates to indices w.r.t. splitting points.
        
        Parameters
        ----------
        x : array of numbers
            The coordinates. Can be passed in two formats: 1) a structured
            array where each leaf field represents a dimension, 2) a normal
            array where the last axis runs over dimensions. In the structured
            case, each index in any shaped field is a different dimension.
        splits : pair of arrays
            The first is an int (p,) array containing the number of splitting
            points along each dimension, the second has shape (n, p) and
            contains the sorted splitting points in each column, filled with
            high values after the length.
        
        Returns
        -------
        ix : int array
            An array with the same shape as `x`, unless `x` is a structured
            array, in which case the last axis of `ix` is the flattened version
            of the structured type. `ix` contains indices mapping `x` to
            positions between splitting points along each coordinate, with the
            following convention: index 0 means before the first split, index
            i > 0 means between split i - 1 and split i.
        
        """
        x = _check_x(x)
        splits = _check_splits(splits)
        assert x.shape[-1] == splits[0].size
        return _searchsorted_vectorized(splits[1], x)

def _check_x(x):
    x = _array.asarray(x)
    if x.dtype.names:
        x = numpy.structured_to_unstructured(x)
    return x

def _check_splits(splits):
    l, s = splits
    l = jnp.asarray(l)
    s = jnp.asarray(s)
    assert l.ndim == 1 and 1 <= s.ndim <= 2
    if s.ndim == 1:
        s = s[:, None]
    assert l.size == s.shape[1]
    return l, s

@jax.jit
def _unique_vectorized(X):
    """
    X : (..., p)
    length : int (p,)
    unique : (n, p)
    """
    if jnp.issubdtype(X.dtype, jnp.inexact):
        info = jnp.finfo
    else:
        info = jnp.iinfo
    fill = info(X.dtype).max
    def loop(_, x):
        u = jnp.unique(x, size=x.size, fill_value=fill)
        l = jnp.searchsorted(u, fill)
        return _, (l, u)
    _, (length, unique) = lax.scan(loop, None, X.T)
    return length, unique.T

@functools.partial(jax.jit, static_argnames=('side',))
def _searchsorted_vectorized(A, V, **kw):
    """
    A : (n, p)
    V : (..., p)
    out : (..., p)
    """
    def loop(_, av):
        return _, jnp.searchsorted(*av, **kw)
    _, out = lax.scan(loop, None, (A.T, V.T))
    return out.T

@functools.partial(jax.jit, static_argnums=(5, 6, 7))
def _bart_correlation_maxd_recursive(nminus, n0, nplus, alpha, beta, maxd, d, upper):
    
    if d >= maxd and upper:
        return 1.
    
    pnt = alpha / (1 + d) ** beta
    
    if d >= maxd and not upper:
        return 1 - pnt
        
    p = len(nminus)

    if d + 1 >= maxd:
        nout = nminus + nplus
        terms = nout / (nout + n0)
        if not upper:
            pnt1 = alpha / (2 + d) ** beta
            terms *= pnt1
        sump = jnp.sum(jnp.where(n0, terms, 1))
        return 1 - pnt * (1 - sump / p).astype(float)
    
    # TODO hardcode d + 2 case to something O(np)
    #
    # the sum I need is sum_k=0^n-1 1/(a+k), this is zeta(1, a) - zeta(1, a+n)
    # (https://dlmf.nist.gov/25.11.E4) however it's inf at 1.
    #
    # since a is an integer, it can be expressed as a difference of harmonic
    # numbers, which can be calculated from the digamma function
    # (https://dlmf.nist.gov/5.4.E14).
    
    val = (0., nminus, n0, nplus)
    def loop(i, val):
        sump, nminus, n0, nplus = val

        nminusi = nminus[i]
        n0i = n0[i]
        nplusi = nplus[i]
        
        val = (0., nminus, n0, nplus, i, nminusi)
        def loop(k, val):
            sumn, nminus, n0, nplus, i, nminusi = val
            
            nminus = nminus.at[jnp.where(k < nminusi, i, i + p)].set(k)
            nplus = nplus.at[jnp.where(k >= nminusi, i, i + p)].set(k - nminusi)
            
            sumn += _bart_correlation_maxd_recursive(nminus, n0, nplus, alpha, beta, maxd, d + 1, upper)
            
            nminus = nminus.at[i].set(nminusi)
            nplus = nplus.at[i].set(nplusi)
            
            return sumn, nminus, n0, nplus, i, nminusi
        
        sumn, nminus, n0, nplus, _, _ = lax.fori_loop(jnp.int32(0), nminusi + nplusi, loop, val)

        sump += jnp.where(n0i, sumn / (nminusi + n0i + nplusi), 1)

        return sump, nminus, n0, nplus

    sump, _, _, _ = lax.fori_loop(0, p, loop, val)

    return 1 - pnt * (1 - sump / p)
    
_bart_correlation_maxd_vectorized = jax.vmap(_bart_correlation_maxd_recursive, [
    0, 0, 0, None, None, None, None, None,
])
    
def bart_correlation_maxd(splitsbefore, splitsbetween, splitsafter, alpha, beta, maxd, upper=True):
    """
    Compute the BART prior correlation between two points.
    
    The correlation is computed approximately by limiting the maximum depth
    of the trees. Limiting trees to depth 1 is equivalent to setting beta to
    infinity.
        
    Parameters
    ----------
    splitsbefore : int (..., p) array
        The number of splitting points less than the two points, separately
        along each coordinate.
    splitsbetween : int (..., p) array
        The number of splitting points between the two points, separately along
        each coordinate.
    splitsafter : int (..., p) array
        The number of splitting points greater than the two points, separately
        along each coordinate.
    alpha, beta : scalar
        The hyperparameters of the branching probability.
    maxd : int
        The maximum depth of the trees. The root has depth zero.
    upper : bool
        If True (default), the result is an upper bound on the limit of
        infinite maxd. If False, it is a lower bound.
    
    Return
    ------
    corr : scalar
        The prior correlation.
    """
    args = [splitsbefore, splitsbetween, splitsafter]
    args = [jnp.asarray(a, 'i4') for a in args]
    for a in args:
        assert a.ndim >= 1
    p = args[0].shape[-1]
    for a in args:
        assert a.shape[-1] == p
    args = jnp.broadcast_arrays(*args)
    shape = args[0].shape[:-1]
    args = [a.reshape(-1, p) for a in args]
    alpha = jnp.array(alpha, float)
    beta = jnp.array(beta, float)
    out = _bart_correlation_maxd_vectorized(*args, alpha, beta, int(maxd), 0, bool(upper))
    return out.reshape(shape)

