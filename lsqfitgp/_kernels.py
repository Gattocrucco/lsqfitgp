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

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy import special as jspecial
from scipy import special

from . import _array
from . import _Kernel
from . import _linalg
from . import _patch_jax
from ._Kernel import kernel, stationarykernel, isotropickernel

__all__ = [
    'Constant',
    'White',
    'ExpQuad',
    'Linear',
    'Matern',
    'Matern12',
    'Matern32',
    'Matern52',
    'GammaExp',
    'RatQuad',
    'NNKernel',
    'Wiener',
    'Gibbs',
    'Periodic',
    'Categorical',
    'Rescaling',
    'Cos',
    'FracBrownian',
    'PPKernel',
    'WienerIntegral',
    'Taylor',
    'Fourier',
    'OrnsteinUhlenbeck',
    'Celerite',
    'BrownianBridge',
    'Harmonic',
    'Expon',
    'BagOfWords',
]

def _dot(x, y):
    return _Kernel._sum_recurse_dtype(lambda x, y: x * y, x, y)
    
@isotropickernel(derivable=True)
def Constant(r2):
    """
    Constant kernel.
    
    .. math::
        k(r) = 1
    
    This means that all points are completely correlated, thus it is equivalent
    to fitting with an horizontal line. This can be seen also by observing that
    1 = 1 x 1.
    """
    return _linalg.choose_numpy(r2).ones(r2.shape)
    # TODO add support for input that doesn't have a distance but
    # just equality, like for example non-numerical input.
    
@isotropickernel(derivable=False)
def White(r2):
    """
    White noise kernel.
    
    .. math::
        k(x, y) = \\begin{cases}
            1 & x = y     \\\\
            0 & x \\neq y
        \\end{cases}
    """
    # TODO add support for input that doesn't have a distance but
    # just equality, like for example non-numerical input.
    return _linalg.choose_numpy(r2).where(r2 == 0, 1, 0)

@isotropickernel(derivable=True)
def ExpQuad(r2):
    """
    Exponential quadratic kernel.
    
    .. math::
        k(r) = \\exp \\left( -\\frac 12 r^2 \\right)
    
    It is smooth and has a strict typical lengthscale, i.e., oscillations are
    strongly suppressed under a certain wavelength, and correlations are
    strongly suppressed over a certain distance.
    """
    return _linalg.choose_numpy(r2).exp(-1/2 * r2)

@kernel(derivable=True)
def Linear(x, y):
    """
    Dot product kernel.
    
    .. math::
        k(x, y) = x \\cdot y = \\sum_i x_i y_i
    
    In 1D it is equivalent to fitting with a line passing by the origin.
    """
    return _dot(x, y)

@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _maternp(x, p):
    poly = 1
    for k in reversed(range(p)):
        c_kp1_over_ck = (p - k) / ((2 * p - k) * (k + 1))
        poly *= c_kp1_over_ck * 2 * x
        poly += 1
    return jnp.exp(-x) * poly

def _maternp_deriv(x, p):
    if p == 0:
        return -jnp.exp(-x)
    poly = 1
    for k in reversed(range(1, p)):
        c_kp1_over_ck = (p - k) / ((2 * p - k - 1) * k)
        poly = 1 + poly * c_kp1_over_ck * 2 * x
    poly = poly / (1 - 2 * p) * x
    return jnp.exp(-x) * poly

@_maternp.defjvp
def _maternp_jvp(p, primals, tangents):
    x, = primals
    xdot, = tangents
    return _maternp(x, p), _maternp_deriv(x, p) * xdot

def _matern_derivable(**kw):
    nu = kw.get('nu', None)
    if np.isscalar(nu) and nu > 0 and (2 * nu) % 1 == 0:
        return int(nu - 1/2)
    else:
        return False

# TODO I'm using soft input for the Matérn kernels, however for the
# half-integer case it is probably not necessary to use the softabs. Add an
# option 'hard' to the IsotropicKernel init and see if it works.

@isotropickernel(input='soft', derivable=_matern_derivable)
def Matern(r, nu=None):
    """
    Matérn kernel of real order. 
    
    .. math::
        k(r) = \\frac {2^{1-\\nu}} {\\Gamma(\\nu)} x^\\nu K_\\nu(x),
        \\quad \\nu = \\texttt{nu} > 0,
        \\quad x = \\sqrt{2\\nu} r
    
    The nearest integer below `nu` indicates how many times the Gaussian
    process is derivable: so for `nu` < 1 it is continuous but not derivable,
    for 1 <= `nu` < 2 it is derivable but has not a decond derivative, etc. The
    half-integer case (nu = 1/2, 3/2, ...) uses internally a simpler formula so
    you should prefer it. Also, taking derivatives of the process is supported
    only for half-integer nu.

    """
    assert 0 < nu < jnp.inf
    x = jnp.sqrt(2 * nu) * r
    if (2 * nu) % 1 == 0:
        return _maternp(x, int(nu - 1/2))
    else:
        return 2 ** (1 - nu) / special.gamma(nu) * x ** nu * _patch_jax.kv(nu, x)

@isotropickernel(input='soft', derivable=False)
def Matern12(r):
    """
    Matérn kernel of order 1/2 (continuous, not derivable).
    
    .. math::
        k(r) = \\exp(-r)
    """
    return jnp.exp(-r)

@jax.custom_jvp
def _matern32(x):
    return (1 + x) * jnp.exp(-x)

_matern32.defjvps(lambda g, ans, x: (g.T * (-x * jnp.exp(-x)).T).T)

@isotropickernel(input='soft', derivable=1)
def Matern32(r):
    """
    Matérn kernel of order 3/2 (derivable one time).
    
    .. math::
        k(r) = (1 + x) \\exp(-x), \\quad x = \\sqrt3 r
    """
    return _matern32(np.sqrt(3) * r)

@jax.custom_jvp
def _matern52(x):
    return (1 + x * (1 + x/3)) * jnp.exp(-x)

_matern52.defjvps(lambda g, ans, x: (g.T * (-x/3 * _matern32(x)).T).T)

@isotropickernel(input='soft', derivable=2)
def Matern52(r):
    """
    Matérn kernel of order 5/2 (derivable two times).
    
    .. math::
        k(r) = (1 + x + x^2/3) \\exp(-x), \\quad x = \\sqrt5 r
    """
    return _matern52(np.sqrt(5) * r)
    
def _gammaexp_derivable(gamma=1):
    return gamma == 2

@isotropickernel(input='soft', derivable=_gammaexp_derivable)
def GammaExp(r, gamma=1):
    """
    Gamma exponential kernel.
    
    .. math::
        k(r) = \\exp(-r^\\texttt{gamma}), \\quad
        \\texttt{gamma} \\in [0, 2]
    
    For `gamma` = 2 it is the Gaussian kernel, for `gamma` = 1 it is the Matérn
    1/2 kernel, for `gamma` = 0 it is the constant kernel. The process is
    differentiable only for `gamma` = 2, however as `gamma` gets closer to 2
    the variance of the non-derivable component goes to zero.

    """
    assert 0 <= gamma <= 2, gamma
    return jnp.exp(-(r ** gamma))

@isotropickernel(derivable=True)
def RatQuad(r2, alpha=2):
    """
    Rational quadratic kernel.
    
    .. math::
        k(r) = \\left( 1 + \\frac {r^2} {2 \\alpha} \\right)^{-\\alpha},
        \\quad \\alpha = \\texttt{alpha}
    
    It is equivalent to a lengthscale mixture of Gaussian kernels where the
    scale distribution is a gamma with shape parameter `alpha`. For `alpha` ->
    infinity, it becomes the Gaussian kernel. It is smooth.
    
    """
    assert 0 < alpha < np.inf, alpha
    return (1 + r2 / (2 * alpha)) ** -alpha

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
    
    """
    
    # TODO the `2`s in the formula are a bit arbitrary. Remove them or give
    # motivation relative to the precise formulation of the neural network.
    
    assert 0 < sigma0 < jnp.inf
    q = sigma0 ** 2
    denom = (1 + 2 * (q + _dot(x, x))) * (1 + 2 * (q + _dot(y, y)))
    return 2/np.pi * jnp.arcsin(2 * (q + _dot(x, y)) / denom)
    
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
    
    """
    assert jnp.all(x >= 0)
    assert jnp.all(y >= 0)
    return jnp.minimum(x, y)

@kernel(forcekron=True)
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
    
    """
    sx = scalefun(x)
    sy = scalefun(y)
    assert jnp.all(_linalg.notracer(sx) > 0)
    assert jnp.all(_linalg.notracer(sy) > 0)
    denom = sx ** 2 + sy ** 2
    factor = jnp.sqrt(2 * sx * sy / denom)
    return factor * jnp.exp(-(x - y) ** 2 / denom)

@stationarykernel(derivable=True, forcekron=True)
def Periodic(delta, outerscale=1):
    """
    Periodic Gaussian kernel.
    
    .. math::
        k(x, y) = \\exp \\left(
        -2 \\left(
        \\frac {\\sin((x - y) / 2)} {\\texttt{outerscale}}
        \\right)^2
        \\right)
    
    A Gaussian kernel over a transformed periodic space. It represents a
    periodic process. The usual `scale` parameter sets the period, with the
    default `scale` = 1 giving a period of 2π, while the `outerscale` parameter
    sets the length scale of the correlations.
    
    """
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
    
    # TODO support an array-like for cov, do not force it to be a numpy
    # array. In particular I'd like to support sparse matrices, but indexing
    # with two arrays is not supported for pydata/sparse. I can circumvent it
    # by flattening cov and converting x and y to flat indices manually. (Make
    # specific tests when I implement this.)
    
    assert jnp.issubdtype(x.dtype, jnp.integer)
    cov = jnp.array(cov, copy=False)
    assert len(cov.shape) == 2
    assert cov.shape[0] == cov.shape[1]
    assert jnp.allclose(cov, cov.T)
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
    return stdfun(x) * stdfun(y)

@stationarykernel(derivable=True, forcekron=True)
def Cos(delta):
    """
    Cosine kernel.
    
    .. math::
        k(x, y) = \\cos(x - y)
        = \\cos x \\cos y + \\sin x \\sin y
    
    Samples from this kernel are harmonic functions. It can be multiplied with
    another kernel to introduce anticorrelations.
    
    """
    return jnp.cos(delta)

@kernel(forcekron=True, derivable=False)
def FracBrownian(x, y, H=1/2):
    """
    Fractional brownian motion kernel.
    
    .. math::
        k(x, y) = \\frac 12 (x^{2H} + y^{2H} - |x-y|^{2H}),
        \\quad H \\in (0, 1), \\quad x, y \\ge 0
    
    For `H` = 1/2 (default) it is the Wiener kernel. For `H` in (0, 1/2) the
    increments are anticorrelated (strong oscillation), for `H` in (1/2, 1)
    the increments are correlated (tends to keep a slope).
    
    """
    
    # TODO I think the correlation between successive same step increments
    # is 2^(2H-1) - 1 in (-1/2, 1). Maybe add this to the docstring.
    
    assert 0 < H < 1
    assert jnp.all(x >= 0)
    assert jnp.all(y >= 0)
    H2 = 2 * H
    return 1/2 * (x ** H2 + y ** H2 - jnp.abs(x - y) ** H2)

def _ppkernel_derivable(**kw):
    return kw.get('q', 0)

@isotropickernel(input='soft', derivable=_ppkernel_derivable)
def PPKernel(r, q=0, D=1):
    """
    Piecewise polynomial kernel.
    
    .. math::
        k(r) = \\text{polynomial}_{q,D}(r)
        \\begin{cases}
            1 - r & r \\in [0, 1)     \\\\
            0     & \\text{otherwise}
        \\end{cases}
    
    An isotropic kernel with finite support. The covariance is nonzero only
    when the distance between the points is less than 1. Parameter `q` in (0,
    1, 2, 3) sets the differentiability, while parameter `D` sets the maximum
    dimensionality the kernel can be used with. Default is `q` = 0 (non
    derivable), `D` = 1 (can be used only in 1D).
    
    """
    
    # TODO get the general formula for any q.
    
    # TODO add error checking on the dimensionality in IsotropicKernel, init
    # parameter `maxdim`.
    
    # TODO compute the kernel only on the nonzero points.
    
    # TODO find the nonzero points in O(nlogn) instead of O(n^2) by sorting
    # the inputs.
    
    assert isinstance(q, (int, np.integer))
    assert 0 <= q <= 3
    assert isinstance(D, (int, np.integer))
    assert D >= 1
    j = int(np.floor(D / 2)) + q + 1
    x = 1 - r
    if q == 0:
        poly = 1
    elif q == 1:
        poly = 1 + r * (j + 1)
    elif q == 2:
        poly = 1 + r * (j + 2 + r * ((1/3 * j +  4/3) * j + 1))
    elif q == 3:
        poly = 1 + r * (j + 3 + r * ((2/5 * j + 12/5) * j + 3 + r * (((1/15 * j + 3/5) * j + 23/15) * j + 1)))
    else:
        raise NotImplementedError
    return jnp.where(x > 0, x ** (j + q) * poly, 0)

@kernel(derivable=1, forcekron=True)
def WienerIntegral(x, y):
    """
    Kernel for a process whose derivative is a Wiener process.
    
    .. math::
        k(x, y) = \\frac 12 \\begin{cases}
            x^2 (y - x/3) & x < y, \\\\
            y^2 (x - y/3) & y \\le x
        \\end{cases}
    
    """
    
    # TODO can I generate this algorithmically for arbitrary integration order?
    # If I don't find a closed formula I can use sympy.
    
    # TODO write formula in terms of min(x, y) and max(x, y).
    
    assert jnp.all(_linalg.notracer(x) >= 0)
    assert jnp.all(_linalg.notracer(y) >= 0)
    return 1/2 * jnp.where(x < y, x**2 * (y - x/3), y**2 * (x - y/3))

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
    # TODO what is the "natural" extension of this to multidim? Is forcekron
    # appropriate?
    
    mul = x * y
    val = 2 * jnp.sqrt(jnp.abs(mul))
    return jnp.where(mul >= 0, jspecial.i0(val), _patch_jax.j0(val))

@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _bernoulli_poly(n, x):
    # takes x mod 1
    bernoulli = special.bernoulli(n)
    k = np.arange(n + 1)
    binom = special.binom(n, k)
    coeffs = binom[::-1] * bernoulli
    x = x % 1
    cond = x < 0.5
    x = jnp.where(cond, x, 1 - x)
    out = 0. * x # to handle the case n == 0
    for c in coeffs[:-1]:
        out += c
        out *= x
    out += coeffs[-1]
    if n % 2 == 1:
        out *= jnp.where(cond, 1, -1)
    return out

@_bernoulli_poly.defjvp
def _bernoulli_poly_jvp(n, primals, tangents):
    x, = primals
    xt, = tangents
    if n:
        t = n * _bernoulli_poly(n - 1, x) * xt
    else:
        t = 0 * xt
    return _bernoulli_poly(n, x), t

def _fourier_derivable(**kw):
    return kw.get('n', 2) - 1

@stationarykernel(forcekron=True, derivable=_fourier_derivable, saveargs=True)
def _FourierBase(delta, n=2):
    """
    Fourier kernel.
    
    .. math::
        k(x, y) &= \\frac1{\\zeta(2n)} \\sum_{k=1}^\\infty
        \\frac {\\cos(2\\pi kx)}{k^n} \\frac {\\cos(2\\pi ky)}{k^n}
        + \\frac1{\\zeta(2n)} \\sum_{k=1}^\\infty
        \\frac {\\sin(2\\pi kx)}{k^n} \\frac {\\sin(2\\pi ky)}{k^n} = \\\\
        &= \\frac1{\\zeta(2n)} \\sum_{k=1}^\\infty
        \\frac {\\cos(2\\pi k(x-y))} {k^{2n}} = \\\\
        &= (-1)^{n+1}
        \\frac1{\\zeta(2n)} \\frac {(2\\pi)^{2n}} {2(2n)!}
        B_{2n}(x-y \\bmod 1),
    
    where :math:`B_s(x)` is a Bernoulli polynomial. It is equivalent to fitting
    with a Fourier series of period 1 with independent priors on the
    coefficients with mean zero and variance
    :math:`1/(\\zeta(2n)k^{2n})`. The process is :math:`n - 1` times
    derivable.
    
    Note that the :math:`k = 0` term is not included in the summation, so the
    mean of the process over one period is forced to be zero.
    
    """
    
    # TODO maxk parameter to truncate the series. I have to manually sum the
    # components? => Bad idea then. => Can I sum analitically the residual?
    
    # TODO add constant as option, otherwise I can't compute the Fourier
    # series when I add a constant. => Maybe this will be solved when I
    # improve the transformations system.
    
    assert isinstance(n, (int, np.integer)), type(n)
    assert n >= 1, n # TODO I could allow n == 0 to be a constant kernel
    s = 2 * n
    sign0 = -(-1) ** n
    factor = (2 * np.pi) ** s / (2 * special.factorial(s) * special.zeta(s))
    return sign0 * factor * _bernoulli_poly(s, delta)

class Fourier(_FourierBase):
    
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
                order = np.ceil(k / 2)
                denom = order ** s * special.zeta(s)
                return np.where((k == q) & (k > 0), 1 / denom, 0)
        
        else:
            def kernel(k, y):
                order = np.ceil(k / 2)
                denom = order ** s * special.zeta(s)
                odd = k % 2
                arg = 2 * np.pi * order * y
                return np.where(k > 0, np.where(odd, np.sin(arg), np.cos(arg)) / denom, 0)
        
            if doy:
                kernel = lambda x, q, kernel=kernel: kernel(q, x)
        
        cls = _Kernel.Kernel if dox == doy and isinstance(self, _Kernel.Kernel) else _Kernel._CrossKernel
        # obj = cls(kernel, forcebroadcast=self._forcebroadcast)
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
    assert jnp.all(x >= 0)
    assert jnp.all(y >= 0)
    return jnp.exp(-jnp.abs(x - y)) - jnp.exp(-(x + y))
    
def _Celerite_derivable(**kw):
    gamma = kw.get('gamma', 1)
    B = kw.get('B', 0)
    if np.isscalar(gamma) and np.isscalar(B) and B == gamma:
        return 1
    else:
        return False

@stationarykernel(forcekron=True, derivable=_Celerite_derivable)
def Celerite(delta, gamma=1, B=0):
    """
    Celerite kernel.
    
    .. math::
        k(x, y) = \\exp(-\\gamma|x - y|)
        \\big( \\cos(x - y) + B \\sin(|x - y|) \\big)
    
    This is the covariance function of an AR(2) process with complex roots. The
    parameters must satisfy the condition :math:`|B| \\le \\gamma`. For
    :math:`B = \\gamma` it is equivalent to the :class:`Harmonic` kernel with
    :math:`\\eta Q = 1/B, Q > 1`, and it is derivable.
    
    Reference: Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, and Ruth
    Angus: *Fast and Scalable Gaussian Process Modeling With Applications To
    Astronomical Time Series*.
    """
    
    assert 0 <= gamma < jnp.inf, gamma
    assert jnp.abs(B) <= gamma, (B, gamma)
    tau = jnp.abs(delta)
    # TODO option input='abs' in StationaryKernel
    return jnp.exp(-gamma * tau) * (jnp.cos(tau) + B * jnp.sin(tau))

@kernel(forcekron=True, derivable=False)
def BrownianBridge(x, y):
    """
    Brownian bridge kernel.
    
    .. math::
        k(x, y) = \\min(x, y) - xy,
        \\quad x, y \\in [0, 1]
    
    It is a Wiener process conditioned on being zero at x = 1.
    """
    
    # TODO can this have a Hurst index? I think the kernel would be
    # (t^2H(1-s) + s^2H(1-t) + s(1-t)^2H + t(1-s)^2H - (t+s) - |t-s|^2H + 2ts)/2
    # but I have to check if it is correct. (In new kernel FracBrownianBridge.)
    
    assert jnp.all(0 <= x) and jnp.all(x <= 1)
    assert jnp.all(0 <= y) and jnp.all(y <= 1)
    return jnp.minimum(x, y) - x * y

@stationarykernel(forcekron=True, derivable=1)
def Harmonic(delta, Q=1):
    """
    Damped stochastically driven harmonic oscillator kernel.
    
    .. math::
        k(x, y) =
        \\exp\\left( -\\frac {\\tau} {Q} \\right)
        \\begin{cases}
            \\cosh(\\eta\\tau) + \\sinh(\\eta\\tau) / (\\eta Q)
            & 0 < Q < 1 \\\\
            1 + \\tau & Q = 1 \\\\
            \\cos(\\eta\\tau) + \\sin(\\eta\\tau) / (\\eta Q)
            & Q > 1,
        \\end{cases}
    
    where :math:`\\tau = |x - y|` and :math:`\\eta = \\sqrt{|1 - 1/Q^2|}`.
    
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
    
    assert 0 < Q < np.inf, Q
    
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

@stationarykernel(forcekron=True, derivable=False)
def Expon(delta):
    """
    Exponential kernel.
    
    .. math::
        k(x, y) = \\exp(-|x - y|)
    
    In 1D it is equivalent to the Matérn 1/2 kernel, however in more dimensions
    it acts separately while the Matérn kernel is isotropic.
    """
    return np.exp(-jnp.abs(delta))

_bow_regexp = re.compile(r'\s|[!«»"“”‘’/()\'?¡¿„‚<>,;.:-–—]')

@kernel(forcekron=True, derivable=False)
@np.vectorize
def BagOfWords(x, y):
    """
    Bag of words kernel.
    
    .. math::
        k(x, y) &= \\sum_{w \\in \\text{words}} c_w(x) c_w(y), \\\\
        c_w(x) &= \\text{number of times word $w$ appears in $x$}
    
    The words are defined as non-empty substrings delimited by spaces or one of
    the following punctuation characters: ! « » " “ ” ‘ ’ / ( ) ' ? ¡ ¿ „ ‚ < >
    , ; . : - – —.
    """
    
    # TODO precompute the bags for x and y, then call a vectorized private
    # function.
    
    # TODO iterate on the shorter bag and use get on the other instead of
    # computing set intersection? Or: convert words to integers and then do
    # set intersection with sorted arrays?
    
    xbag = collections.Counter(_bow_regexp.split(x))
    ybag = collections.Counter(_bow_regexp.split(y))
    xbag[''] = 0
    ybag[''] = 0
    common = set(xbag) & set(ybag)
    return sum(xbag[k] * ybag[k] for k in common)
