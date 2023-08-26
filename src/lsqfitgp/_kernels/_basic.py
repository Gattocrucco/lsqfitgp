# lsqfitgp/_kernels/_basic.py
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

import sys
import re
import collections

import numpy
import jax
from jax import numpy as jnp
from jax.scipy import special as jspecial

from .. import _special
from .. import _jaxext
from .. import _Kernel
from .._Kernel import kernel, stationarykernel, isotropickernel

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

def _dot(x, y):
    return _Kernel.sum_recurse_dtype(lambda x, y: x * y, x, y)
    
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

@isotropickernel(derivable=lambda gamma=1: gamma == 2)
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
    with _jaxext.skipifabstract():
        assert 0 < gamma <= 2, gamma
    nondiff = jnp.exp(-(r2 ** (gamma / 2)))
    diff = jnp.exp(-r2)
    return jnp.where(gamma == 2, diff, nondiff)
    # I need to keep separate the case where derivatives w.r.t. r2 could be
    # computed because the second derivative at x=0 of x^p with floating
    # point p is nan.
    
    # TODO extend to gamma=0, the correct limit is
    # e^-1 constant + (1 - e^-1) white noise. Use lax.switch.

    # TODO derivatives w.r.t. gamma at gamma==2 are probably broken, although
    # I guess they are not needed since it's on the boundary of the domain
    
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
    where ``sigma0`` sets the dispersion of the centers of the sigmoids.
    
    Reference: Rasmussen and Williams (2006, p. 90).
    """
    
    # TODO the `2`s in the formula are a bit arbitrary. Remove them or give
    # motivation relative to the precise formulation of the neural network.
    with _jaxext.skipifabstract():
        assert 0 < sigma0 < jnp.inf
    q = sigma0 ** 2
    denom = (1 + 2 * (q + _dot(x, x))) * (1 + 2 * (q + _dot(y, y)))
    return 2/jnp.pi * jnp.arcsin(2 * (q + _dot(x, y)) / denom)
    
    # TODO this is not fully equivalent to an arbitrary transformation on the
    # augmented vector even if x and y are transformed, unless I support q
    # being a vector or an additional parameter.

    # TODO if arcsin has positive taylor coefficients, this can be obtained as
    # arcsin(1 + linear) * rescaling.

@kernel
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
    point. By default ``scalefun`` returns 1 so it is a Gaussian kernel.
    
    Consider that the default parameter ``scale`` acts before ``scalefun``, so
    for example if ``scalefun(x) = x`` then ``scale`` has no effect. You should
    include all rescalings in ``scalefun`` to avoid surprises.
    
    Reference: Rasmussen and Williams (2006, p. 93).
    """
    sx = scalefun(x)
    sy = scalefun(y)
    with _jaxext.skipifabstract():
        assert jnp.all(sx > 0)
        assert jnp.all(sy > 0)
    denom = sx ** 2 + sy ** 2
    factor = jnp.sqrt(2 * sx * sy / denom)
    distsq = _Kernel.sum_recurse_dtype(lambda x, y: (x - y) ** 2, x, y)
    return factor * jnp.exp(-distsq / denom)

@stationarykernel(derivable=True, maxdim=1)
def Periodic(delta, outerscale=1):
    r"""
    Periodic Gaussian kernel.
    
    .. math::
        k(\Delta) = \exp \left(
        -2 \left(
        \frac {\sin(\Delta / 2)} {\texttt{outerscale}}
        \right)^2
        \right)
    
    A Gaussian kernel over a transformed periodic space. It represents a
    periodic process. The usual `scale` parameter sets the period, with the
    default ``scale=1`` giving a period of 2π, while `outerscale` sets the
    length scale of the correlations.
    
    Reference: Rasmussen and Williams (2006, p. 92).
    """
    with _jaxext.skipifabstract():
        assert 0 < outerscale < jnp.inf
    return jnp.exp(-2 * (jnp.sin(delta / 2) / outerscale) ** 2)

@kernel(derivable=False, maxdim=1)
def Categorical(x, y, cov=None):
    r"""
    Categorical kernel.
    
    .. math::
        k(x, y) = \texttt{cov}[x, y]
    
    A kernel over integers from 0 to N-1. The parameter `cov` is the covariance
    matrix of the values.
    """
        
    # TODO support sparse matrix for cov (replace jnp.asarray and numpy
    # check)
    
    assert jnp.issubdtype(x.dtype, jnp.integer)
    cov = jnp.asarray(cov)
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1]
    with _jaxext.skipifabstract():
        assert jnp.allclose(cov, cov.T)
    return cov[x, y]

@kernel
def Rescaling(x, y, stdfun=None):
    r"""
    Outer product kernel.
    
    .. math::
        k(x, y) = \texttt{stdfun}(x) \texttt{stdfun}(y)
    
    A totally correlated kernel with arbitrary variance. Parameter `stdfun`
    must be a function that takes ``x`` or ``y`` and computes the standard
    deviation at the point. It can yield negative values; points with the same
    sign of `stdfun` will be totally correlated, points with different sign will
    be totally anticorrelated. Use this kernel to modulate the variance of
    other kernels. By default `stdfun` returns a constant, so it is equivalent
    to `Constant`.
    
    """
    if stdfun is None:
        stdfun = lambda x: jnp.ones(x.shape)
        # do not use np.ones_like because it does not recognize StructuredArray
        # do not use x.dtype because it could be structured
    return stdfun(x) * stdfun(y)

@stationarykernel(derivable=False, input='abs', maxdim=1)
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

    # TODO rename Laplace, write it in terms of the 1-norm directly, then do
    # TruncLaplace that reaches zero over a prescribed box. (Or truncate with
    # an option truncbox=[(l0, r0), (l1, r1), ...]).

_bow_regexp = re.compile(r'\s|[!«»"“”‘’/()\'?¡¿„‚<>,;.:-–—]')

@kernel(derivable=False, maxdim=1)
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

@stationarykernel(derivable=False, input='abs', maxdim=1)
def HoleEffect(delta):
    """
    
    Hole effect kernel.
    
    .. math:: k(\\Delta) = (1 - \\Delta) \\exp(-\\Delta)
    
    Reference: Dietrich and Newsam (1997, p. 1096).
    
    """
    return (1 - delta) * jnp.exp(-delta)

def _cauchy_derivable(alpha=2, **_):
    return alpha == 2

@isotropickernel(derivable=_cauchy_derivable)
def Cauchy(r2, alpha=2, beta=2):
    r"""
    Generalized Cauchy kernel.
    
    .. math::
        k(r) = \left(1 + \frac{r^\alpha}{\beta} \right)^{-\beta/\alpha},
        \quad \alpha \in (0, 2], \beta > 0.
    
    In the geostatistics literature, the case :math:`\alpha=2` and
    :math:`\beta=2` (default) is known as the Cauchy kernel. In the machine
    learning literature, the case :math:`\alpha=2` (for any :math:`\beta`) is
    known as the rational quadratic kernel. For :math:`\beta\to\infty` it is
    equivalent to ``GammaExp(gamma=alpha, scale=alpha ** (1/alpha))``, while
    for :math:`\beta\to 0` to ``Constant``. It is smooth only for
    :math:`\alpha=2`.
    
    References: Gneiting and Schlather (2004, p. 273), Rasmussen and Williams
    (2006, p. 86).
    
    """
    with _jaxext.skipifabstract():
        assert 0 < alpha <= 2, alpha
        assert 0 < beta, beta
    power = jnp.where(alpha == 2, r2, r2 ** (alpha / 2))
    # I need to keep separate the case where derivatives w.r.t. r2 could be
    # computed because the second derivative at x=0 of x^p with floating
    # point p is nan.
    return (1 + power / beta) ** (-beta / alpha)
    
    # TODO derivatives w.r.t. alpha at alpha==2 are probably broken, although
    # I guess they are not needed since it's on the boundary of the domain
    
@isotropickernel(derivable=lambda alpha=1: alpha == 0, input='posabs')
def CausalExpQuad(r, alpha=1):
    r"""
    Causal exponential quadratic kernel.
    
    .. math::
        k(r) = \big(1 - \operatorname{erf}(\alpha r/4)\big)
        \exp\left(-\frac12 r^2 \right)
        
    From https://github.com/wesselb/mlkernels.
    """
    with _jaxext.skipifabstract():
        assert alpha >= 0, alpha
    return jspecial.erfc(alpha / 4 * r) * jnp.exp(-1/2 * jnp.square(r))
    # TODO taylor-expand erfc near 0 and use r2

    # TODO is the erfc part a standalone valid kernel? If so, separate it,
    # since this can be obtained as the product

@kernel(derivable=True, maxdim=1)
def Decaying(x, y, alpha=1):
    r"""
    Decaying kernel.
    
    .. math::
        k(x, y) =
        \frac{1}{(1 + x + y)^\alpha},
        \quad x, y, \alpha \ge 0
        
    Reference: Swersky, Snoek and Adams (2014).
    """
    # TODO high dimensional version of this, see mlkernels issue #3
    with _jaxext.skipifabstract():
        assert jnp.all(x >= 0)
        assert jnp.all(y >= 0)
    return 1 / (x + y + 1) ** alpha
    # use x + y + 1 instead of 1 + x + y because the latter is less numerically
    # accurate and symmetric for small x and y

@isotropickernel(derivable=False, input='posabs')
def Log(r):
    """
    Log kernel.
    
    .. math::
        k(r) = \\log(1 + r) / r
    
    From https://github.com/wesselb/mlkernels.
    """
    return jnp.log1p(r) / r

@kernel(derivable=True, maxdim=1)
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
    
    mul = x * y
    val = 2 * jnp.sqrt(jnp.abs(mul))
    return jnp.where(mul >= 0, jspecial.i0(val), _special.j0(val))

    # TODO reference? Maybe it's called bessel kernel in the literature?
    # => nope, bessel kernel is the J_v one
        
    # TODO what is the "natural" extension of this to multidim?
    
    # TODO probably the rescaled version of this (e^-x) makes more sense
