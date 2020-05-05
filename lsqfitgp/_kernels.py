from ._imports import numpy as np
from ._imports import special
from ._imports import autograd
import numpy # to bypass autograd

from . import _array
from . import _Kernel
from ._Kernel import kernel, isotropickernel

__all__ = [
    'Constant',
    'White',
    'ExpQuad',
    'Linear',
    'Polynomial',
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
    'WienerIntegral'
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
    return np.ones_like(r2)
    # TODO maybe use dtype=int8 as optimization?
    
@isotropickernel
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
    return np.where(r2 == 0, 1, 0)

@isotropickernel(derivable=True)
def ExpQuad(r2):
    """
    Exponential quadratic kernel.
    
    .. math::
        k(r) = \\exp \\left( -\\frac 12 r^2 \\right)
    
    It is smooth and has a strict typical lengthscale, i.e. oscillations are
    strongly suppressed under a certain wavelength, and correlations are
    strongly suppressed over a certain distance.
    """
    return np.exp(-1/2 * r2)

@kernel(derivable=True)
def Linear(x, y):
    """
    Dot product kernel.
    
    .. math::
        k(x, y) = x \\cdot y = \\sum_i x_i y_i
    
    In 1D it is equivalent to fitting with a line passing by the origin.
    """
    return _dot(x, y)

@kernel(derivable=True)
def Polynomial(x, y, exponent=None, sigma0=1):
    """
    Dot product polynomial kernel.
    
    .. math::
        k(x, y) = (x \\cdot y + \\texttt{sigma0}^2)^\\texttt{exponent}
    
    In 1D it is equivalent to fitting with a polynomial of degree `exponent`.
    """
    assert np.isscalar(exponent)
    assert exponent >= 0
    assert np.isscalar(sigma0)
    assert sigma0 >= 0
    return (_dot(x, y) + sigma0 ** 2) ** exponent

# TODO make a pull request to autograd for kv and kvp.
# This still does not work with derivatives due to the pole of kv. I need a
# direct calculation of x ** nu * kv(nu, x).
# _kvp = extend.primitive(special_noderiv.kvp)
# extend.defvjp(
#     _kvp,
#     lambda ans, v, z, n: lambda g: g * _kvp(v, z, n + 1),
#     argnums=[1]
# )
# _kv = lambda v, z: _kvp(v, z, 0)
if special is not None:
    from scipy import special
    _kv = special.kv

def _maternp(x, p):
    poly = 1
    for k in reversed(range(p)):
        c_kp1_over_ck = (p - k) / ((2 * p - k) * (k + 1))
        poly *= c_kp1_over_ck * 2 * x
        poly += 1
    return np.exp(-x) * poly

def _maternp_deriv(x, p):
    if p == 0:
        return -np.exp(-x)
    poly = 1
    for k in reversed(range(1, p)):
        c_kp1_over_ck = (p - k) / ((2 * p - k - 1) * k)
        poly = 1 + poly * c_kp1_over_ck * 2 * x
    poly = poly / (1 - 2 * p) * x
    return np.exp(-x) * poly

if autograd is not None:
    _maternp = autograd.extend.primitive(_maternp)
    autograd.extend.defvjp(
        _maternp,
        lambda ans, x, p: lambda g: g * _maternp_deriv(x, p)
    )

def _matern_derivable(**kw):
    nu = kw.get('nu', None)
    if np.isscalar(nu) and nu > 0 and (2 * nu) % 1 == 0:
        return int(nu - 1/2)
    else:
        return False

@isotropickernel(input='soft', derivable=_matern_derivable)
def Matern(r, nu=None):
    """
    Matérn kernel of real order. 
    
    .. math::
        k(r) = \\frac {2^{1-\\nu}} {\\Gamma(\\nu)} x^\\nu K_\\nu(x),
        \\quad \\nu = \\texttt{nu} > 0,
        \\quad x = \\sqrt{2\\nu} r
    
    The nearest integer below `nu` indicates how many times the gaussian
    process is derivable: so for `nu` < 1 it is continuous but not derivable,
    for 1 <= `nu` < 2 it is derivable but has not a decond derivative, etc. The
    half-integer case (nu = 1/2, 3/2, ...) uses internally a simpler formula so
    you should prefer it. Also, taking derivatives of the process is supported
    only for half-integer nu.

    """
    assert np.isscalar(nu)
    assert nu > 0
    x = np.sqrt(2 * nu) * r
    if (2 * nu) % 1 == 0:
        return _maternp(x, int(nu - 1/2))
    else:
        return 2 ** (1 - nu) / special.gamma(nu) * x ** nu * _kv(nu, x)

@isotropickernel(input='soft')
def Matern12(r):
    """
    Matérn kernel of order 1/2 (continuous, not derivable).
    
    .. math::
        k(r) = \\exp(-r)
    """
    return np.exp(-r)

def _matern32(x):
    return (1 + x) * np.exp(-x)

if autograd is not None:
    _matern32 = autograd.extend.primitive(_matern32)
    autograd.extend.defvjp(
        _matern32,
        lambda ans, x: lambda g: g * -x * np.exp(-x)
    )

@isotropickernel(input='soft', derivable=1)
def Matern32(r):
    """
    Matérn kernel of order 3/2 (derivable one time).
    
    .. math::
        k(r) &= (1 + x) \\exp(-x), \\\\
        x &= \\sqrt3 r
    """
    return _matern32(np.sqrt(3) * r)

def _matern52(x):
    return (1 + x * (1 + x/3)) * np.exp(-x)

if autograd is not None:
    _matern52 = autograd.extend.primitive(_matern52)
    autograd.extend.defvjp(
        _matern52,
        lambda ans, x: lambda g: g * -x/3 * _matern32(x)
    )

@isotropickernel(input='soft', derivable=2)
def Matern52(r):
    """
    Matérn kernel of order 5/2 (derivable two times).
    
    .. math::
        k(r) &= (1 + x + x^2/3) \\exp(-x), \\\\
        x &= \\sqrt5 r
    """
    return _matern52(np.sqrt(5) * r)

@isotropickernel(input='soft')
def GammaExp(r, gamma=1):
    """
    Gamma exponential kernel.
    
    .. math::
        k(r) = \\exp(-r^\\texttt{gamma}), \\quad
        \\texttt{gamma} \\in [0, 2]
    
    For `gamma` = 2 it is the gaussian kernel, for `gamma` = 1 it is the Matérn
    1/2 kernel, for `gamma` = 0 it is the constant kernel. The process is
    differentiable only for `gamma` = 2, however as `gamma` gets closer to 2
    the variance of the non-derivable component goes to zero.

    """
    assert np.isscalar(gamma)
    assert 0 <= gamma <= 2
    return np.exp(-(r ** gamma))

@isotropickernel(derivable=True)
def RatQuad(r2, alpha=2):
    """
    Rational quadratic kernel.
    
    .. math::
        k(r) = \\left( 1 + \\frac {r^2} {2 \\alpha} \\right)^{-\\alpha},
        \\quad \\alpha = \\texttt{alpha}
    
    It is equivalent to a lengthscale mixture of gaussian kernels where the
    scale distribution is a gamma with shape parameter `alpha`. For `alpha` ->
    infinity, it becomes the gaussian kernel. It is smooth.
    
    """
    assert np.isscalar(alpha)
    assert 0 < alpha < np.inf
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
    layer with gaussian priors on the weights and error function response. In
    other words, you can think of the process as a superposition of sigmoids
    where `sigma0` sets the dispersion of the centers of the sigmoids.
    
    """
    assert np.isscalar(sigma0)
    assert np.isfinite(sigma0)
    assert sigma0 > 0
    q = sigma0 ** 2
    denom = (1 + 2 * (q + _dot(x, x))) * (1 + 2 * (q + _dot(y, y)))
    return 2/np.pi * np.arcsin(2 * (q + _dot(x, y)) / denom)
    
    # TODO this is not fully equivalent to an arbitrary transformation on the
    # augmented vector even if x and y are transformed, unless I support q
    # being a vector or an additional parameter.

@kernel(forcekron=True)
def Wiener(x, y):
    """
    Wiener kernel.
    
    .. math::
        k(x, y) = \\min(x, y), \\quad x, y > 0
    
    A kernel representing a non-differentiable random walk starting at 0.
    
    """
    assert np.all(x >= 0)
    assert np.all(y >= 0)
    return np.minimum(x, y)

@kernel(forcekron=True)
def Gibbs(x, y, scalefun=lambda x: 1):
    """
    Gibbs kernel.
    
    .. math::
        k(x, y) = \\sqrt{ \\frac {2 s(x) s(y)} {s(x)^2 + s(y)^2} }
        \\exp \\left( -\\frac {(x - y)^2} {s(x)^2 + s(y)^2} \\right),
        \\quad s = \\texttt{scalefun}.
    
    Kernel which in some sense is like a gaussian kernel where the scale
    changes at every point. The scale is computed by the parameter `scalefun`
    which must be a callable taking the x array and returning a scale for each
    point. By default `scalefun` returns a constant so it is a gaussian kernel.
    
    """
    sx = scalefun(x)
    sy = scalefun(y)
    assert np.all(sx > 0)
    assert np.all(sy > 0)
    denom = sx ** 2 + sy ** 2
    factor = np.sqrt(2 * sx * sy / denom)
    return factor * np.exp(-(x - y) ** 2 / denom)

@kernel(derivable=True, forcekron=True)
def Periodic(x, y, outerscale=1):
    """
    Periodic gaussian kernel.
    
    .. math::
        k(x, y) = \\exp \\left(
        -2 \\left(
        \\frac {\\sin((x - y) / 2)} {\\texttt{outerscale}}
        \\right)^2
        \\right)
    
    A gaussian kernel over a transformed periodic space. It represents a
    periodic process. The usual `scale` parameter sets the period, with the
    default `scale` = 1 giving a period of 2π, while the `outerscale` parameter
    sets the length scale of the correlations.
    
    """
    assert np.isscalar(outerscale)
    assert outerscale > 0
    return np.exp(-2 * (np.sin((x - y) / 2) / outerscale) ** 2)

@kernel(forcekron=True)
def Categorical(x, y, cov=None):
    """
    Categorical kernel.
    
    .. math::
        k(x, y) = \\texttt{cov}[x, y]
    
    A kernel over integers from 0 to N-1. The parameter `cov` is the covariance
    matrix of the values.
    """
    assert np.issubdtype(x.dtype, numpy.integer)
    cov = np.array(cov, copy=False)
    assert len(cov.shape) == 2
    assert cov.shape[0] == cov.shape[1]
    assert np.allclose(cov, cov.T)
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
        stdfun = lambda x: np.ones_like(x, dtype=float)
    return stdfun(x) * stdfun(y)

@kernel(derivable=True, forcekron=True)
def Cos(x, y):
    """
    Cosine kernel.
    
    .. math::
        k(x, y) = \\cos(x - y)
    
    Samples from this kernel are harmonic functions. It can be multiplied with
    another kernel to introduce anticorrelations.
    
    """
    return np.cos(x - y)

@kernel(forcekron=True)
def FracBrownian(x, y, H=1/2):
    """
    Fractional brownian motion kernel.
    
    .. math::
        k(x, y) = \\frac 12 (x^{2H} + y^{2H} - |x-y|^{2H}),
        \\quad H \\in (0, 1)
    
    For `H` = 1/2 (default) it is the Wiener kernel. For `H` in (0, 1/2) the
    increments are anticorrelated (strong oscillation), for `H` in (1/2, 1)
    the increments are correlated (tends to keep a slope).
    
    """
    assert np.isscalar(H)
    assert 0 < H < 1
    assert np.all(x >= 0)
    assert np.all(y >= 0)
    H2 = 2 * H
    return 1/2 * (x ** H2 + y ** H2 - np.abs(x - y) ** H2)

def _ppkernel_derivable(**kw):
    q = kw.pop('q', 0)
    if isinstance(q, (int, np.integer)):
        return q
    else:
        return 0

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
    return np.where(x > 0, x ** (j + q) * poly, 0)

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
    assert np.all(x >= 0)
    assert np.all(y >= 0)
    return 1/2 * np.where(x < y, x**2 * (y - x/3), y**2 * (x - y/3))
