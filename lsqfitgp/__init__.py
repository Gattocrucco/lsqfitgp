from ._imports import numpy, scipy # keep this first
from ._GP import *
from ._Kernel import *
from ._kernels import *
from ._array import *
from ._fit import *
from ._Deriv import *

__version__ = '0.1.3'

__doc__ = """

Module to fit gaussian processes with gvar/lsqfit.

Manual: https://lsqfitgp.readthedocs.io/en/latest

It can both be used standalone to fit data with a gaussian process only, and
with lsqfit inside a possibly nonlinear model with other parameters. In lsqfit
style, all the results will be properly correlated with prior, data, and other
non-gaussian process parameters in the fit, even when doing conditional
prediction.

The main class is `GP`, which represents a gaussian process over arbitrary
input. It can be used both autonomously and with lsqfit. The inputs/outputs can
be arrays or dictionaries of arrays. It supports doing inference with the
derivatives of the process, using `autograd` to compute automatically
derivatives of the kernels. Indirectly, this can be used to make inference with
integrals.

Functions and classes
---------------------
    
    GP : class
        Class of objects representing a gaussian process.
    empbayes_fit : function
        Fit the hyperparameters of a gaussian process.
    StructuredArray : class
        Autograd-friendly wrapper of numpy structured arrays.
    where : function
        Make a kernel that switches between two kernels based on a condition.
    Deriv : class
        Class representing a derivative specification.

Kernels
-------

The covariance kernels are represented by subclasses of class `Kernel`. There's
also `IsotropicKernel` for covariance functions that depend only on the
distance between the arguments. Kernel objects can be summed, multiplied and
raised to a power.

To make a custom kernel, you can instantiate one of the two general classes by
passing them a function, or subclass them. For convenience, decorators `kernel`
and `isotropickernel` are provided to convert a function to a covariance
kernel. Otherwise, use one of the already available subclasses listed below.
Isotropic kernels are normalized to have unit variance and roughly unit
lengthscale.

    Constant
        Equivalent to fitting with a constant.
    Linear
        Equivalent to fitting with a line.
    Polynomial
        Equivalent to fitting with a polynomial.
    ExpQuad
        Gaussian kernel.
    White
        White noise, each point is indipendent.
    Matern
        Matérn kernel, you can set how many times it is differentiable.
    Matern12, Matern32, Matern52
        Matérn kernel for the specific cases nu = 1/2, 3/2, 5/2.
    GammaExp
        Gamma exponential. Not differentiable, but you can set how close it is
        to being differentiable.
    RatQuad
        Equivalent to a mixture of gaussian kernels with gamma-distributed
        length scales.
    NNKernel
        Equivalent to training a neural network with one latent infinite layer.
    Wiener
        Random walk.
    WienerIntegral
        Integral of the random walk.
    Gibbs
        A gaussian kernel with a custom variable length scale.
    Periodic
        A periodic gaussian kernel, represents a periodic function.
    Categorical
        Arbitrary covariance matrix over a finite set of values.
    Cos
        A cosine.
    FracBrownian
        Fractional Brownian motion, like Wiener but with correlations.
    PPKernel
        Finite support isotropic kernel.
    Rescaling
        Kernel used to change the variance of other kernels.

Reference: Rasmussen et al. (2006), "Gaussian Processes for Machine Learning".

"""
