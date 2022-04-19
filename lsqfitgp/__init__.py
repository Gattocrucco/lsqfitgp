# lsqfitgp/__init__.py
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

from ._imports import numpy, scipy # keep this first
from ._GP import *
from ._Kernel import *
from ._kernels import *
from ._array import *
from ._fit import *
from ._Deriv import *
from ._fastraniter import *

__version__ = '0.7'

__doc__ = """

Module to fit Gaussian processes with gvar/lsqfit.

Manual: https://lsqfitgp.readthedocs.io/en/latest

It can both be used standalone to fit data with a Gaussian process only, and
with lsqfit inside a possibly nonlinear model with other parameters. In lsqfit
style, all the results will be properly correlated with prior, data, and other
non-Gaussian process parameters in the fit, even when doing conditional
prediction.

The main class is `GP`, which represents a Gaussian process over arbitrary
input. It can be used both autonomously and with lsqfit. The inputs/outputs can
be arrays or dictionaries of arrays. It supports doing inference with the
derivatives of the process, using `autograd` to compute automatically
derivatives of the kernels. Indirectly, this can be used to make inference with
integrals.

Functions and classes
---------------------
    
    GP : class
        Class of objects representing a Gaussian process.
    empbayes_fit : class
        Fit the hyperparameters of a Gaussian process.
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
        Equivalent to a mixture of Gaussian kernels with gamma-distributed
        length scales.
    NNKernel
        Equivalent to training a neural network with one latent infinite layer.
    Wiener
        Random walk.
    WienerIntegral
        Integral of the random walk.
    FracBrownian
        Fractional Brownian motion, like Wiener but with correlations.
    OrnsteinUhlenbeck
        Random walk with asymptotically finite variance.
    BrownianBridge
        Random walk which comes back to the starting point.
    Gibbs
        A Gaussian kernel with a custom variable length scale.
    Categorical
        Arbitrary covariance matrix over a finite set of values.
    Cos
        A cosine.
    Celerite
        A pair of conjugated complex exponentials.
    Harmonic
        Stochastically driven damped harmonic oscillator.
    PPKernel
        Finite support isotropic kernel.
    Rescaling
        Kernel used to change the variance of other kernels.
    Taylor
        Exponential-like taylor series.
    Fourier
        Kernel for periodic functions, the decay of the Fourier coefficients is
        adjustable.
    Periodic
        A periodic Gaussian kernel, represents a periodic function.

Reference: Rasmussen et al. (2006), "Gaussian Processes for Machine Learning".

"""
