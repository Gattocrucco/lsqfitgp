# lsqfitgp/_kernels/_celerite.py
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

import jax
from jax import numpy as jnp

from .. import _jaxext
from .._Kernel import stationarykernel
    
def _Celerite_derivable(**kw):
    gamma = kw.get('gamma', 1)
    B = kw.get('B', 0)
    if jnp.isscalar(gamma) and jnp.isscalar(B) and B == gamma:
        return 1
    else:
        return False

@stationarykernel(derivable=_Celerite_derivable, input='abs', maxdim=1)
def Celerite(delta, gamma=1, B=0):
    """
    Celerite kernel.
    
    .. math::
        k(\\Delta) = \\exp(-\\gamma|\\Delta|)
        \\big( \\cos(\\Delta) + B \\sin(|\\Delta|) \\big)
    
    This is the covariance function of an AR(2) process with complex roots. The
    parameters must satisfy the condition :math:`|B| \\le \\gamma`. For
    :math:`B = \\gamma` it is equivalent to the `Harmonic` kernel with
    :math:`\\eta Q = 1/B, Q > 1`, and it is derivable.
    
    Reference: Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, and Ruth
    Angus: *Fast and Scalable Gaussian Process Modeling With Applications To
    Astronomical Time Series*.
    """
    with _jaxext.skipifabstract():
        assert 0 <= gamma < jnp.inf, gamma
        assert abs(B) <= gamma, (B, gamma)
    return jnp.exp(-gamma * delta) * (jnp.cos(delta) + B * jnp.sin(delta))

@stationarykernel(derivable=1, maxdim=1)
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
    
    where :math:`w` is white noise.
    
    The parameter :math:`Q` is the quality factor, i.e., the ratio between the energy
    stored in the oscillator and the energy lost in each cycle due to damping.
    The angular frequency is 1, i.e., the period is 2π. The process is derivable
    one time.
    
    In 1D, for :math:`Q = 1` (default) and ``scale=sqrt(1/3)``, it is the Matérn 3/2
    kernel.
    
    Reference: Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, and Ruth
    Angus: *Fast and Scalable Gaussian Process Modeling With Applications To
    Astronomical Time Series*.
    """
    
    # TODO improve and test the numerical accuracy for derivatives near x=0
    # and Q=1. I don't know if the derivatives have problems away from Q=1.
    
    # TODO probably second derivatives w.r.t. Q at Q=1 are wrong.
    
    # TODO will fail if Q is traced.
    
    with _jaxext.skipifabstract():
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
