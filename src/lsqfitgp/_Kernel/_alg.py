# lsqfitgp/_Kernel/_alg.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

""" register algops on CrossKernel and AffineSpan """

import functools

from jax import numpy as jnp
from jax.scipy import special as jspecial

from .. import _special

from . import _util
from ._crosskernel import CrossKernel, AffineSpan

@CrossKernel.register_algop
def add(tcls, self, other):
    r"""
    
    Sum of kernels.
    
    .. math::
        \mathrm{newkernel}(x, y) &= \mathrm{kernel}(x, y) + \mathrm{other}(x, y), \\
        \mathrm{newkernel}(x, y) &= \mathrm{kernel}(x, y) + \mathrm{other}.
    
    Parameters
    ----------
    other : CrossKernel or scalar
        The other kernel.
    
    """
    core = self.core
    if _util.is_numerical_scalar(other):
        newcore = lambda x, y, **kw: core(x, y, **kw) + other
    elif isinstance(other, CrossKernel):
        other = other.core
        newcore = lambda x, y, **kw: core(x, y, **kw) + other(x, y, **kw)
    else:
        return NotImplemented
    return self._clone(core=newcore)

@CrossKernel.register_algop
def mul(tcls, self, other):
    r"""
    
    Product of kernels.
    
    .. math::
        \mathrm{newkernel}(x, y) &= \mathrm{kernel}(x, y) \cdot \mathrm{other}(x, y), \\
        \mathrm{newkernel}(x, y) &= \mathrm{kernel}(x, y) \cdot \mathrm{other}.
    
    Parameters
    ----------
    other : CrossKernel or scalar
        The other kernel.
    
    """
    core = self.core
    if _util.is_numerical_scalar(other):
        newcore = lambda x, y, **kw: core(x, y, **kw) * other
    elif isinstance(other, CrossKernel):
        other = other.core
        newcore = lambda x, y, **kw: core(x, y, **kw) * other(x, y, **kw)
    else:
        return NotImplemented
    return self._clone(core=newcore)

@CrossKernel.register_algop
def pow(tcls, self, *, exponent):
    r"""
    
    Power of the kernel.
    
    .. math::
        \mathrm{newkernel}(x, y) = \mathrm{kernel}(x, y)^{\mathrm{exponent}}
    
    Parameters
    ----------
    exponent : nonnegative integer
        The exponent. If traced by jax, it must have unsigned integer type.
    
    """
    if _util.is_nonnegative_integer_scalar(exponent):
        core = self.core
        newcore = lambda x, y, **kw: core(x, y, **kw) ** exponent
        return self._clone(core=newcore)
    else:
        return NotImplemented

    # TODO this will raise TypeError on negative integers. It should stop
    # method search and raise ValueError. Same for rpow. Check if it is a
    # scalar, then check if it satisfies the bound.

@CrossKernel.register_algop
def rpow(tcls, self, *, base):
    r"""
    
    Exponentiation of the kernel.
    
    .. math::
        \text{newkernel}(x, y) = \text{base}^{\text{kernel}(x, y)}
    
    Parameters
    ----------
    base : scalar
        A number >= 1. If traced by jax, the value is not checked.
    
    """
    if _util.is_scalar_cond_trueontracer(base, lambda x: x >= 1):
        core = self.core
        newcore = lambda x, y, **kw: base ** core(x, y, **kw)
        return self._clone(core=newcore)
    else:
        return NotImplemented

CrossKernel.register_ufuncalgop(jnp.tan)
# CrossKernel.register_ufuncalgop(lambda x: 1 / jnp.sinc(x), '1/sinc')
CrossKernel.register_ufuncalgop(lambda x: 1 / jnp.cos(x), '1/cos')
CrossKernel.register_ufuncalgop(jnp.arcsin)
CrossKernel.register_ufuncalgop(lambda x: 1 / jnp.arccos(x), '1/arccos')
CrossKernel.register_ufuncalgop(lambda x: 1 / (1 - x), '1/(1-x)')
CrossKernel.register_ufuncalgop(jnp.exp)
CrossKernel.register_ufuncalgop(lambda x: -jnp.log1p(-x), '-log1p(-x)')
CrossKernel.register_ufuncalgop(jnp.expm1)
CrossKernel.register_ufuncalgop(_special.expm1x)
CrossKernel.register_ufuncalgop(jnp.sinh)
CrossKernel.register_ufuncalgop(jnp.cosh)
CrossKernel.register_ufuncalgop(jnp.arctanh)
CrossKernel.register_ufuncalgop(jspecial.i0)
CrossKernel.register_ufuncalgop(jspecial.i1)
# @CrossKernel.register_ufuncalgop
# def iv(x, *, order):
#     assert _util.is_nonnegative_scalar_trueontracer(order)
#     return _special.iv(order, x)

# TODO other unary algop:
# - hypergeom (wrap the scipy impl in _special)

@functools.partial(AffineSpan.register_algop, transfname='add')
def affine_add(tcls, self, other):
    newself = AffineSpan.super_transf('add', self, other)
    if _util.is_numerical_scalar(other):
        dynkw = dict(self.dynkw)
        dynkw['offset'] = dynkw['offset'] + other
        return newself._clone(self.__class__, dynkw=dynkw)
    else:
        return newself

@functools.partial(AffineSpan.register_algop, transfname='mul')
def affine_mul(tcls, self, other):
    newself = AffineSpan.super_transf('mul', self, other)
    if _util.is_numerical_scalar(other):
        dynkw = dict(self.dynkw)
        dynkw['offset'] = other * dynkw['offset']
        dynkw['ampl'] = other * dynkw['ampl']
        return newself._clone(self.__class__, dynkw=dynkw)
    else:
        return newself
