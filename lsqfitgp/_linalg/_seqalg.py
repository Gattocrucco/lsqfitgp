# lsqfitgp/_linalg/_seqalg.py
#
# Copyright (c) 2022, Giacomo Petrillo
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

import abc

from jax import numpy as jnp
from jax import lax

class SequentialOperation(metaclass=abc.ABCMeta):
    """see jax.lax.fori_loop for semantics"""
    
    @abc.abstractmethod
    def __init__(self, *args): # pragma: no cover
        pass
    
    @property
    @abc.abstractmethod
    def inputs(self): # pragma: no cover
        """tuple of indices of other ops to be used as input"""
        pass
    
    @abc.abstractmethod
    def init_val(self, n, *inputs): # pragma: no cover
        pass
    
    @abc.abstractmethod
    def iter_out(self, i, val): # pragma: no cover
        """output passed to other ops who request it through `inputs`"""
        pass
    
    @abc.abstractmethod
    def iter(self, i, val, *inputs): # pragma: no cover
        """return updated val"""
        pass

    @abc.abstractmethod
    def finalize_val(self, val): # pragma: no cover
        """return final product"""
        pass

# TODO maybe I can register the class as a pytree with automatic childrening,
# such that I don't need to pass `val` around explicitly.

def sequential_algorithm(n, ops):
    init_val = []
    for op in ops:
        args = (ops[j].iter_out(0, init_val[j]) for j in op.inputs)
        init_val.append(op.init_val(n, *args))
    def body_fun(i, val):
        newval = []
        for op, v in zip(ops, val):
            args = (ops[j].iter_out(i, newval[j]) for j in op.inputs)
            newval.append(op.iter(i, v, *args))
        return newval
    val = lax.fori_loop(1, n, body_fun, init_val)
    # TODO use jax.lax.scan because fori_loop could switch to while_loop
    # if n is abstract, which breaks reverse autodiff. Moreover lax.scan
    # allows loop unrolling, although that's currently not really useful
    # probably since I do large row or matrix operations in each cycle.
    return tuple(op.finalize_val(v) for op, v in zip(ops, val))
    
class Producer(SequentialOperation):
    """produces something at each iteration but no final output"""
    
    def finalize_val(self, val):
        pass

class Consumer(SequentialOperation):
    """produces a final output but no iteration output"""
    
    def iter_out(self, i, val):
        pass

class SingleInput(SequentialOperation):
    
    def __init__(self, input):
        self.input = input
        
    @property
    def inputs(self):
        return (self.input,)

class Stack(Consumer, SingleInput):
    
    """input = an operation producing arrays"""
        
    def init_val(self, n, a0):
        out = jnp.zeros((n,) + a0.shape, a0.dtype)
        return out.at[0, ...].set(a0)
    
    def iter(self, i, out, ai):
        return out.at[i, ...].set(ai)
    
    def finalize_val(self, out):
        """the stacked arrays"""
        return out

class MatMulIterByFull(Consumer, SingleInput):
    
    def __init__(self, input, b):
        """input = an operation producing pieces of left operand (a)
        b = right operand"""
        self.input = input
        b = jnp.asarray(b)
        assert b.ndim in (1, 2)
        vec = b.ndim < 2
        if vec:
            b = b[:, None]
        self.vec = vec
        self.b = b
    
    @abc.abstractmethod
    def init_val(self, n, a0): # pragma: no cover
        return ab, ...
    
    @abc.abstractmethod
    def iter(self, i, val, ai): # pragma: no cover
        return ab, ...
    
    def finalize_val(self, val):
        ab = val[0]
        if self.vec:
            ab = jnp.squeeze(ab, -1)
        return ab

class MatMulRowByFull(MatMulIterByFull):
    
    def init_val(self, n, a0):
        b = self.b
        assert a0.ndim == 1
        assert b.shape[0] == len(a0)
        ab = jnp.empty((n, b.shape[1]), jnp.result_type(a0.dtype, b.dtype))
        ab = ab.at[0, :].set(a0 @ b)
        return ab, b
    
    def iter(self, i, val, ai):
        ab, b = val
        ab = ab.at[i, :].set(ai @ b)
        return ab, b
    
class MatMulColByFull(MatMulIterByFull):

    def init_val(self, n, a0):
        b = self.b
        assert a0.ndim == 1
        assert b.shape[0] == n
        ab = a0[:, None] * b[0, None, :]
        return ab, b
    
    def iter(self, i, val, ai):
        ab, b = val
        ab = ab + ai[:, None] * b[i, None, :]
        return ab, b

class SolveTriLowerColByFull(MatMulIterByFull):
    # x[0] /= a[0, 0]
    # for i in range(1, len(x)):
    #     x[i:] -= x[i - 1] * a[i:, i - 1]
    #     x[i] /= a[i, i]
        
    def init_val(self, n, a0):
        b = self.b
        assert a0.shape == (n,)
        assert b.shape[0] == n
        prevai = a0.at[0].set(0)
        ab = b.at[0, :].divide(a0[0])
        return ab, prevai
    
    def iter(self, i, val, ai):
        ab, prevai = val
        ab = ab - ab[i - 1, :] * prevai[:, None]
        ab = ab.at[i, :].divide(ai[i])
        prevai = ai.at[i].set(0)
        return ab, prevai

class SumLogDiag(Consumer, SingleInput):
    
    """input = operation producing the rows/columns of a square matrix"""
    
    def init_val(self, n, m0):
        assert m0.shape == (n,)
        return jnp.log(m0[0])
    
    def iter(self, i, sld, mi):
        return sld + jnp.log(mi[i])
    
    def finalize_val(self, sld):
        """sum(log(diag(m)))"""
        return sld
