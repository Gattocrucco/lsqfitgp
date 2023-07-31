# lsqfitgp/_linalg/_seqalg.py
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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
from jax import tree_util

from . import _pytree

class SequentialOperation(_pytree.AutoPyTree, metaclass=abc.ABCMeta):
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
    def init(self, n, *inputs): # pragma: no cover
        """called before the cycle starts with the requested inputs"""
        pass
    
    @abc.abstractmethod
    def iter_out(self, i): # pragma: no cover
        """output passed to other ops who request it through `inputs`,
        guaranteed to be called after `init`"""
        pass
    
    @abc.abstractmethod
    def iter(self, i, *inputs): # pragma: no cover
        """update for iteration"""
        pass

    @abc.abstractmethod
    def finalize(self): # pragma: no cover
        """return final product"""
        pass

def sequential_algorithm(n, ops):
    """
    Define and execute a sequential algorithm on matrices.

    Parameters
    ----------
    n : int
        Number of steps of the algorithm.
    ops : list of SequentialOperation
        Instantiated `SequentialOperation`s that represent the algorithm.

    Return
    ------
    results : tuple
        The sequence of final outputs of each operation in `ops`.
    """
    for i, op in enumerate(ops):
        inputs = op.inputs
        if any(j >= i for j in inputs):
            raise ValueError(f'{i}-th operation {op.__class__.__name__} requested inputs {inputs!r} with forward references')
        args = (ops[j].iter_out(0) for j in inputs)
        op.init(n, *args)
    def body_fun(i, ops):
        for op in ops:
            args = (ops[j].iter_out(i) for j in op.inputs)
            op.iter(i, *args)
        return ops
    ops = lax.fori_loop(1, n, body_fun, ops) # TODO convert to lax.scan and unroll?
    return tuple(op.finalize() for op in ops)
    
class Producer(SequentialOperation):
    """produces something at each iteration but no final output"""
    
    def finalize(self):
        pass

class Consumer(SequentialOperation):
    """produces a final output but no iteration output"""
    
    iter_out = NotImplemented

class SingleInput(SequentialOperation):
    
    def __init__(self, input):
        self.inputs = (input,)
        
    inputs = NotImplemented

class Stack(Consumer, SingleInput):
    """input = an operation producing arrays"""
        
    def init(self, n, a0):
        out = jnp.zeros((n,) + a0.shape, a0.dtype)
        self.out = out.at[0, ...].set(a0)
    
    def iter(self, i, ai):
        self.out = self.out.at[i, ...].set(ai)
    
    def finalize(self):
        """the stacked arrays"""
        return self.out

class MatMulIterByFull(Consumer, SingleInput):
    
    def __init__(self, input, b):
        """input = an operation producing pieces of left operand (a)
        b = right operand"""
        self.inputs = (input,)
        b = jnp.asarray(b)
        assert b.ndim in (1, 2)
        vec = b.ndim < 2
        if vec:
            b = b[:, None]
        self.vec = vec
        self.b = b
    
    @abc.abstractmethod
    def init(self, n, a0): # pragma: no cover
        self.ab = ...
    
    @abc.abstractmethod
    def iter(self, i, ai): # pragma: no cover
        self.ab = ...
    
    def finalize(self):
        ab = self.ab
        if self.vec:
            ab = jnp.squeeze(ab, -1)
        return ab

class MatMulRowByFull(Producer, MatMulIterByFull):
    
    def init(self, n, a0):
        assert a0.ndim == 1
        assert self.b.shape[0] == len(a0)
        self.abi = a0 @ self.b
    
    def iter_out(self, i):
        abi = self.abi
        if self.vec:
            abi = jnp.squeeze(abi, -1)
        return abi
    
    def iter(self, i, ai):
        """ i-th row of input @ b """
        self.abi = ai @ self.b
    
class SolveTriLowerColByFull(MatMulIterByFull):
    # x[0] /= a[0, 0]
    # for i in range(1, len(x)):
    #     x[i:] -= x[i - 1] * a[i:, i - 1]
    #     x[i] /= a[i, i]
        
    def init(self, n, a0):
        b = self.b
        del self.b
        assert a0.shape == (n,)
        assert b.shape[0] == n
        self.prevai = a0.at[0].set(0)
        self.ab = b.at[0, :].divide(a0[0])
    
    def iter(self, i, ai):
        ab = self.ab
        ab = ab - ab[i - 1, :] * self.prevai[:, None]
        self.ab = ab.at[i, :].divide(ai[i])
        self.prevai = ai.at[i].set(0)

class Rows(Producer):
    
    def __init__(self, x):
        self.x = x
    
    inputs = ()
    
    def init(self, n):
        pass
        
    def iter_out(self, i):
        return self.x[i, ...]
    
    def iter(self, i):
        pass

class MatMulColByRow(Consumer):
    
    def __init__(self, inputa, inputb):
        self.inputs = (inputa, inputb)
    
    inputs = None
    
    def init(self, n, a0, b0):
        assert a0.ndim == 1 and b0.ndim <= 1
        self.vec = b0.ndim > 0
        if self.vec:
            self.ab = a0[:, None] * b0[None, :]
        else:
            self.ab = a0 * b0
    
    def iter(self, i, ai, bi):
        if self.vec:
            self.ab = self.ab + ai[:, None] * bi[None, :]
        else:
            self.ab = self.ab + ai * bi
    
    def finalize(self):
        return self.ab

class SumLogDiag(Consumer, SingleInput):
    """input = operation producing the rows/columns of a square matrix"""
    
    def init(self, n, m0):
        assert m0.shape == (n,)
        self.sld = jnp.log(m0[0])
    
    def iter(self, i, mi):
        self.sld = self.sld + jnp.log(mi[i])
    
    def finalize(self):
        """sum(log(diag(m)))"""
        return self.sld
