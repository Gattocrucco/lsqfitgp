# lsqfitgp/_toeplitz.py
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

# This code originally copied from the TOEPLITZ_CHOLESKY code by John Burkardt,
# released under the LGPL.
# people.sc.fsu.edu/~jburkardt/py_src/toeplitz_cholesky/toeplitz_cholesky.html

import abc

from jax import numpy as jnp
from jax import lax
import jax
import numpy

from . import _patch_jax

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
    return tuple(op.finalize_val(v) for op, v in zip(ops, val))

class SymSchur(SequentialOperation):
    
    def __init__(self, t):
        """t = first row of a symmetric toeplitz matrix"""
        t = jnp.asarray(t)
        assert t.ndim == 1
        # assert t[0] > 0, '1-th leading minor is not positive definite'
        self.t = t
    
    @property
    def inputs(self):
        return ()
    
    def init_val(self, n):
        t = self.t
        assert len(t) == n
        norm = t[0]
        t = t / norm
        g = jnp.stack([t, t])
        return g, jnp.sqrt(norm)
    
    def iter_out(self, i, val):
        """i-th column of cholesky factor L"""
        g, snorm = val
        return g[0, :] * snorm
    
    def iter(self, i, val):
        g, snorm = val
        g = g.at[0, :].set(jnp.roll(g[0, :], 1))
        g = g.at[:, 0].set(0).at[:, i - 1].set(0)
        # assert g[0, i] > 0, 'what??'
        rho = -g[1, i] / g[0, i]
        # assert abs(rho) < 1, f'{i+1}-th leading minor is not positive definite'
        gamma = jnp.sqrt((1 - rho) * (1 + rho))
        g = (g + g[::-1] * rho) / gamma
        return g, snorm
    
    def finalize_val(self, val):
        pass

class Stack(SequentialOperation):
    
    def __init__(self, input):
        """input = an operation producing arrays"""
        self.input = input
        
    @property
    def inputs(self):
        return (self.input,)
        
    def init_val(self, n, a0):
        out = jnp.zeros((n,) + a0.shape, a0.dtype)
        return out.at[0, ...].set(a0)
    
    def iter_out(self, i, out):
        pass
    
    def iter(self, i, out, ai):
        return out.at[i, ...].set(ai)
    
    def finalize_val(self, out):
        """the stacked arrays"""
        return out

class MatMulIterByFull(SequentialOperation):
    
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
    
    @property
    def inputs(self):
        return (self.input,)
    
    @abc.abstractmethod
    def init_val(self, n, a0): # pragma: no cover
        return ab, ...
    
    def iter_out(self, i, val):
        pass
    
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

class SumLogDiag(SequentialOperation):
    
    def __init__(self, input):
        """input = operation producing the rows/columns of a square matrix"""
        self.input = input
    
    @property
    def inputs(self):
        return (self.input,)
    
    def init_val(self, n, m0):
        assert m0.shape == (n,)
        return jnp.log(m0[0])
    
    def iter_out(self, i, sld):
        pass
    
    def iter(self, i, sld, mi):
        return sld + jnp.log(mi[i])
    
    def finalize_val(self, sld):
        """sum(log(diag(m)))"""
        return sld

def cholesky(t, b=None, *, lower=True, inverse=False, logdet=False):
    """
    
    Cholesky decomposition of a positive definite Toeplitz matrix.
    
    Parameters
    ----------
    t : (n,) array
        The first row/column of the matrix.
    b : (n,) or (n, m) array, optional
        If provided, the Cholesky factor is multiplied by this vector/matrix.
    lower : bool
        Compute the lower (True, default) or the upper triangular Cholesky
        factor.
    inverse : bool
        If True, compute the inverse of the Cholesky factor.
    logdet : bool
        If True, ignore all other options and return the logarithm of the
        determinant of the Cholesky factor.
    
    Returns
    -------
    If `b` is None (default):
    
    L : (n, n) array
        Lower triangular matrix such that toeplitz(t) == L @ L.T, or
        upper U = L.T if `lower == False`, or L^-1 if `inverse == True`.
    
    If `b` is an array:
    
    Lb : (n,) or (n, m) array
        The product L @ b, or L^-1 @ b if `inverse == True`, or
        L.T @ b if `lower == False`.
    
    If `logdet` is True:
    
    l : scalar
        log(det(L))
    
    Notes
    -----
    If b is specified, the Cholesky factor is never formed in memory, but it is
    not possible to compute U^-1 or U^-1 @ b directly in this way.
    
    Reference:
        
        Michael Stewart,
        Cholesky factorization of semi-definite Toeplitz matrices.
        Linear Algebra and its Applications,
        Volume 254, pages 497-525, 1997.
    
    """

    # TODO vectorize
    
    t = jnp.asarray(t)

    if logdet:
        op = SumLogDiag(0)
    elif b is None and inverse:
        assert lower
        op = SolveTriLowerColByFull(0, jnp.eye(len(t)))
    elif b is None:
        op = Stack(0)
    elif inverse:
        assert lower
        op = SolveTriLowerColByFull(0, b)
    elif lower:
        op = MatMulColByFull(0, b)
    else:
        op = MatMulRowByFull(0, b)

    _, out = sequential_algorithm(len(t), [SymSchur(t), op])

    if b is None and not inverse and lower:
        out = out.T
    return out

cholesky_jit = jax.jit(cholesky, static_argnames=['lower', 'inverse', 'logdet'])

def chol_solve_numpy(t, b, diageps=None):
    """
    t (..., n)
    b (..., n, m) or (n,)
    t[0] += diageps
    m = toeplitz(t)
    l = chol(m)
    return solve(l, b)
    pure numpy, object arrays supported
    """
        
    t = numpy.copy(t, subok=True)
    n = t.shape[-1]
        
    b = numpy.asanyarray(b)
    vec = b.ndim < 2
    if vec:
        b = b[:, None]
    assert b.shape[-2] == n
    
    if n == 0:
        shape = numpy.broadcast_shapes(t.shape[:-1], b.shape[:-2])
        shape += (n,) if vec else b.shape[-2:]
        dtype = numpy.result_type(t.dtype, b.dtype)
        return numpy.empty(shape, dtype)

    if diageps is not None:
        t[..., 0] += diageps

    if numpy.any(t[..., 0] <= 0):
        msg = '1-th leading minor is not positive definite'
        raise numpy.linalg.LinAlgError(msg)
    
    norm = numpy.copy(t[..., 0, None], subok=True)
    t /= norm
    invLb = numpy.copy(numpy.broadcast_arrays(b, t[..., None])[0], subok=True)
    prevLi = t
    g = numpy.stack([numpy.roll(t, 1, -1), t], -2)
    
    for i in range(1, n):
        
        assert numpy.all(g[..., 0, i] > 0)
        rho = -g[..., 1, i, None, None] / g[..., 0, i, None, None]

        if numpy.any(numpy.abs(rho) >= 1):
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise numpy.linalg.LinAlgError(msg)

        gamma = numpy.sqrt((1 - rho) * (1 + rho))
        g[..., :, i:] += g[..., ::-1, i:] * rho
        g[..., :, i:] /= gamma
        Li = g[..., 0, i:] # i-th column of L from row i
        invLb[..., i:, :] -= invLb[..., i - 1, None, :] * prevLi[..., i:, None]
        invLb[..., i, :] /= Li[..., 0, None]
        prevLi[..., i:] = Li
        g[..., 0, i:] = numpy.roll(g[..., 0, i:], 1, -1)

    invLb /= numpy.sqrt(norm[..., None])
    if vec:
        invLb = numpy.squeeze(invLb, -1)
    return invLb

def eigv_bound(t):
    """
    
    Bound the eigenvalues of a symmetric Toeplitz matrix.
    
    Parameters
    ----------
    t : array
        The first row of the matrix.
    
    Returns
    -------
    m : scalar
        Any eigenvalue `v` of the matrix satisfies `|v| <= m`.
    
    """
    s = jnp.abs(t)
    c = jnp.cumsum(s)
    d = c + c[::-1] - s[0]
    return jnp.max(d)
