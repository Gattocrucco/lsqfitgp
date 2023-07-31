# lsqfitgp/_linalg/_toeplitz.py
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

import jax
from jax import numpy as jnp
import numpy

from . import _seqalg

class SymSchur(_seqalg.Producer):
    """
    Cholesky decomposition of a symmetric Toeplitz matrix
    
    Adapted from TOEPLITZ_CHOLESKY by John Burkardt (LGPL license)
    http://people.sc.fsu.edu/~jburkardt/py_src/toeplitz_cholesky/toeplitz_cholesky.html
    """
    
    def __init__(self, t):
        """t = first row of the matrix"""
        t = jnp.asarray(t)
        assert t.ndim == 1
        # assert t[0] > 0, '1-th leading minor is not positive definite'
        self.t = t
    
    inputs = ()
    
    def init(self, n):
        t = self.t
        del self.t
        assert len(t) == n
        norm = t[0]
        t = t / norm
        self.g = jnp.stack([t, t])
        self.snorm = jnp.sqrt(norm)
    
    def iter_out(self, i):
        """i-th column of Cholesky factor L"""
        return self.g[0, :] * self.snorm
    
    def iter(self, i):
        g = self.g
        g = g.at[0, :].set(jnp.roll(g[0, :], 1))
        g = g.at[:, 0].set(0).at[:, i - 1].set(0)
        # assert g[0, i] > 0, 'what??'
        rho = -g[1, i] / g[0, i]
        # assert abs(rho) < 1, f'{i+1}-th leading minor is not positive definite'
        gamma = jnp.sqrt((1 - rho) * (1 + rho))
        self.g = (g + g[::-1] * rho) / gamma
    
    # TODO Schur, Levinson, and in general algorithms with triangular matrices
    # are not efficient as I implement them because due to the constraint of
    # fixed size in the jit I can not take advantage of the triangularity and
    # the number of operations doubles. This could be solved with a blocked
    # version which decreases the block size a few times, n blocks brings down
    # the increase factor to 1 + 1/n. The compilation time and code size would
    # be proportional to n. Aligning block size to powers of 2 would help
    # caching the compilation, bringing the compilation time to the normal one
    # after warmup (does jax compilation reuse functions or inline them?)
    #
    # Anyway, this 2 factor is not currently relevant.

class SymLevinson(_seqalg.Producer):
    """
    Cholesky decomposition of the *inverse* of a symmetric Toeplitz matrix
    
    Adapted form SuperGauss/DurbinLevinson.h (GPL license)
    https://cran.r-project.org/package=SuperGauss
    
    Note: Schur should be more accurate than Levinson
    """
    
    def __init__(self, t):
        """t = first row of the matrix"""
        t = jnp.asarray(t, float)
        assert t.ndim == 1
        # assert t[0] > 0, '1-th leading minor is not positive definite'
        self.t = t
    
    inputs = ()
        
    def init(self, n):
        self.phi1 = jnp.zeros(n)
        self.phi2 = jnp.zeros(n)
        self.nu = self.t[0]
        self.tlag = jnp.roll(self.t, -1)
        del self.t
        
    def iter_out(self, i):
        """i-th row of L^-1"""
        return -self.phi2.at[i].set(-1) / jnp.sqrt(self.nu)
        
    def iter(self, i):
        phi1 = self.phi1
        phi2 = self.phi2
        nu = self.nu
        tlag = self.tlag
        
        pi = i - 1
        rp = phi2 @ tlag
        phi1 = phi1.at[pi].set((tlag[pi] - rp) / nu)
        phi1 = phi1 - phi1[pi] * phi2
        # assert abs(phi1[pi]) < 1, f'{i+1}-th leading minor is not positive definite'
        nu = nu * (1 - phi1[pi]) * (1 + phi1[pi])
        phi2 = jnp.roll(phi1[::-1], i)
        
        self.phi1 = phi1
        self.phi2 = phi2
        self.nu = nu

@jax.jit
def chol(t):
    _, out = _seqalg.sequential_algorithm(len(t), [SymSchur(t), _seqalg.Stack(0)])
    return out.T

@jax.jit
def chol_solve(t, *bs):
    ops = [SymSchur(t)] + [_seqalg.SolveTriLowerColByFull(0, b) for b in bs]
    out = _seqalg.sequential_algorithm(len(t), ops)
    return out[1] if len(bs) == 1 else out[1:]

@jax.jit
def chol_matmul(t, b):
    ops = [SymSchur(t), _seqalg.Rows(b), _seqalg.MatMulColByRow(0, 1)]
    _, _, out = _seqalg.sequential_algorithm(len(t), ops)
    return out

@jax.jit
def chol_transp_matmul(t, b):
    ops = [SymSchur(t), _seqalg.MatMulRowByFull(0, b), _seqalg.Stack(1)]
    _, _, out = _seqalg.sequential_algorithm(len(t), ops)
    return out

@jax.jit
def logdet(t):
    _, out = _seqalg.sequential_algorithm(len(t), [SymSchur(t), _seqalg.SumLogDiag(0)])
    return 2 * out

@jax.jit
def solve(t, b):
    ops = [SymLevinson(t), _seqalg.MatMulRowByFull(0, b), _seqalg.MatMulColByRow(0, 1)]
    _, _, out = _seqalg.sequential_algorithm(len(t), ops)
    return out

@jax.jit
def chol_transp_solve(t, b):
    ops = [SymLevinson(t), _seqalg.Rows(b), _seqalg.MatMulColByRow(0, 1)]
    _, _, out = _seqalg.sequential_algorithm(len(t), ops)
    return out

def chol_solve_numpy(t, b, diageps=None):
    """
    
    Solve a linear system for the cholesky factor of a symmetric Toeplitz
    matrix. The algorithm is:
    
    t[0] += diageps
    m = toeplitz(t)
    l = chol(m)
    return solve(l, b)
    
    Numpy object arrays are supported. Broadcasts like matmul.
    
    Parameters
    ----------
    t : (..., n) array
        The first row or column of the matrix.
    b : (..., n, m) or (n,) array
        The right hand side of the linear system.
    diageps : scalar, optional
        Term added to the diagonal elements of the matrix for regularization.

    """
    
    t = numpy.array(t, subok=True)
    n = t.shape[-1]
        
    b = numpy.asanyarray(b)
    vec = b.ndim < 2
    if vec:
        b = b[:, None]
    assert b.shape[-2] == n
    
    t = t.astype(numpy.result_type(t, 0.1), copy=False)
    b = b.astype(numpy.result_type(b, 0.1), copy=False)
    
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
