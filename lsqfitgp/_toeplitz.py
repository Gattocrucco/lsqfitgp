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

from jax import numpy as jnp
from jax import lax
import jax
import numpy

from . import _patch_jax

def cholesky(t, b=None, *, lower=True, inverse=False, diageps=None, logdet=False):
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
    diageps : float, optional
        The diagonal of the matrix (the first element of t) is increased by
        `diageps` to correct for numerical non-positivity.
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
    n, = t.shape
    
    assert lower or not inverse
    
    vec = False
    if b is not None:
        b = jnp.asarray(b)
        assert b.ndim in (1, 2) and b.shape[0] == n
        vec = b.ndim < 2
    elif inverse:
        b = jnp.eye(n)
    if vec:
        b = b[:, None]
    
    if n == 0:
        return jnp.empty((0, 0) if b is None else b.shape)

    if diageps is not None:
        t = t.at[0].add(diageps)

    if _patch_jax.isconcrete(t) and t[0] <= 0:
        msg = '1-th leading minor is not positive definite'
        raise numpy.linalg.LinAlgError(msg)
    
    norm = t[0]
    t = t / norm
    
    if logdet:
        init_val = (jnp.log(t[0]),)
    elif b is None:
        L = jnp.zeros((n, n))
        L = L.at[0, :].set(t)
        init_val = (L,)
    elif not inverse:
        if lower:
            Lb = t[:, None] * b[0, None, :]
        else:
            Lb = jnp.zeros(b.shape)
            Lb = Lb.at[0, :].set(t @ b)
        init_val = (Lb,)
    else:
        prevLi = t.at[0].set(0)
        init_val = (b, prevLi) # invLb = b

    g = jnp.stack([t, t])
    g = g.at[0, :].set(jnp.roll(g[0, :], 1))
    g = g.at[:, 0].set(0)
    
    init_val += (g,)
    
    def body_fun(i, val):
        
        g = val[-1]
        if logdet:
            ld = val[0]
        elif b is None:
            L = val[0]
        elif not inverse:
            Lb = val[0]
        else:
            invLb, prevLi = val[:2]
        
        if _patch_jax.isconcrete(g):
            assert g[0, i] > 0
        
        rho = -g[1, i] / g[0, i]

        if _patch_jax.isconcrete(rho) and abs(rho) >= 1:
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise numpy.linalg.LinAlgError(msg)

        gamma = jnp.sqrt((1 - rho) * (1 + rho))
        g = (g + g[::-1] * rho) / gamma
        Li = g[0, :] # i-th column of L
        
        if logdet:
            ld = ld + jnp.log(Li[i])
            val = (ld,)
        elif b is None:
            L = L.at[i, :].set(Li)
            val = (L,)
        elif not inverse:
            if lower:
                Lb = Lb + Li[:, None] * b[i, None, :]
            else:
                Lb = Lb.at[i, :].set(Li @ b)
            val = (Lb,)
        else:
            # x[0] /= a[0, 0]
            # for i in range(1, len(x)):
            #     x[i:] -= x[i - 1] * a[i:, i - 1]
            #     x[i] /= a[i, i]
            invLb = invLb - invLb[i - 1, :] * prevLi[:, None]
            invLb = invLb.at[i].divide(Li[i])
            prevLi = Li.at[i].set(0)
            val = (invLb, prevLi)
            
        g = g.at[0, :].set(jnp.roll(g[0, :], 1))
        g = g.at[:, 0].set(0)
        g = g.at[:, i].set(0)
        
        return val + (g,)
        
    val = lax.fori_loop(1, n, body_fun, init_val)
    
    if logdet:
        ld = val[0]
        return ld + n / 2 * jnp.log(norm)
    if b is None:
        L = val[0]
        assert L.shape == (n, n)
        if lower:
            L = L.T
        return L * jnp.sqrt(norm)
    elif not inverse:
        Lb = val[0]
        assert Lb.shape == b.shape
        if vec:
            Lb = jnp.squeeze(Lb, -1)
        return Lb * jnp.sqrt(norm)
    else:
        invLb = val[0]
        assert invLb.shape == b.shape
        if vec:
            invLb = jnp.squeeze(invLb, -1)
        return invLb / jnp.sqrt(norm)

def chol_solve(t, b, diageps=None):
    """
    t[0] += diageps
    m = toeplitz(t)
    l = chol(m)
    return solve(l, b)
    pure numpy, object arrays supported
    """
    
    # TODO vectorize
    
    t = numpy.copy(t, subok=True)
    n, = t.shape
        
    b = numpy.copy(b, subok=True)
    assert b.ndim in (1, 2) and b.shape[0] == n
    vec = b.ndim < 2
    if vec:
        b = b[:, None]
    
    if n == 0:
        return numpy.empty(b.shape, numpy.result_type(t.dtype, b.dtype))

    if diageps is not None:
        t[0] += diageps

    if t[0] <= 0:
        msg = '1-th leading minor is not positive definite'
        raise numpy.linalg.LinAlgError(msg)
    
    norm = t[0]
    t /= norm
    invLb = b
    prevLi = t
    g = numpy.stack([numpy.roll(t, 1), t])
    
    for i in range(1, n):
        
        assert g[0, i] > 0
        rho = -g[1, i] / g[0, i]

        if abs(rho) >= 1:
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise numpy.linalg.LinAlgError(msg)

        gamma = numpy.sqrt((1 - rho) * (1 + rho))
        g[:, i:] += g[::-1, i:] * rho
        g[:, i:] /= gamma
        Li = g[0, i:] # i-th column of L from row i
        invLb[i:, :] -= invLb[i - 1, None, :] * prevLi[i:, None]
        invLb[i, :] /= Li[0]
        prevLi[i:] = Li
        g[0, i:] = numpy.roll(g[0, i:], 1)

    if vec:
        invLb = numpy.squeeze(invLb, -1)
    invLb /= numpy.sqrt(norm)
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
