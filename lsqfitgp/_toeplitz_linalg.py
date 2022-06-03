# lsqfitgp/_toeplitz_linalg.py
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
import numpy

from . import _patch_jax

def cholesky(t, b=None, *, lower=True, inverse=False, diageps=None):
    """
    
    Cholesky decomposition of a positive definite Toeplitz matrix.
    
    Parameters
    ----------
    t : (n,) array
        The first row/column of the matrix.
    b : (n,) or (n, m) array, optional
        If provided, the cholesky factor is multiplied by this vector/matrix.
    lower : bool
        Compute the lower (True, default) or the upper triangular cholesky
        factor.
    inverse : bool
        If True, compute the inverse of the cholesky factor.
    diageps: float, optional
        The diagonal of the matrix (the first element of t) is increased by
        `diageps` to correct for numerical non-positivity.
    
    Returns
    -------
    If `b` is None (default):
    
    L : (n, n) array
        Lower triangular matrix such that toeplitz(t) == L @ L.T, or
        upper U = L.T.
    
    If `b` is an array:
    
    Lb : (n,) or (n, m) array
        The product L @ b, or L^-1 @ b if `inverse == True`, or
        L.T @ b, L.T^-1 @ b if `lower == False`.
    
    Notes
    -----
    The reference for the algorithm is:
        
        Michael Stewart,
        Cholesky factorization of semi-definite Toeplitz matrices.
        Linear Algebra and its Applications,
        Volume 254, pages 497-525, 1997.
    
    """
    
    # TODO reimplement using jax.lax.scan. Use the y for stacking and the
    # carry for summing. Use lax.fori_loop for summing only.
        
    t = jnp.asarray(t)
    assert len(t.shape) == 1
    n = len(t)
    
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

    if _patch_jax.isconcrete(t) and t[0] <= 0:
        msg = '1-th leading minor is not positive definite'
        raise numpy.linalg.LinAlgError(msg)

    if diageps is not None:
        t = jnp.concatenate([t[:1] + diageps, t[1:]])
    
    if b is None:
        L = jnp.zeros((n, n))
        L = L.at[0, :].set(t)
    elif not inverse:
        if lower:
            Lb = t[:, None] * b[0, None, :]
        else:
            Lb = jnp.zeros(b.shape)
            Lb = Lb.at[0, :].set(t @ b)
    else:
        if lower:
            invLb = b.at[0].divide(t[0])
            prevLi = t
        else:
            pass

    g = jnp.stack([t, t])
    g = g.at[0, 1:].set(g[0, :-1])

    for i in range(1, n):
        if _patch_jax.isconcrete(g):
            assert g[0, i] > 0
        
        rho = -g[1, i] / g[0, i]

        if _patch_jax.isconcrete(rho) and abs(rho) >= 1:
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise numpy.linalg.LinAlgError(msg)

        gamma = jnp.sqrt((1 - rho) * (1 + rho))
        g = g.at[:, i:].add(g[::-1, i:] * rho)
        g = g.at[:, i:].divide(gamma)
        Li = g[0, :] # i-th column of L starting from row i, garbage before
        
        if b is None:
            L = L.at[i, i:].set(Li[i:])
        elif not inverse:
            if lower:
                Lb = Lb.at[i:, :].add(Li[i:, None] * b[i, None, :])
            else:
                Lb = Lb.at[i, :].set(Li[i:] @ b[i:, :])
        else:
            # x[0] /= a[0, 0]
            # for i in range(1, len(x)):
            #     x[i:] -= x[i - 1] * a[i:, i - 1]
            #     x[i] /= a[i, i]
            if lower:
                invLb = invLb.at[i:].add(-invLb[i - 1] * prevLi[i:, None])
                invLb = invLb.at[i].divide(Li[i])
            else:
                pass
            prevLi = Li
            
        g = g.at[0, i + 1:].set(g[0, i:-1])

    if b is None:
        assert L.shape == (n, n)
        if lower:
            L = L.T
        return L
    elif not inverse:
        assert Lb.shape == b.shape
        if vec:
            Lb = jnp.squeeze(Lb, -1)
        return Lb
    else:
        assert invLb.shape == b.shape
        if vec:
            invLb = jnp.squeeze(invLb, -1)
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
