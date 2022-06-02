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

def cholesky(t, b=None, *, inverse=False, diageps=None):
    """
    
    Cholesky decomposition of a positive definite Toeplitz matrix.
    
    Parameters
    ----------
    t : (n,) array
        The first row/column of the matrix.
    b : (n,) or (n, m) array, optional
        If provided, the cholesky factor is multiplied by this vector/matrix.
    inverse : bool
        If True, multiply the inverse of the cholesky factor by `b`.
    diageps: float, optional
        The diagonal of the matrix (the first element of t) is increased by
        `diageps` to correct for numerical non-positivity.
    
    Returns
    -------
    If `b` is None (default):
    
    L : (n, n) array
        Lower triangular matrix such that toeplitz(t) == L @ L.T.
    
    If `b` is an array:
    
    Lb : (n,) or (n, m) array
        The product L @ b, or L^-1 @ b if `inverse == True`.
    
    Notes
    -----
    The reference for the algorithm is:
        
        Michael Stewart,
        Cholesky factorization of semi-definite Toeplitz matrices.
        Linear Algebra and its Applications,
        Volume 254, pages 497-525, 1997.
    
    """
    
    # TODO Split the function in an interface and a numba core. Exceptions are
    # raised by the interface, the core returns the last i. Probably if I
    # compiled this with numba the code as originally written with explicit
    # indices is faster.
    
    # TODO reimplement using jax.lax.scan. Is there something analogous for
    # reduction instead of stacking?
    
    t = jnp.asarray(t)
    assert len(t.shape) == 1
    n = len(t)
    
    if b is None:
        assert not inverse
    else:
        b = jnp.asarray(b)
        assert b.ndim in (1, 2) and b.shape[0] == n
        bshape = b.shape
        if b.ndim == 1:
            b = b[:, None]
    
    if n == 0:
        return jnp.empty((0, 0) if b is None else b.shape)

    if _patch_jax.isconcrete(t) and t[0] <= 0:
        msg = '1-th leading minor is not positive definite'
        raise numpy.linalg.LinAlgError(msg)

    if diageps is not None:
        t = jnp.concatenate([t[:1] + diageps, t[1:]])
    
    if b is None:
        L = t[None, :] # transposed
    elif not inverse:
        Lb = t[:, None] * b[0, None, :]
    else:
        idx = jnp.arange(n)[:, None] # row indices
        invLb = jnp.where(idx, b, b / t[0])
        prevLi = t

    g = jnp.stack([jnp.roll(t, 1), t])

    for i in range(1, n):
        if _patch_jax.isconcrete(g):
            assert g[0, i] > 0
        
        rho = -g[1, i] / g[0, i]

        if _patch_jax.isconcrete(rho) and abs(rho) >= 1:
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise numpy.linalg.LinAlgError(msg)

        gamma = jnp.sqrt((1 - rho) * (1 + rho))
        g = (g + g[::-1] * rho) / gamma
        Li = jnp.concatenate([jnp.zeros(i), g[0, i:]]) # i-th column of L
        g = jnp.stack([jnp.roll(g[0], 1), g[1]])
        
        if b is None:
            L = jnp.concatenate([L, Li[None, :]])
        elif not inverse:
            Lb = Lb + Li[:, None] * b[i, None, :]
        else:
            # x[0] /= a[0, 0]
            # for i in range(1, len(x)):
            #     x[i:] -= x[i - 1] * a[i:, i - 1]
            #     x[i] /= a[i, i]
            invLb = jnp.where(idx < i, invLb, invLb - invLb[i - 1] * prevLi[:, None])
            invLb = jnp.where(idx != i, invLb, invLb / Li[i])
            prevLi = Li
            
    if b is None:
        assert L.shape == (n, n)
        return L.T
    elif not inverse:
        assert Lb.shape == b.shape
        return Lb.reshape(bshape)
    else:
        assert invLb.shape == b.shape
        return invLb.reshape(bshape)

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
