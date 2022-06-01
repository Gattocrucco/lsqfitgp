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

def cholesky(t, diageps=None):
    """
    
    Cholesky decomposition of a symmetric Toeplitz matrix.
    
    Parameters
    ----------
    t : (N,) array
        The first row/column of the matrix.
    diageps: float, optional
        The diagonal of the matrix (the first element of t) is increased by
        `diageps`.
    
    Returns
    -------
    L : (N, N) array
        Lower triangular matrix such that toeplitz(t) == L @ L.T.
    
    Notes
    -----
    The reference for the algorithm is:
        
        Michael Stewart,
        Cholesky factorization of semi-definite Toeplitz matrices.
        Linear Algebra and its Applications,
        Volume 254, pages 497-525, 1997.
    
    """
    
    # TODO Split the function in an interface and a numba core. Exceptions are
    # raised by the interface, the core returns the last i.
    
    # TODO Make a function to compute L @ M instead of just computing L, and
    # one to do L^-1 @ M. Both matrix multiplication and triangular solve can
    # be done one column at a time.
    
    # TODO Probably if I compile this with numba the code as originally
    # written with explicit indices is faster.
    
    t = jnp.asarray(t)
    assert len(t.shape) == 1
    n = len(t)
    if n == 0:
        return jnp.empty((0, 0))
    if _patch_jax.isconcrete(t) and t[0] <= 0:
        msg = '1-th leading minor is not positive definite'
        raise numpy.linalg.LinAlgError(msg)

    if diageps is not None:
        t = jnp.concatenate([t[:1] + diageps, t[1:]])
    L = t[None, :]
    g = jnp.stack([jnp.roll(t, 1), t])

    for i in range(1, n):
        if _patch_jax.isconcrete(g):
            assert g[0, i] > 0
        rho = -g[1, i] / g[0, i]
        if _patch_jax.isconcrete(rho) and abs(rho) >= 1:
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise numpy.linalg.LinAlgError(msg)
        gamma = jnp.sqrt((1 - rho) * (1 + rho))
        # g[:, i:] += g[::-1, i:] * rho
        # g[:, i:] /= gamma
        g = (g + g[::-1] * rho) / gamma
        # L[i:, i] = g[0, i:]
        newcol = jnp.concatenate([jnp.zeros(i), g[0, i:]])
        L = jnp.concatenate([L, newcol[None, :]])
        # g[0, i:] = jnp.roll(g[0, i:], 1)
        g = jnp.stack([jnp.roll(g[0], 1), g[1]])
    
    assert L.shape == (n, n)
    return L.T

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
