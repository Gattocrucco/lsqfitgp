# lsqfitgp/_toeplitz_linalg.py
#
# Copyright (c) 2020, Giacomo Petrillo
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

import numpy as np

def cholesky(t, diageps=None):
    """
    
    Cholesky decomposition of a Toeplitz matrix.
    
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
    
    """
    t = np.asanyarray(t)
    assert len(t.shape) == 1
    if len(t) == 0:
        return np.empty((0, 0), dtype=t.dtype)
    if t[0] <= 0:
        msg = '1-th leading minor is not positive definite'
        raise np.linalg.LinAlgError(msg)

    n = len(t)
    L = np.zeros((n, n), order='F', dtype=t.dtype)
    L[:, 0] = t
    if diageps is not None:
        t = L[:, 0]
        t[0] += diageps
    g = np.stack([np.roll(t, 1), t])

    for i in range(1, n):
        assert g[0, i] > 0
        rho = -g[1, i] / g[0, i]
        if np.abs(rho) >= 1:
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise np.linalg.LinAlgError(msg)
        gamma = np.sqrt((1 - rho) * (1 + rho))
        g[:, i:] += g[::-1, i:] * rho
        g[:, i:] /= gamma
        L[i:, i] = g[0, i:]
        g[0, i:] = np.roll(g[0, i:], 1)

    return L

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
    s = np.abs(t)
    c = np.cumsum(s)
    d = c + c[::-1] - s[0]
    return np.max(d)
