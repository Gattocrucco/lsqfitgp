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

def cholesky(t, b=None, *, lower=True, inverse=False, diageps=None, logdet=False, difft=None):
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
    
    # TODO make this an internal implementation function, and define
    # single-purpose wrappers. Call it 'schur'.
    
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
        init_val = [jnp.log(t[0])]
    elif b is None:
        L = jnp.zeros((n, n))
        L = L.at[0, :].set(t)
        init_val = [L]
    elif not inverse:
        if lower:
            Lb = t[:, None] * b[0, None, :]
        else:
            Lb = jnp.zeros(b.shape)
            Lb = Lb.at[0, :].set(t @ b)
        init_val = [Lb]
    else:
        prevLi = t.at[0].set(0)
        init_val = [b, prevLi] # invLb = b
        
    if difft is None:
        g = jnp.stack([t, t])
        g = g.at[0, :].set(jnp.roll(g[0, :], 1))
        g = g.at[:, 0].set(0)
        init_val += [g]

    else:
        difft = jnp.asarray(difft) / norm
        assert difft.shape == (len(t) - 1,)
        if _patch_jax.isconcrete(t, difft):
            rtol = numpy.finfo(t.dtype).eps * t[0] / (t[0] - t[1])
            numpy.testing.assert_allclose(numpy.diff(t), difft, rtol=rtol, atol=0)
        t1 = jnp.roll(t, 1)
        s = (t1 + t) / 2 # (g[0] + g[1]) / 2
        d = t1.at[1:].set(-difft / 2) # (g[0] - g[1]) / 2
        s = s.at[0].set(0)
        d = d.at[0].set(0)
        init_val += [s, d]
    
    def body_fun(i, val):
        
        if logdet:
            ld = val.pop(0)
        elif b is None:
            L = val.pop(0)
        elif not inverse:
            Lb = val.pop(0)
        else:
            invLb = val.pop(0)
            prevLi = val.pop(0)
        
        if difft is None:
            g = val.pop(0)

            if _patch_jax.isconcrete(g):
                assert g[0, i] > 0
        
            rho = -g[1, i] / g[0, i]
            error = _patch_jax.isconcrete(rho) and abs(rho) >= 1
            gamma = jnp.sqrt((1 - rho) * (1 + rho))
            g = (g + g[::-1] * rho) / gamma
            Li = g[0, :] # i-th column of L
        
        else:
            s = val.pop(0)
            d = val.pop(0)
            si = s[i]
            di = d[i]
            
            if _patch_jax.isconcrete(s, d):
                assert si + di > 0
            
            error = _patch_jax.isconcrete(si, di) and si * di <= 0
            ssd = jnp.sqrt(si / di) # sqrt((1 + rho)/(1 - rho))
            s = s / ssd
            d = d * ssd
            Li = s + d
        
        if error:
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise numpy.linalg.LinAlgError(msg)

        if logdet:
            ld = ld + jnp.log(Li[i])
            val = [ld]
        elif b is None:
            L = L.at[i, :].set(Li)
            val = [L]
        elif not inverse:
            if lower:
                Lb = Lb + Li[:, None] * b[i, None, :]
            else:
                Lb = Lb.at[i, :].set(Li @ b)
            val = [Lb]
        else:
            # x[0] /= a[0, 0]
            # for i in range(1, len(x)):
            #     x[i:] -= x[i - 1] * a[i:, i - 1]
            #     x[i] /= a[i, i]
            invLb = invLb - invLb[i - 1, :] * prevLi[:, None]
            invLb = invLb.at[i].divide(Li[i])
            prevLi = Li.at[i].set(0)
            val = [invLb, prevLi]
        
        if difft is None:
            g = g.at[0, :].set(jnp.roll(g[0, :], 1))
            g = g.at[:, 0].set(0).at[:, i].set(0)
            val += [g]
        else:
            g01 = jnp.roll(s + d, 1)
            g1 = s - d
            s = (g01 + g1) / 2
            d = (g01 - g1) / 2
            s = s.at[0].set(0).at[i].set(0)
            d = d.at[0].set(0).at[i].set(0)
            val += [s, d]
        
        return val
        
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

cholesky_jit = jax.jit(cholesky, static_argnames=['lower', 'inverse', 'logdet'])

def chol_solve(t, b, diageps=None):
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
