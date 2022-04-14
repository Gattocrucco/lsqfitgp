# lsqfitgp/_linalg.py
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

"""

Decompositions of positive definite matrices. A decomposition object is
initialized with a matrix and then can solve linear systems for that matrix.
These classes never check for infs/nans in the matrices.

Classes
-------
Decomposition
    Abstract base class.
DecompAutograd
    Abstract subclass that adds autograd support.
Diag
    Diagonalization.
EigCutFullRank
    Diagonalization rounding up small eigenvalues.
EigCutLowRank
    Diagonalization removing small eigenvalues.
ReduceRank
    Partial diagonalization with higher eigenvalues only.
Chol
    Cholesky decomposition.
CholReg
    Abstract base class for regularized Cholesky decomposition.
CholMaxEig
    Cholesky regularized using the maximum eigenvalue.
CholGersh
    Cholesky regularized using an estimate of the maximum eigenvalue.
CholToeplitz
    Cholesky decomposition of a Toeplitz matrix regularized using an estimate
    of the maximum eigenvalue.
BlockDecomp
    Decompose a 2x2 block matrix.
BlockDiagDecomp
    Decompose a 2x2 block diagonal matrix.

"""

import abc
import functools

from ._imports import numpy as np
from ._imports import linalg
from ._imports import autograd
from ._imports import sparse

from . import _toeplitz_linalg

# TODO optimize the matrix multiplication with gvars. Use these gvar internals:
# gvar.svec(int size)
# gvar.svec._assign(float[] values, int[] indices)
# gvar.GVar(float mean, svec derivs, smat cov)
# it may require cython to be fast since it's not vectorized

# TODO investigate using the QR decomposition. It's twice as fast as
# diagonalization, it does not require positivity, it allows tweaking the
# generalized eigenvalues (the diagonal of the triangular factor). Does not
# support the methods correlate and decorrelate. Low-rank updates already
# provided by scipy.

# TODO rethink the class hierarchy to move correlate and decorrelate in a
# separate superclass.

def noautograd(x):
    """
    Unpack an autograd numpy array.
    """
    if isinstance(x, np.numpy_boxes.ArrayBox):
        return noautograd(x._value)
    else:
        return x

def asinexact(dtype):
    """
    Return dtype if it is inexact, else float64.
    """
    if np.issubdtype(dtype, np.inexact):
        return dtype
    else:
        return np.float64

class Decomposition(metaclass=abc.ABCMeta):
    """
    
    Abstract base class for positive definite symmetric matrices decomposition.
    
    Methods
    -------
    solve
    quad
    logdet
    correlate
    decorrelate
    inv
    
    Properties
    ----------
    n
    
    """
    
    @abc.abstractmethod
    def __init__(self, K):
        """
        Decompose matrix K.
        """
        pass
        
    @abc.abstractmethod
    def solve(self, b):
        """
        Solve the linear system K @ x = b. `b` can be an array of gvars.
        """
        pass
    
    def quad(self, b, c=None):
        """
        Compute the quadratic form b.T @ inv(K) @ b if c is not specified, else
        b.T @ inv(K) @ c. `b` and `c` can be arrays of gvars.
        """
        if c is None:
            c = b
        return b.T @ self.solve(c)
    
    @abc.abstractmethod
    def logdet(self):
        """
        Compute log(det(K)).
        """
        pass
    
    @abc.abstractmethod
    def correlate(self, b):
        """
        Compute A @ b where K = A @ A.T. If b represents iid variables with
        unitary variance, A @ b has covariance matrix K.
        """
        pass
    
    @abc.abstractmethod
    def decorrelate(self, b):
        """
        Solve A @ x = b, where K = A @ A.T. If b represents variables with
        covariance matrix K, x has identity covariance.
        """
        pass
    
    def inv(self):
        """
        Invert the matrix.
        """
        
        # TODO currently no subclass overrides this, some should probably do.
        
        return self.quad(np.eye(self.n))
    
    @property
    def n(self):
        """
        Return n where the decomposed matrix is n x n.
        """
        return len(self._K)

class DecompAutograd(Decomposition):
    """
    Abstract subclass adding autograd support to subclasses of Decomposition.
    Moreover, it fixes the convention of tensor multiplication by passing
    the concrete class methods only 2d matrices and then reshaping back the
    result.
    """
        
    def __init_subclass__(cls, **kw):
        
        # For __init__ I can't use an _autograd flag like below to avoid double
        # wrapping because the wrapper is called as super().__init__ in
        # subclasses, so I assign self._K *after* calling old__init__.
        
        old__init__ = cls.__init__
        
        @functools.wraps(old__init__)
        def __init__(self, K, **kw):
            old__init__(self, noautograd(K), **kw)
            self._K = K
        
        cls.__init__ = __init__
        
        for name in 'solve', 'quad', 'logdet':
            meth = getattr(cls, name)
            if not hasattr(meth, '_autograd'):
                # print(f'defining {cls.__name__}.{name}')
                newmeth = getattr(DecompAutograd, '_make_' + name)(meth)
                newmeth = functools.wraps(meth)(newmeth)
                if not hasattr(newmeth, '_autograd'):
                    newmeth._autograd = True
                setattr(cls, name, newmeth)
        
        super().__init_subclass__(**kw)

    @staticmethod
    def _make_solve(oldsolve):
        
        @autograd.extend.primitive
        def solve_autograd(self, K, b):
            b2d = b.reshape(b.shape[0], -1)
            return oldsolve(self, b2d).reshape(b.shape)
        
        def solve_vjp_K(ans, self, K, b):
            assert ans.shape == b.shape
            assert b.shape[0] == K.shape[0] == K.shape[1]
            def vjp(g):
                assert g.shape[-len(b.shape):] == b.shape
                g = np.moveaxis(g, -len(b.shape), 0)
                A = solve_autograd(self, K, g)
                B = np.moveaxis(ans, 0, -1)
                AB = np.tensordot(A, B, len(b.shape) - 1)
                AB = np.moveaxis(AB, 0, -2)
                assert AB.shape == g.shape[:-len(b.shape)] + K.shape
                return -AB
            return vjp
        
        def solve_vjp_b(ans, self, K, b):
            assert ans.shape == b.shape
            assert b.shape[0] == K.shape[0] == K.shape[1]
            def vjp(g):
                assert g.shape[-len(b.shape):] == b.shape
                g = np.moveaxis(g, -len(b.shape), 0)
                gj = solve_autograd(self, K, g)
                gj = np.moveaxis(gj, 0, -len(b.shape))
                assert gj.shape == g.shape
                return gj
            return vjp
        
        autograd.extend.defvjp(
            solve_autograd,
            solve_vjp_K,
            solve_vjp_b,
            argnums=[1, 2]
        )
        
        def solve_jvp_K(g, ans, self, K, b):
            assert ans.shape == b.shape
            assert ans.shape[0] == K.shape[0] == K.shape[1]
            assert g.shape[:2] == K.shape
            invKg = solve_autograd(self, K, g)
            ans2D = ans.reshape(ans.shape[0], -1)
            jvp = -np.tensordot(invKg, ans2D, axes=(1, 0))
            jvp = np.moveaxis(jvp, -1, 1)
            return jvp.reshape(ans.shape + g.shape[2:])
        
        def solve_jvp_b(g, ans, self, K, b):
            assert ans.shape == b.shape
            assert b.shape[0] == K.shape[0] == K.shape[1]
            assert g.shape[:len(b.shape)] == b.shape
            return solve_autograd(self, K, g)
        
        autograd.extend.defjvp(
            solve_autograd,
            solve_jvp_K,
            solve_jvp_b,
            argnums=[1, 2]
        )
        
        def solve(self, b):
            return solve_autograd(self, self._K, b)
        
        # solve_autograd is used by logdet_vjp, so I store it here in case
        # logdet but not solve needs wrapping in a subclass
        solve._autograd = solve_autograd
        
        return solve
    
    @staticmethod
    def _make_quad(oldquad):
        
        @autograd.extend.primitive
        def quad_autograd(self, K, b, c):
            b2d = b.reshape(b.shape[0], -1)
            if c is None:
                outshape = b.shape[1:][::-1] + b.shape[1:]
                return oldquad(self, b2d).reshape(outshape)
            else:
                c2d = c.reshape(c.shape[0], -1)
                outshape = b.shape[1:][::-1] + c.shape[1:]
                return oldquad(self, b2d, c2d).reshape(outshape)

        def quad_vjp_K(ans, self, K, b, c):
            assert b.shape[0] == K.shape[0] == K.shape[1]
            bshape = b.shape[1:]
            if c is None:
                cshape = bshape
            else:
                assert c.shape[0] == b.shape[0]
                cshape = c.shape[1:]
            assert ans.shape == tuple(reversed(bshape)) + cshape

            def vjp(g):
                assert g.shape[len(g.shape) - len(ans.shape):] == ans.shape
                
                invKb = self.solve._autograd(self, K, b)
                if c is None:
                    invKc = invKb
                else:
                    invKc = self.solve._autograd(self, K, c)

                axes = 2 * (tuple(range(-len(cshape), 0)),)
                ginvKc = np.tensordot(g, invKc, axes)

                axes = 2 * (tuple(range(-len(bshape) - 1, -1)),)
                ginvKcb = np.tensordot(ginvKc, invKb.T, axes)

                assert ginvKcb.shape == g.shape[:len(g.shape) - len(ans.shape)] + K.shape
                return -ginvKcb
            return vjp

        def quad_vjp_b(ans, self, K, b, c):
            assert b.shape[0] == K.shape[0] == K.shape[1]
            bshape = b.shape[1:]
            if c is None:
                cshape = bshape
            else:
                assert c.shape[0] == b.shape[0]
                cshape = c.shape[1:]
            assert ans.shape == tuple(reversed(bshape)) + cshape

            if c is None:
                def vjp(g):
                    glen = len(g.shape) - len(ans.shape)
                    assert g.shape[glen:] == ans.shape
            
                    invKb = self.solve._autograd(self, K, b)

                    axes = 2 * (range(-len(bshape), 0),)
                    gj1 = np.tensordot(g, invKb, axes)

                    axes = tuple(range(glen))
                    axes += tuple(reversed(range(glen, len(gj1.shape))))
                    gj1 = np.transpose(gj1, axes)
                    assert gj1.shape == g.shape[:glen] + b.shape
                    
                    axes = (
                        tuple(range(-len(bshape) - 1, -2 * len(bshape) - 1, -1)),
                        tuple(range(-len(bshape), 0))
                    )
                    gj2 = np.tensordot(g, invKb, axes)

                    gj2 = np.moveaxis(gj2, -1, -len(bshape) - 1)
                    assert gj2.shape == gj1.shape

                    return gj1 + gj2
            else:
                def vjp(g):
                    glen = len(g.shape) - len(ans.shape)
                    assert g.shape[glen:] == ans.shape
            
                    invKc = self.solve._autograd(self, K, c)

                    axes = 2 * (range(-len(cshape), 0),)
                    gj = np.tensordot(g, invKc, axes)

                    axes = tuple(range(glen))
                    axes += tuple(reversed(range(glen, len(gj.shape))))
                    gj = np.transpose(gj, axes)
                    assert gj.shape == g.shape[:glen] + b.shape
                    return gj
            
            return vjp

        def quad_vjp_c(ans, self, K, b, c):
            assert b.shape[0] == K.shape[0] == K.shape[1]
            assert c.shape[0] == b.shape[0]
            bshape = b.shape[1:]
            cshape = c.shape[1:]
            assert ans.shape == tuple(reversed(bshape)) + cshape

            def vjp(g):
                assert g.shape[len(g.shape) - len(ans.shape):] == ans.shape
            
                invKb = self.solve._autograd(self, K, b)

                axes = (
                    tuple(range(-len(cshape) - 1, -len(cshape) - len(bshape) - 1, -1)),
                    tuple(range(-len(bshape), 0))
                )
                gj = np.tensordot(g, invKb, axes)

                gj = np.moveaxis(gj, -1, -len(cshape) - 1)
                assert gj.shape == g.shape[:len(g.shape) - len(ans.shape)] + c.shape
                return gj
            
            return vjp

        autograd.extend.defvjp(
            quad_autograd,
            quad_vjp_K,
            quad_vjp_b,
            quad_vjp_c,
            argnums=[1, 2, 3]
        )
        
        def quad_jvp_K(g, ans, self, K, b, c):
            if c is None:
                c = b
            bshape = b.shape[1:]
            cshape = c.shape[1:]
            assert ans.shape == tuple(reversed(bshape)) + cshape
            assert g.shape[:2] == K.shape
            g3d = g.reshape(K.shape + (-1,))
            ginvKc = quad_autograd(self, K, g3d, c)
            ginvKc = np.moveaxis(ginvKc, 0, -1)
            ginvKc = ginvKc.reshape(c.shape + g.shape[2:])
            jvp = -quad_autograd(self, K, b, ginvKc)
            assert jvp.shape == ans.shape + g.shape[2:]
            return jvp
        
        def quad_jvp_b(g, ans, self, K, b, c):
            assert g.shape[:len(b.shape)] == b.shape
            g1d = g.reshape(b.shape + (-1,))
            if c is None:
                jvp = quad_autograd(self, K, b, g1d)
                jvpT = np.moveaxis(jvp.T, 0, -1)
                jvp = jvp + jvpT
            else:
                jvp = quad_autograd(self, K, g1d, c)
                jvp = np.moveaxis(jvp, 0, -1)
            return jvp.reshape(ans.shape + g.shape[len(b.shape):])
        
        def quad_jvp_c(g, ans, self, K, b, c):
            assert g.shape[:len(c.shape)] == c.shape
            jvp = quad_autograd(self, K, b, g)
            assert jvp.shape == ans.shape + g.shape[len(c.shape):]
            return jvp
        
        autograd.extend.defjvp(
            quad_autograd,
            quad_jvp_K,
            quad_jvp_b,
            quad_jvp_c,
            argnums=[1, 2, 3]
        )

        def quad(self, b, c=None):
            return quad_autograd(self, self._K, b, c)
        quad._autograd = quad_autograd

        return quad
    
    @staticmethod
    def _make_logdet(oldlogdet):
        
        @autograd.extend.primitive
        def logdet_autograd(self, K):
            return oldlogdet(self)
        
        def logdet_vjp(ans, self, K):
            assert ans.shape == ()
            assert K.shape[0] == K.shape[1]
            def vjp(g):
                invK = self.quad._autograd(self, K, np.eye(len(K)), None)
                return g[..., None, None] * invK
            return vjp
        
        autograd.extend.defvjp(
            logdet_autograd,
            logdet_vjp,
            argnums=[1]
        )
        
        def logdet_jvp(g, ans, self, K):
            assert ans.shape == ()
            assert K.shape[0] == K.shape[1]
            assert g.shape[:2] == K.shape
            return np.trace(self.solve._autograd(self, K, g))
        
        autograd.extend.defjvp(
            logdet_autograd,
            logdet_jvp,
            argnums=[1]
        )
        
        def logdet(self):
            return logdet_autograd(self, self._K)
        
        return logdet
    
class Diag(DecompAutograd):
    """
    Diagonalization.
    """
    
    def __init__(self, K):
        self._w, self._V = linalg.eigh(K, check_finite=False)
    
    def solve(self, b):
        return (self._V / self._w) @ (self._V.T @ b)
    
    def quad(self, b, c=None):
        VTb = self._V.T @ b
        if c is None:
            VTc = VTb
        else:
            VTc = self._V.T @ c
        return (VTb.T / self._w) @ VTc
    
    def logdet(self):
        return np.sum(np.log(self._w))
    
    def correlate(self, b):
        return (self._V * np.sqrt(self._w)) @ b
    
    def decorrelate(self, b):
        return (self._V / np.sqrt(self._w)).T @ b
    
    def _eps(self, eps):
        w = self._w
        if eps is None:
            eps = len(w) * np.finfo(asinexact(w.dtype)).eps
        assert np.isscalar(eps) and 0 <= eps < 1
        return eps * np.max(w)

class EigCutFullRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are set to `eps`, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None):
        super().__init__(K)
        eps = self._eps(eps)
        self._w[self._w < eps] = eps
            
class EigCutLowRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are removed, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None):
        super().__init__(K)
        eps = self._eps(eps)
        subset = slice(np.sum(self._w < eps), None) # w is sorted ascending
        self._w = self._w[subset]
        self._V = self._V[:, subset]
        
class ReduceRank(Diag):
    """
    Keep only the first `rank` higher eigenmodes.
    """
    
    def __init__(self, K, rank=1):
        assert isinstance(rank, (int, np.integer)) and rank >= 1
        self._w, self._V = sparse.linalg.eigsh(K, k=rank, which='LM')
    
    def correlate(self, b):
        return super().correlate(b[:len(self._w)])

def solve_triangular(a, b, lower=False):
    """
    Pure python implementation of scipy.linalg.solve_triangular for when
    a or b are object arrays. Differently from the scipy version, it
    satisfies tensordot(a, solve_triangular(a, b), 1) == b instead of
    a @ solve_triangular(a, b) == b. It makes a difference only if b is >2D.
    """
    # TODO maybe commit this to gvar.linalg
    # TODO can I raise a LinAlgError if a[i,i] is 0, and still return the
    # result and have it assigned to a variable using try...finally inside this
    # function?
    x = np.copy(b)
    a = a.reshape(a.shape + (1,) * len(x.shape[1:]))
    if lower:
        x[0] /= a[0, 0]
        for i in range(1, len(x)):
            x[i:] -= x[i - 1] * a[i:, i - 1]
            x[i] /= a[i, i]
    else:
        x[-1] /= a[-1, -1]
        for i in range(len(x) - 1, 0, -1):
            x[:i] -= x[i] * a[:i, i]
            x[i - 1] /= a[i - 1, i - 1]
    return x

def solve_triangular_auto(a, b, lower=False):
    """Works with b both object and non-object array"""
    if b.dtype == object:
        return solve_triangular(a, b, lower=lower)
    else:
        return linalg.solve_triangular(a, b, lower=lower, check_finite=False)

class Chol(DecompAutograd):
    """
    Cholesky decomposition.
    """
    
    def __init__(self, K):
        self._L = linalg.cholesky(K, lower=True, check_finite=False)
    
    def solve(self, b):
        invLb = solve_triangular_auto(self._L, b, lower=True)
        return solve_triangular_auto(self._L.T, invLb, lower=False)
    
    def quad(self, b, c=None):
        invLb = solve_triangular_auto(self._L, b, lower=True)
        if c is None:
            invLc = invLb
        else:
            invLc = solve_triangular_auto(self._L, c, lower=True)
        return invLb.T @ invLc
    
    def logdet(self):
        return 2 * np.sum(np.log(np.diag(self._L)))
    
    def correlate(self, b):
        return self._L @ b
    
    def decorrelate(self, b):
        return solve_triangular_auto(self._L, b, lower=True)

    def _eps(self, eps, mat, maxeigv):
        if eps is None:
            eps = len(mat) * np.finfo(asinexact(mat.dtype)).eps
        assert np.isscalar(eps) and 0 <= eps < 1
        return eps * maxeigv

def _scale(a):
    """
    Compute a vector s of powers of 2 such that diag(a / outer(s, s)) ~ 1.
    """
    return np.exp2(np.rint(0.5 * np.log2(np.diag(a))))

class CholReg(Chol):
    """
    Cholesky decomposition correcting for roundoff. Abstract class.
    """
    
    def __init__(self, K, eps=None):
        s = _scale(K)
        J = K / s[:, None]
        J /= s[None, :]
        self._regularize(J, eps)
        super().__init__(J)
        self._L *= s[:, None]
    
    @abc.abstractmethod
    def _regularize(self, mat, eps):
        """Modify mat in-place to make it positive definite."""
        pass
    
class CholMaxEig(CholReg):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff by
    adding to the diagonal a small number relative to the maximum eigenvalue.
    `eps` multiplies this number.
    """
    
    def _regularize(self, mat, eps):
        w = sparse.linalg.eigsh(mat, k=1, which='LM', return_eigenvectors=False)
        mat[np.diag_indices(len(mat))] += self._eps(eps, mat, w[0])

def _gershgorin_eigval_bound(mat):
    """
    Upper bound on the largest magnitude eigenvalue of the matrix.
    """
    return np.max(np.sum(np.abs(mat), axis=1))

class CholGersh(CholReg):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff
    by adding to the diagonal a small number relative to the maximum eigenvalue.
    `eps` multiplies this number. The maximum eigenvalue is estimated
    with the Gershgorin theorem.
    """
    
    def _regularize(self, mat, eps):
        maxeigv = _gershgorin_eigval_bound(mat)
        mat[np.diag_indices(len(mat))] += self._eps(eps, mat, maxeigv)

class CholToeplitz(Chol):
    """
    Cholesky decomposition of a Toeplitz matrix. Only the first row of the
    matrix is read.
    """
    
    def __init__(self, K, eps=None):
        t = K[0]
        m = _toeplitz_linalg.eigv_bound(t)
        eps = self._eps(eps, t, m)
        self._L = _toeplitz_linalg.cholesky(t, diageps=eps)

class BlockDecomp(Decomposition):
    """
    Decomposition of a 2x2 symmetric block matrix using decompositions of the
    diagonal blocks.
    
    Reference: Gaussian Processes for Machine Learning, A.3, p. 201.
    """
    
    # This class can be used only starting from a seed block and adding
    # other blocks one at a time. Would a divide et impera approach be useful
    # for my case?
    
    def __init__(self, P_decomp, S, Q, S_decomp_class):
        """
        The matrix to be decomposed is
        
            K = [[P,   Q]
                 [Q.T, S]]
        
        Parameters
        ----------
        P_decomp : Decomposition
            An instantiated decomposition of P.
        S, Q : matrices
            The other blocks.
        S_decomp_class : subclass of DecompAutograd
            A subclass of DecompAutograd used to decompose S - Q.T P^-1 Q.
        """
        self._Q = Q
        self._invP = P_decomp
        self._tildeS = S_decomp_class(S - P_decomp.quad(Q))
    
    def solve(self, b):
        invP = self._invP
        tildeS = self._tildeS
        Q = self._Q
        f = b[:len(Q)]
        g = b[len(Q):]
        gQTinvPf = g - invP.quad(Q, f)
        y = tildeS.solve(gQTinvPf)
        x = invP.solve(f - tildeS.quad(Q.T, gQTinvPf))
        return np.concatenate([x, y])
    
    def quad(self, b, c=None):
        invP = self._invP
        tildeS = self._tildeS
        Q = self._Q
        f = b[:len(Q)]
        g = b[len(Q):]
        QTinvPf = invP.quad(Q, f)
        if c is None:
            fTinvPQtildeSg = tildeS.quad(QTinvPf, g)
            gtildeSQTinvPf = fTinvPQtildeSg.T if fTinvPQtildeSg.shape else fTinvPQtildeSg
            return invP.quad(f) + tildeS.quad(QTinvPf) - fTinvPQtildeSg - gtildeSQTinvPf + tildeS.quad(g)
        else:
            h = c[:len(Q)]
            i = c[len(Q):]
            QTinvPh = invP.quad(Q, h)
            fTinvPQtildeSi = tildeS.quad(QTinvPf, i)
            gTtildeSQTinvPh = tildeS.quad(g, QTinvPh)
            return invP.quad(f, h) + tildeS.quad(QTinvPf, QTinvPh) - fTinvPQtildeSi - gTtildeSQTinvPh + tildeS.quad(g, i)
    
    def logdet(self):
        return self._invP.logdet() + self._tildeS.logdet()
    
    def correlate(self, b):
        raise NotImplementedError
    
    def decorrelate(self, b):
        raise NotImplementedError
        
    @property
    def n(self):
        return sum(self._Q.shape)

class BlockDiagDecomp(Decomposition):
    
    def __init__(self, A_decomp, B_decomp):
        """
        
        Decomposition of a 2x2 block diagonal matrix.
        
        The matrix is
        
            [[A  0]
             [0  B]]
        
        Parameters
        ----------
        A_decomp, B_decomp: Decomposition
            Instantiated decompositions of A and B.
        
        """
        self._A = A_decomp
        self._B = B_decomp
    
    def solve(self, b):
        A = self._A
        B = self._B
        An = A.n
        f = b[:An]
        g = b[An:]
        return np.concatenate([A.solve(f), B.solve(g)])
    
    def quad(self, b, c=None):
        A = self._A
        B = self._B
        An = A.n
        f = b[:An]
        g = b[An:]
        if c is None:
            return A.quad(f) + B.quad(g)
        else:
            h = c[:An]
            i = c[An:]
            return A.quad(f, h) + B.quad(g, i)
    
    def logdet(self):
        return self._A.logdet() + self._B.logdet()
    
    def correlate(self, b):
        A = self._A
        B = self._B
        An = A.n
        f = b[:An]
        g = b[An:]
        return np.concatenate([A.correlate(f), B.correlate(g)])
    
    def decorrelate(self, b):
        A = self._A
        B = self._B
        An = A.n
        f = b[:An]
        g = b[An:]
        return np.concatenate([A.decorrelate(f), B.decorrelate(g)])

    @property
    def n(self):
        return self._A.n + self._B.n
