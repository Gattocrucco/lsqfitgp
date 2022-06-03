# lsqfitgp/tests/test_linalg.py
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

import sys
import abc
import inspect

import jax
from jax import numpy as jnp
from jax import test_util
import numpy as np
from scipy import linalg, stats
import gvar

import util

sys.path = ['.'] + sys.path
from lsqfitgp import _linalg, _kernels, _toeplitz_linalg

MAXSIZE = 10 + 1

class DecompTestABC(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def decompclass(self):
        pass
            
    def __init_subclass__(cls):
        cls.matjac = jax.jacfwd(cls.mat, 1)
        cls.mathess = jax.jacfwd(cls.matjac, 1)
            
        # wrap all methods to try again in case of failure,
        # to hedge against bad random matrices
        for name, meth in vars(cls).items():
            # do not get all members, or all xfails would be needed to be
            # marked on all subclasses
            if name.startswith('test_'):
                setattr(cls, name, util.tryagain(meth, method=True))
        
class DecompTestBase(DecompTestABC):
    
    def randsize(self):
        return np.random.randint(1, MAXSIZE)
        
    def randsymmat(self, n):
        O = stats.ortho_group.rvs(n) if n > 1 else np.atleast_2d(1)
        eigvals = np.random.uniform(1e-2, 1e2, size=n)
        K = (O * eigvals) @ O.T
        np.testing.assert_allclose(K, K.T)
        return K
    
    def mat(self, s, n):
        x = np.arange(n)
        return jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
    
    def randvec(self, n):
        return np.random.randn(n)
    
    def randmat(self, m, n=None):
        if not n:
            n = self.randsize()
        return np.random.randn(m, n)
    
    def solve(self, K, b):
        return linalg.solve(K, b)
    
    def check_solve(self, bgen, jit=False):
        fun = lambda K, b: self.decompclass(K).solve(b)
        funjit = jax.jit(fun)
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = bgen(len(K))
            result = fun(K, b)
            if jit:
                result2 = funjit(K, b)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-11)
            else:
                sol = self.solve(K, b)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-4)
    
    def check_solve_jac(self, bgen, jacfun, jit=False):
        # TODO sometimes the jacobian of fun is practically zero. Why? This
        # gives problems because it needs an higher absolute tolerance.
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K).solve(b)
        funjac = jacfun(fun)
        funjacjit = jax.jit(funjac, static_argnums=1)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = bgen(n)
            result = funjac(s, n, b)
            if jit:
                result2 = funjacjit(s, n, b)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-8)
            else:
                sol = -self.solve(K.T, self.solve(K, dK).T).T @ b
                np.testing.assert_allclose(result, sol, atol=1e-13, rtol=1e-3)
    
    def test_solve_vec(self):
        self.check_solve(self.randvec)
    
    def test_solve_vec_jac_rev(self):
        self.check_solve_jac(self.randvec, jax.jacrev)

    def test_solve_vec_jac_fwd(self):
        self.check_solve_jac(self.randvec, jax.jacfwd)

    def test_solve_matrix(self):
        self.check_solve(self.randmat)
    
    def test_solve_matrix_jac_rev(self):
        self.check_solve_jac(self.randmat, jax.jacrev)
    
    def test_solve_matrix_jac_fwd(self):
        self.check_solve_jac(self.randmat, jax.jacfwd)

    def test_solve_vec_jit(self):
        self.check_solve(self.randvec, True)
    
    def test_solve_vec_jac_rev_jit(self):
        self.check_solve_jac(self.randvec, jax.jacrev, True)

    def test_solve_vec_jac_fwd_jit(self):
        self.check_solve_jac(self.randvec, jax.jacfwd, True)

    def test_solve_matrix_jit(self):
        self.check_solve(self.randmat, True)
    
    def test_solve_matrix_jac_rev_jit(self):
        self.check_solve_jac(self.randmat, jax.jacrev, True)
    
    def test_solve_matrix_jac_fwd_jit(self):
        self.check_solve_jac(self.randmat, jax.jacfwd, True)

    def solve_matrix_jac_matrix(self, jacfun):
        def fun(s, n, b, A):
            K = self.mat(s, n)
            return A @ self.decompclass(K).solve(b)
        funjac = jacfun(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randmat(n)
            A = self.randmat(n).T
            sol = -A @ self.solve(K.T, self.solve(K, dK).T).T @ b
            result = funjac(s, n, b, A)
            np.testing.assert_allclose(sol, result, rtol=1e-3)
        
    def test_solve_matrix_jac_rev_matrix(self):
        self.solve_matrix_jac_matrix(jax.jacrev)

    def test_solve_matrix_jac_fwd_matrix(self):
        self.solve_matrix_jac_matrix(jax.jacfwd)

    def assert_close_gvar(self, sol, result):
        diff = np.reshape(sol - result, -1)
        
        diffmean = gvar.mean(diff)
        solcov = gvar.evalcov(gvar.svd(sol))
        q = diffmean @ linalg.solve(solcov, diffmean, assume_a='pos')
        # once I got:
        # LinAlgWarning: Ill-conditioned matrix (rcond=5.70425e-17): result may
        # not be accurate.
        np.testing.assert_allclose(q, 0, atol=1e-7)
        
        diffcov = gvar.evalcov(diff)
        solmax = np.max(linalg.eigvalsh(solcov))
        diffmax = np.max(linalg.eigvalsh(diffcov))
        np.testing.assert_allclose(diffmax / solmax, 0, atol=1e-10)
    
    def randvecgvar(self, n):
        mean = self.randvec(n)
        xcov = np.linspace(0, 3, n)
        cov = np.exp(-(xcov.reshape(-1, 1) - xcov.reshape(1, -1)) ** 2)
        return gvar.gvar(mean, cov)
    
    def test_solve_vec_gvar(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvecgvar(n)
            invK = self.solve(K, np.eye(len(K)))
            sol = invK @ b
            result = self.decompclass(K).solve(b)
            self.assert_close_gvar(sol, result)
    
    def test_quad_vec_gvar(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvecgvar(n)
            invK = self.solve(K, np.eye(len(K)))
            sol = b.T @ invK @ b
            result = self.decompclass(K).quad(b)
            self.assert_close_gvar(sol, result)
    
    def test_quad_vec_vec_gvar(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvec(n)
            c = self.randvecgvar(n)
            invK = self.solve(K, np.eye(len(K)))
            sol = b.T @ invK @ c
            result = self.decompclass(K).quad(b, c)
            self.assert_close_gvar(sol, result)

    def test_quad_matrix_vec_gvar(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvec(n * 3).reshape(n, 3)
            c = self.randvecgvar(n)
            invK = self.solve(K, np.eye(len(K)))
            sol = b.T @ invK @ c
            result = self.decompclass(K).quad(b, c)
            self.assert_close_gvar(sol, result)
    
    def quad(self, K, b, c=None):
        if c is None:
            c = b
        return b.T @ self.solve(K, c)
    
    def quadjac(self, dK, K, b, c=None):
        if c is None:
            c = b
        invKb = self.solve(K, b)
        invKc = self.solve(K, c)
        return -invKb.T @ dK @ invKc

    def check_quad(self, bgen, cgen=lambda n: None, jit=False):
        fun = lambda K, b, c: self.decompclass(K).quad(b, c)
        funjit = jax.jit(fun)
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = bgen(len(K))
            c = cgen(len(K))
            result = fun(K, b, c)
            if jit:
                result2 = funjit(K, b, c)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-11)
            else:
                sol = self.quad(K, b, c)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-9)
    
    def check_quad_jac(self, jacfun, bgen, cgen=lambda n: None, jit=False):
        def fun(s, n, b, c):
            K = self.mat(s, n)
            return self.decompclass(K).quad(b, c)
        fungrad = jacfun(fun)
        fungradjit = jax.jit(fungrad, static_argnums=1)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = bgen(n)
            c = cgen(n)
            result = fungrad(s, n, b, c)
            if jit:
                result2 = fungradjit(s, n, b, c)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-7)
            else:
                sol = self.quadjac(dK, K, b, c)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-7)
    
    def test_quad_vec(self):
        self.check_quad(self.randvec)
    
    def test_quad_vec_vec(self):
        self.check_quad(self.randvec, self.randvec)
    
    def test_quad_matrix(self):
        self.check_quad(self.randmat)
    
    def test_quad_matrix_matrix(self):
        self.check_quad(self.randmat, self.randmat)
    
    def test_quad_vec_jit(self):
        self.check_quad(self.randvec, jit=True)
    
    def test_quad_vec_vec_jit(self):
        self.check_quad(self.randvec, self.randvec, True)
    
    def test_quad_matrix_jit(self):
        self.check_quad(self.randmat, jit=True)
    
    def test_quad_matrix_matrix_jit(self):
        self.check_quad(self.randmat, self.randmat, True)
    
    def test_quad_vec_jac_rev(self):
        self.check_quad_jac(jax.jacrev, self.randvec)
    
    def test_quad_vec_jac_fwd(self):
        self.check_quad_jac(jax.jacfwd, self.randvec)
    
    def test_quad_matrix_jac_rev(self):
        self.check_quad_jac(jax.jacrev, self.randmat)
    
    def test_quad_matrix_jac_fwd(self):
        self.check_quad_jac(jax.jacfwd, self.randmat)
    
    def test_quad_matrix_matrix_jac_fwd(self):
        self.check_quad_jac(jax.jacfwd, self.randmat, self.randmat)
    
    def test_quad_vec_jac_rev_jit(self):
        self.check_quad_jac(jax.jacrev, self.randvec, jit=True)
    
    def test_quad_vec_jac_fwd_jit(self):
        self.check_quad_jac(jax.jacfwd, self.randvec, jit=True)
    
    def test_quad_matrix_jac_rev_jit(self):
        self.check_quad_jac(jax.jacrev, self.randmat, jit=True)
    
    def test_quad_matrix_jac_fwd_jit(self):
        self.check_quad_jac(jax.jacfwd, self.randmat, jit=True)
    
    def test_quad_matrix_matrix_jac_fwd_jit(self):
        self.check_quad_jac(jax.jacfwd, self.randmat, self.randmat, jit=True)

    def logdet(self, K):
        return np.sum(np.log(linalg.eigvalsh(K)))

    def check_logdet(self, jit=False):
        fun = lambda K: self.decompclass(K).logdet()
        funjit = jax.jit(fun)
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            result = fun(K)
            if jit:
                result2 = funjit(K)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-14)
            else:
                sol = self.logdet(K)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-10)
    
    def check_logdet_jac(self, jacfun, jit=False, hess=False, num=False, da=False):
        def fun(s, n):
            K = self.mat(s, n)
            return self.decompclass(K, direct_autodiff=da).logdet()
        fungrad = jacfun(fun)
        fungradjit = jax.jit(fungrad, static_argnums=1)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            if num:
                test_util.check_grads(lambda s: fun(s, n), (s,), order=2)
                continue
            result = fungrad(s, n)
            if jit:
                result2 = fungradjit(s, n)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-9)
                continue
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            if not hess:
                sol = np.trace(self.solve(K, dK))
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-4)
                # TODO 1e-4? really? in general probably low tolerances are
                # needed for CholToeplitz and to a lesser extent Chol, not for
                # the diagonalizations
            else:
                d2K = self.mathess(s, n)
                KdK = self.solve(K, dK)
                Kd2K = self.solve(K, d2K)
                sol = -np.trace(KdK @ KdK) + np.trace(Kd2K)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-5)
    
    def test_logdet(self):
        self.check_logdet()
    
    def test_logdet_jac_rev(self):
        self.check_logdet_jac(jax.jacrev)
    
    def test_logdet_jac_fwd(self):
        self.check_logdet_jac(jax.jacfwd)
        
    def test_logdet_hess(self):
        self.check_logdet_jac(lambda f: jax.jacfwd(jax.jacrev(f)), hess=True, da=True)

    def test_logdet_hess_num(self):
        self.check_logdet_jac(lambda f: jax.jacfwd(jax.jacrev(f)), hess=True, num=True, da=True)

    def test_logdet_jit(self):
        self.check_logdet(True)
    
    def test_logdet_jac_rev_jit(self):
        self.check_logdet_jac(jax.jacrev, True)
    
    def test_logdet_jac_fwd_jit(self):
        self.check_logdet_jac(jax.jacfwd, True)

    def test_inv(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            sol = self.solve(K, np.eye(n))
            result = self.decompclass(K).inv()
            np.testing.assert_allclose(sol, result)
        
    # TODO test derivatives w.r.t. b, c

class DecompTestCorr(DecompTestBase):
    """Tests for `correlate` and `decorrelate` are defined in this
    separate class because BlockDecomp once did not have them, now it has
    but I've left the code around"""
    
    def test_correlate_eye(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            A = self.decompclass(K).correlate(np.eye(n))
            Q = A @ A.T
            np.testing.assert_allclose(K, Q)
    
    def test_decorrelate_mat(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randmat(n)
            x = self.decompclass(K).decorrelate(b)
            result = x.T @ x
            sol = self.decompclass(K).quad(b)
            np.testing.assert_allclose(sol, result)

class TestDiag(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.Diag

class TestEigCutFullRank(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.EigCutFullRank

class TestEigCutLowRank(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.EigCutLowRank

class TestSVDCutFullRank(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.SVDCutFullRank

class TestSVDCutLowRank(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.SVDCutLowRank

class TestChol(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.Chol

class TestCholGersh(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.CholGersh

class ToeplitzBase(DecompTestCorr):

    def randsymmat(self, n):
        x = np.arange(n)
        alpha = np.exp(1/3 * np.random.randn())
        scale = np.exp(1/3 * np.random.randn())
        kernel = _kernels.RatQuad(scale=scale, alpha=alpha)
        return kernel(x[None, :], x[:, None])
    
    def mat(self, s, n):
        x = np.arange(n)
        return jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)

class TestCholToeplitz(ToeplitzBase):
    
    @property
    def decompclass(self):
        return _linalg.CholToeplitz
    
class TestCholToeplitzML(ToeplitzBase):
    
    @property
    def decompclass(self):
        return _linalg.CholToeplitzML    
        
class TestReduceRank(DecompTestCorr):
    
    def rank(self, n):
        if not hasattr(self, 'ranks'):
            self.ranks = dict()
            # do not use __init__ or __new__ because they shoo away pytest
        return self.ranks.setdefault(n, np.random.randint(1, n + 1))
    
    @property
    def decompclass(self):
        return lambda K, **kw: _linalg.ReduceRank(K, rank=self.rank(len(K)), **kw)
    
    def solve(self, K, b):
        invK, rank = linalg.pinv(K, return_rank=True)
        assert rank == self.rank(len(K))
        return invK @ b
    
    def randsymmat(self, n=None):
        if not n:
            n = self.randsize()
        rank = self.rank(n)
        O = stats.ortho_group.rvs(n) if n > 1 else np.atleast_2d(1)
        eigvals = np.random.uniform(1e-2, 1e2, size=rank)
        O = O[:, :rank]
        K = (O * eigvals) @ O.T
        np.testing.assert_allclose(K, K.T)
        return K
    
    def mat(self, s, n=None):
        if not n:
            n = self.randsize()
        rank = self.rank(n)
        x = np.arange(n)
        x[:n - rank + 1] = x[0]
        return jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
        
    def logdet(self, K):
        return np.sum(np.log(np.sort(linalg.eigvalsh(K))[-self.rank(len(K)):]))
        
class BlockDecompTestBase(DecompTestCorr):
    """
    Abstract class for testing BlockDecomp. Concrete subclasses must
    overwrite `subdecompclass`.
    """
    
    @property
    @abc.abstractmethod
    def subdecompclass(self):
        pass
            
    @property
    def decompclass(self):
        def decomp(K, **kw):
            if len(K) == 1:
                return self.subdecompclass(K, **kw)
            p = np.random.randint(1, len(K))
            P = K[:p, :p]
            Q = K[:p, p:]
            S = K[p:, p:]
            args = (self.subdecompclass(P, **kw), S, Q, lambda K, **kw: self.subdecompclass(K, **kw))
            return _linalg.BlockDecomp(*args)
        return decomp
    
class TestBlockChol(BlockDecompTestBase):
    
    @property
    def subdecompclass(self):
        return _linalg.Chol

class TestBlockDiag(BlockDecompTestBase):
    
    @property
    def subdecompclass(self):
        return _linalg.Diag

class BlockDiagDecompTestBase(DecompTestCorr):
    """
    Abstract class for testing BlockDiagDecomp. Concrete subclasses must
    overwrite `subdecompclass`.
    """
    
    @property
    @abc.abstractmethod
    def subdecompclass(self):
        pass
    
    def randsymmat(self, n):
        p = np.random.randint(1, n) if n > 1 else 0
        K = np.zeros((n, n))
        if p > 0:
            K[:p, :p] = super().randsymmat(p)
        K[p:, p:] = super().randsymmat(n - p)
        self._p = p
        return K
    
    def mat(self, s, n):
        p = n // 2
        A = super().mat(s, p)
        B = super().mat(s, n - p)
        self._p = p
        return jnp.block([
            [A, jnp.zeros((p, n-p))],
            [jnp.zeros((n-p, p)), B],
        ])
                
    @property
    def decompclass(self):
        def decomp(K, **kw):
            if len(K) == 1:
                return self.subdecompclass(K, **kw)
            p = self._p
            assert p < len(K)
            A = K[:p, :p]
            B = K[p:, p:]
            args = (self.subdecompclass(A, **kw), self.subdecompclass(B, **kw))
            return _linalg.BlockDiagDecomp(*args)
        return decomp

class TestBlockDiagChol(BlockDiagDecompTestBase):
    
    @property
    def subdecompclass(self):
        return _linalg.Chol

class TestBlockDiagDiag(BlockDiagDecompTestBase):
    
    @property
    def subdecompclass(self):
        return _linalg.Diag

@util.tryagain
def test_solve_triangular():
    for n in range(1, MAXSIZE):
        for ndim in range(4):
            for lower in [True, False]:
                tri = np.tril if lower else np.triu
                A = tri(np.random.randn(n, n))
                diag = np.sqrt(np.sum(np.random.randn(n, 2) ** 2, axis=-1) / 2)
                A[np.diag_indices(n)] = diag
                shape = np.random.randint(1, 4, size=ndim)
                B = np.random.randn(*shape[:-1], n, *shape[-1:])
                x = _linalg.solve_triangular(A, B, lower=lower)
                np.testing.assert_allclose(A @ x, B, rtol=1e-11)

@util.tryagain
def test_toeplitz_gershgorin():
    t = np.random.randn(100)
    m = linalg.toeplitz(t)
    b1 = _linalg._gershgorin_eigval_bound(m)
    b2 = _toeplitz_linalg.eigv_bound(t)
    np.testing.assert_allclose(b2, b1, rtol=1e-15)

def check_toeplitz_chol(lower=True, jit=False):
    x = np.linspace(0, 3, 10)
    t = np.exp(-1/2 * x ** 2)
    m = linalg.toeplitz(t)
    
    if jit:
        cholesky = _toeplitz_linalg.cholesky_jit
    else:
        cholesky = _toeplitz_linalg.cholesky
    
    l1 = cholesky(t, lower=lower)
    l2 = linalg.cholesky(m, lower=lower)
    np.testing.assert_allclose(l1, l2, rtol=1e-8)
    
    b = np.random.randn(len(t), 30)
    lb1 = cholesky(t, b, lower=lower)
    lb2 = l2 @ b
    np.testing.assert_allclose(lb1, lb2, rtol=1e-7)
    
    ld1 = cholesky(t, logdet=True)
    _, ld2 = np.linalg.slogdet(l2)
    np.testing.assert_allclose(ld1, ld2, rtol=1e-9)
    
    if lower:
    
        il1 = cholesky(t, inverse=True, lower=lower)
        il2 = linalg.solve_triangular(l2, np.eye(len(t)), lower=lower)
        np.testing.assert_allclose(il1, il2, rtol=1e-6)
    
        ilb1 = cholesky(t, b, inverse=True, lower=lower)
        ilb2 = linalg.solve_triangular(l2, b, lower=lower)
        np.testing.assert_allclose(ilb1, ilb2, rtol=1e-6)

@util.tryagain
def test_toeplitz_chol_lower():
    check_toeplitz_chol()

@util.tryagain
def test_toeplitz_chol_upper():
    check_toeplitz_chol(False)

@util.tryagain
def test_toeplitz_chol_lower_jit():
    check_toeplitz_chol(True, True)

@util.tryagain
def test_toeplitz_chol_upper_jit():
    check_toeplitz_chol(False, True)

#### XFAILS ####
# keep last to avoid hiding them in wrappings

# solve does not support gvar any more, and quad now supports gvar only on
# second argument
util.xfail(DecompTestBase, 'test_solve_vec_gvar')
util.xfail(DecompTestBase, 'test_quad_vec_gvar')

# TODO second derivatives not working
util.xfail(DecompTestBase, 'test_logdet_hess')
util.skip(DecompTestBase, 'test_logdet_hess_num')

# TODO linalg.sparse.eigsh does not have an XLA counterpart, but apparently
# they are now making an effort to add sparse support to jax, let's wait
for name, meth in inspect.getmembers(TestReduceRank, inspect.isfunction):
    if name.endswith('_jit'):
        util.xfail(TestReduceRank, name)

# TODO reverse diff broken because they use quads within other stuff probably.
# Subclassing DecompAutoDiff does not work. Maybe just using quad to compute
# tildeS is too much.
util.xfail(BlockDecompTestBase, 'test_solve_vec_jac_rev')
util.xfail(BlockDecompTestBase, 'test_solve_matrix_jac_rev')
util.xfail(BlockDecompTestBase, 'test_solve_vec_jac_rev_jit')
util.xfail(BlockDecompTestBase, 'test_solve_matrix_jac_rev_jit')
util.xfail(BlockDecompTestBase, 'test_solve_matrix_jac_rev_matrix')
util.xfail(BlockDecompTestBase, 'test_quad_vec_jac_rev')
util.xfail(BlockDecompTestBase, 'test_quad_matrix_jac_rev')
util.xfail(BlockDecompTestBase, 'test_quad_vec_jac_rev_jit')
util.xfail(BlockDecompTestBase, 'test_quad_matrix_jac_rev_jit')
util.xfail(BlockDecompTestBase, 'test_logdet_jac_rev')
util.xfail(BlockDecompTestBase, 'test_logdet_jac_rev_jit')

# TODO why?
util.xfail(BlockDecompTestBase, 'test_logdet_hess_num')

# TODO solve is not implemented by TestCholToeplitzML
for name, meth in inspect.getmembers(TestCholToeplitzML, inspect.isfunction):
    if name.startswith('test_') and ('solve' in name or 'jac' in name or 'gvar' in name):
        util.xfail(TestCholToeplitzML, name)
