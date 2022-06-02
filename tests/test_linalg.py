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
import functools

import jax
from jax import numpy as jnp
import numpy as np
from scipy import linalg, stats
import gvar
import pytest

import util

sys.path = ['.'] + sys.path
from lsqfitgp import _linalg, _kernels

MAXSIZE = 10 + 1

class DecompTestBase(metaclass=abc.ABCMeta):
    
    @property
    @abc.abstractmethod
    def decompclass(self):
        pass
            
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
    
    matjac = jax.jacobian(mat, 1)
    
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
                np.testing.assert_allclose(result2, result, atol=1e-12, rtol=1e-12)
            else:
                sol = self.solve(K, b)
                np.testing.assert_allclose(result, sol, rtol=1e-4)
    
    def check_solve_jac(self, bgen, jacfun, jit=False):
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
                np.testing.assert_allclose(result2, result, atol=1e-8, rtol=1e-8)
            else:
                sol = -self.solve(K.T, self.solve(K, dK).T).T @ b
                np.testing.assert_allclose(result, sol, atol=1e-8, rtol=1e-3)

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

    def test_quad_vec(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvec(len(K))
            sol = b.T @ self.solve(K, b)
            result = self.decompclass(K).quad(b)
            np.testing.assert_allclose(sol, result, rtol=1e-3)
    
    def test_quad_vec_vec(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvec(len(K))
            c = self.randvec(len(K))
            sol = b.T @ self.solve(K, c)
            result = self.decompclass(K).quad(b, c)
            np.testing.assert_allclose(sol, result)
    
    def test_quad_matrix(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randmat(len(K))
            sol = b.T @ self.solve(K, b)
            result = self.decompclass(K).quad(b)
            np.testing.assert_allclose(sol, result)
    
    def test_quad_matrix_matrix(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randmat(len(K))
            c = self.randmat(len(K))
            sol = b.T @ self.solve(K, c)
            result = self.decompclass(K).quad(b, c)
            np.testing.assert_allclose(sol, result)
    
    def quad_vec_jac(self, jacfun):
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K).quad(b)
        fungrad = jacfun(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randvec(n)
            invKb = self.solve(K, b)
            sol = -invKb.T @ dK @ invKb
            result = fungrad(s, n, b)
            np.testing.assert_allclose(sol, result, rtol=1e-4)
    
    def test_quad_vec_jac_rev(self):
        self.quad_vec_jac(jax.jacrev)
    
    def test_quad_vec_jac_fwd(self):
        self.quad_vec_jac(jax.jacfwd)
    
    def quad_matrix_jac(self, jacfun):
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K).quad(b)
        funjac = jacfun(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randmat(n)
            invKb = self.solve(K, b)
            sol = -invKb.T @ dK @ invKb
            result = funjac(s, n, b)
            np.testing.assert_allclose(sol, result, rtol=1e-4)
    
    def test_quad_matrix_jac_rev(self):
        self.quad_matrix_jac(jax.jacrev)
    
    def test_quad_matrix_jac_fwd(self):
        self.quad_matrix_jac(jax.jacfwd)
    
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
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-15)
            else:
                sol = self.logdet(K)
                np.testing.assert_allclose(result, sol, atol=1e-10, rtol=1e-10)
    
    def check_logdet_jac(self, jacfun, jit=False):
        def fun(s, n):
            K = self.mat(s, n)
            return self.decompclass(K).logdet()
        fungrad = jacfun(fun)
        fungradjit = jax.jit(fungrad, static_argnums=1)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            result = fungrad(s, n)
            if jit:
                result2 = fungradjit(s, n)
                np.testing.assert_allclose(result2, result, atol=1e-12, rtol=1e-10)
            else:
                K = self.mat(s, n)
                dK = self.matjac(s, n)
                sol = np.trace(self.solve(K, dK))
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-4)
    
    def test_logdet(self):
        self.check_logdet()
    
    def test_logdet_jac_rev(self):
        self.check_logdet_jac(jax.jacrev)
    
    def test_logdet_jac_fwd(self):
        self.check_logdet_jac(jax.jacfwd)

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
    
    # TODO test second derivatives
    
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

class TestCholToeplitz(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.CholToeplitz
    
    def randsymmat(self, n):
        x = np.arange(n)
        alpha = np.exp(1/3 * np.random.randn())
        scale = np.exp(1/3 * np.random.randn())
        kernel = _kernels.RatQuad(scale=scale, alpha=alpha)
        return kernel(x[None, :], x[:, None])
    
    def mat(self, s, n):
        x = np.arange(n)
        return jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
    
    matjac = jax.jacobian(mat, 1)

class TestReduceRank(DecompTestCorr):
    
    @property
    def decompclass(self):
        return lambda K: _linalg.ReduceRank(K, rank=self.rank(len(K)))
    
    def solve(self, K, b):
        invK, rank = linalg.pinv(K, return_rank=True)
        assert rank == self.rank(len(K))
        return invK @ b
    
    def rank(self, n):
        return self.ranks.setdefault(n, np.random.randint(1, n + 1))
    
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
    
    matjac = jax.jacobian(mat, 1)
    
    def logdet(self, K):
        return np.sum(np.log(np.sort(linalg.eigvalsh(K))[-self.rank(len(K)):]))
        
for name, meth in inspect.getmembers(TestReduceRank, inspect.isfunction):
    if name.startswith('test_'):
        @functools.wraps(meth)
        def newmeth(self, *args, _meth=meth, **kw):
            self.ranks = dict()
            return _meth(self, *args, **kw)
        setattr(TestReduceRank, name, newmeth)

def test_solve_triangular():
    for n in range(1, MAXSIZE):
        for ndim in range(3):
            for lower in [True, False]:
                tri = np.tril if lower else np.triu
                A = tri(np.random.randn(n, n))
                diag = np.sqrt(np.sum(np.random.randn(n, 2) ** 2, axis=-1) / 2)
                A[np.diag_indices(n)] = diag
                shape = np.random.randint(1, 4, size=ndim)
                B = np.random.randn(n, *shape)
                check_solve_triangular(A, B, lower)

def check_solve_triangular(A, B, lower):
    x1 = linalg.solve_triangular(A, B.reshape(B.shape[0], -1), lower=lower).reshape(B.shape)
    np.testing.assert_allclose(np.tensordot(A, x1, 1), B)
    x2 = _linalg.solve_triangular(A, B, lower=lower)
    np.testing.assert_allclose(x1, x2)

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
        def decomp(K):
            if len(K) == 1:
                return self.subdecompclass(K)
            p = np.random.randint(1, len(K))
            P = K[:p, :p]
            Q = K[:p, p:]
            S = K[p:, p:]
            args = (self.subdecompclass(P), S, Q, self.subdecompclass)
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
            
    matjac = jax.jacobian(mat, 1)
    
    @property
    def decompclass(self):
        def decomp(K):
            if len(K) == 1:
                return self.subdecompclass(K)
            p = self._p
            assert p < len(K)
            A = K[:p, :p]
            B = K[p:, p:]
            args = (self.subdecompclass(A), self.subdecompclass(B))
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

#### XFAILS ####

# solve does not support gvar any more, and quad now supports gvar only on
# second argument
util.xfail(DecompTestBase, 'test_solve_vec_gvar')
util.xfail(DecompTestBase, 'test_quad_vec_gvar')

# TODO quad backward derivatives are broken
util.xfail(DecompTestBase, 'test_quad_vec_jac_rev')
util.xfail(DecompTestBase, 'test_quad_matrix_jac_rev')

# TODO linalg.sparse.eigsh does not have an XLA counterpart, but apparently
# they are now making an effort to add sparse support to jax, let's wait
for name, attr in vars(DecompTestBase).items():
    if callable(attr) and name.endswith('_jit'):
        util.xfail(TestReduceRank, name)

# TODO probably these xfailures would be solved by making BlockDecomp
# a subclass of DecompAutoDiff, the problem must be quad
util.xfail(BlockDecompTestBase, 'test_solve_vec_jac_rev')
util.xfail(BlockDecompTestBase, 'test_solve_vec_jac_rev_jit')
util.xfail(BlockDecompTestBase, 'test_solve_matrix_jac_rev')
util.xfail(BlockDecompTestBase, 'test_solve_matrix_jac_rev_jit')
util.xfail(BlockDecompTestBase, 'test_solve_matrix_jac_rev_matrix')
util.xfail(BlockDecompTestBase, 'test_logdet_jac_rev')
util.xfail(BlockDecompTestBase, 'test_logdet_jac_rev_jit')
