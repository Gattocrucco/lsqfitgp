# lsqfitgp/tests/test_linalg.py
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

import sys
import abc

import autograd
from autograd import numpy as np
from scipy import linalg, stats
import gvar
import pytest

sys.path = ['.'] + sys.path
from lsqfitgp import _linalg

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
        assert np.allclose(K, K.T)
        return K
    
    def mat(self, s, n):
        x = np.arange(n)
        return np.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
    
    matjac = autograd.jacobian(mat, 1)
    
    def randvec(self, n):
        return np.random.randn(n)
    
    def randmat(self, m, n=None):
        if not n:
            n = self.randsize()
        return np.random.randn(m, n)
    
    def solve(self, K, b):
        return linalg.solve(K, b)
            
    def test_solve_vec(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvec(len(K))
            sol = self.solve(K, b)
            result = self.decompclass(K).solve(b)
            assert np.allclose(sol, result, rtol=1e-4)
    
    def test_solve_vec_jac(self):
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K).solve(b)
        funjac = autograd.jacobian(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randvec(n)
            sol = -self.solve(K.T, self.solve(K, dK).T).T @ b
            result = funjac(s, n, b)
            assert np.allclose(sol, result, rtol=1e-3)

    def test_solve_vec_jac_fwd(self):
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K).solve(b)
        funjac = autograd.deriv(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randvec(n)
            sol = -self.solve(K.T, self.solve(K, dK).T).T @ b
            result = funjac(s, n, b)
            assert np.allclose(sol, result, rtol=1e-3)

    def test_solve_matrix(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randmat(len(K))
            sol = self.solve(K, b)
            result = self.decompclass(K).solve(b)
            assert np.allclose(sol, result, rtol=1e-4)

    def test_solve_matrix_jac(self):
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K).solve(b)
        funjac = autograd.jacobian(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randmat(n)
            sol = -self.solve(K.T, self.solve(K, dK).T).T @ b
            result = funjac(s, n, b)
            assert np.allclose(sol, result, rtol=1e-3)

    def test_solve_matrix_jac_matrix(self):
        def fun(s, n, b, A):
            K = self.mat(s, n)
            return A @ self.decompclass(K).solve(b)
        funjac = autograd.jacobian(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randmat(n)
            A = self.randmat(n).T
            sol = -A @ self.solve(K.T, self.solve(K, dK).T).T @ b
            result = funjac(s, n, b, A)
            assert np.allclose(sol, result, rtol=1e-3)

    def assert_close_gvar(self, sol, result):
        diff = np.reshape(sol - result, -1)
        
        diffmean = gvar.mean(diff)
        solcov = gvar.evalcov(gvar.svd(sol))
        q = diffmean @ linalg.solve(solcov, diffmean, assume_a='pos')
        assert np.allclose(q, 0, atol=1e-7)
        
        diffcov = gvar.evalcov(diff)
        solmax = np.max(linalg.eigvalsh(solcov))
        diffmax = np.max(linalg.eigvalsh(diffcov))
        assert np.allclose(diffmax / solmax, 0)
    
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
            b = self.randvecgvar(n)
            c = self.randvecgvar(n)
            invK = self.solve(K, np.eye(len(K)))
            sol = b.T @ invK @ c
            result = self.decompclass(K).quad(b, c)
            self.assert_close_gvar(sol, result)

    def test_quad_matrix_vec_gvar(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvecgvar(n * 3).reshape(n, 3)
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
            assert np.allclose(sol, result)
    
    def test_quad_vec_vec(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randvec(len(K))
            c = self.randvec(len(K))
            sol = b.T @ self.solve(K, c)
            result = self.decompclass(K).quad(b, c)
            assert np.allclose(sol, result)
    
    def test_quad_matrix(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randmat(len(K))
            sol = b.T @ self.solve(K, b)
            result = self.decompclass(K).quad(b)
            assert np.allclose(sol, result)
    
    def test_quad_matrix_matrix(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randmat(len(K))
            c = self.randmat(len(K))
            sol = b.T @ self.solve(K, c)
            result = self.decompclass(K).quad(b, c)
            assert np.allclose(sol, result)
            
    def test_quad_vec_grad(self):
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K).quad(b)
        fungrad = autograd.grad(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randvec(n)
            invKb = self.solve(K, b)
            sol = -invKb.T @ dK @ invKb
            result = fungrad(s, n, b)
            assert np.allclose(sol, result, rtol=1e-4)

    def test_quad_matrix_jac(self):
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K).quad(b)
        funjac = autograd.jacobian(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            b = self.randmat(n)
            invKb = self.solve(K, b)
            sol = -invKb.T @ dK @ invKb
            result = funjac(s, n, b)
            assert np.allclose(sol, result, rtol=1e-4)

    def test_logdet(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            sol = np.sum(np.log(linalg.eigvalsh(K)))
            result = self.decompclass(K).logdet()
            assert np.allclose(sol, result)
    
    def test_logdet_grad(self):
        def fun(s, n):
            K = self.mat(s, n)
            return self.decompclass(K).logdet()
        fungrad = autograd.grad(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            sol = np.trace(self.solve(K, dK))
            result = fungrad(s, n)
            assert np.allclose(sol, result, rtol=1e-4)
    
    def test_logdet_grad_fwd(self):
        def fun(s, n):
            K = self.mat(s, n)
            return self.decompclass(K).logdet()
        fungrad = autograd.deriv(fun)
        for n in range(1, MAXSIZE):
            s = np.exp(np.random.uniform(-1, 1))
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            sol = np.trace(self.solve(K, dK))
            result = fungrad(s, n)
            assert np.allclose(sol, result, rtol=1e-4)
    
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
    separate class because BlockDecomp does not have them."""
    
    def test_correlate_eye(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            A = self.decompclass(K).correlate(np.eye(n))
            Q = A @ A.T
            assert np.allclose(K, Q)
    
    def test_decorrelate_mat(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            b = self.randmat(n)
            x = self.decompclass(K).decorrelate(b)
            result = x.T @ x
            sol = self.decompclass(K).quad(b)
            assert np.allclose(sol, result)

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

class TestChol(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.Chol

class TestCholMaxEig(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.CholMaxEig

class TestCholGersh(DecompTestCorr):
    
    @property
    def decompclass(self):
        return _linalg.CholGersh

def _noautograd(x):
    if isinstance(x, np.numpy_boxes.ArrayBox):
        return x._value
    else:
        return x

class TestReduceRank(DecompTestCorr):
    
    @property
    def decompclass(self):
        return lambda K: _linalg.ReduceRank(K, rank=self._rank)
    
    def solve(self, K, b):
        invK, rank = linalg.pinv(K, return_rank=True)
        assert rank == self._rank
        return invK @ b
    
    def randsymmat(self, n=None):
        if not n:
            n = self.randsize()
        self._rank = np.random.randint(1, n + 1)
        O = stats.ortho_group.rvs(n) if n > 1 else np.atleast_2d(1)
        eigvals = np.random.uniform(1e-2, 1e2, size=self._rank)
        O = O[:, :self._rank]
        K = (O * eigvals) @ O.T
        assert np.allclose(K, K.T)
        return K
    
    def mat(self, s, n=None):
        if not isinstance(s, np.numpy_boxes.ArrayBox):
            if not n:
                n = self.randsize()
            self._rank = np.random.randint(1, n + 1)
        x = np.arange(n)
        x[:n - self._rank + 1] = x[0]
        return np.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
    
    matjac = autograd.jacobian(mat, 1)
        
    def test_logdet(self):
        for n in range(1, MAXSIZE):
            K = self.randsymmat(n)
            sol = np.sum(np.log(np.sort(linalg.eigvalsh(K))[-self._rank:]))
            result = self.decompclass(K).logdet()
            assert np.allclose(sol, result)

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
    assert np.allclose(np.tensordot(A, x1, 1), B)
    x2 = _linalg.solve_triangular(A, B, lower=lower)
    assert np.allclose(x1, x2)

class BlockDecompTestBase(DecompTestBase):
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

class BlockDiagDecompTestBase(DecompTestBase):
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
        return np.concatenate([
            np.concatenate([A, np.zeros((p, n-p))], axis=1),
            np.concatenate([np.zeros((n-p, p)), B], axis=1)
        ])
            
    matjac = autograd.jacobian(mat, 1)
    
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
