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
from pytest import mark

import util

sys.path = ['.'] + sys.path
from lsqfitgp import _linalg, _kernels

# TODO rewrite most comparisons to check for closeness of inputs with inverse
# operation applied to solution in 2-norm instead of comparing solutions
# computed in different ways

rng = np.random.default_rng(202208091144)

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
    """
    
    Restrictions:
    
    - self.decompclass can be applied only on a matrix generated either with
      self.mat or self.randsymmat
    
    - in each method, use only one of self.mat or self.randsymmat
    
    """
    
    sizes = [1, 2, 3, 10]
    
    def randsize(self):
        return rng.integers(1, 11)
        
    def randsymmat(self, n):
        O = stats.ortho_group.rvs(n) if n > 1 else np.atleast_2d(1)
        eigvals = rng.uniform(1e-2, 1e2, size=n)
        K = (O * eigvals) @ O.T
        np.testing.assert_allclose(K, K.T)
        return K
    
    def mat(self, s, n):
        x = np.arange(n)
        return np.pi * jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
    
    def randvec(self, n):
        return rng.standard_normal(n)
    
    def randmat(self, m, n=None):
        if not n:
            n = self.randsize()
        return rng.standard_normal((m, n))
    
    def solve(self, K, b):
        return linalg.solve(K, b)
    
    def check_solve(self, bgen, jit=False):
        fun = lambda K, b: self.decompclass(K).solve(b)
        funjit = jax.jit(fun)
        for n in self.sizes:
            K = self.randsymmat(n)
            b = bgen(len(K))
            result = fun(K, b)
            if jit:
                result2 = funjit(K, b)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-11)
            else:
                sol = self.solve(K, b)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-4)
    
    def check_solve_jac(self, bgen, jacfun, jit=False, hess=False, da=False):
        # TODO sometimes the jacobian of fun is practically zero. Why? This
        # gives problems because it needs an higher absolute tolerance.
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K, direct_autodiff=da).solve(b)
        funjac = jacfun(fun)
        funjacjit = jax.jit(funjac, static_argnums=1)
        for n in self.sizes:
            s = np.exp(rng.uniform(-1, 1))
            b = bgen(n)
            result = funjac(s, n, b)
            if jit:
                result2 = funjacjit(s, n, b)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-8)
                continue
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            KdK = self.solve(K, dK)
            Kb = self.solve(K, b)
            if not hess:
                # -K^-1 dK K^-1 b
                sol = -KdK @ Kb
                np.testing.assert_allclose(result, sol, atol=1e-13, rtol=1e-3)
            else:
                #  K^-1 dK K^-1 dK K^-1 b   +
                # -K^-1 d2K K^-1 b          +
                #  K^-1 dK K^-1 dK K^-1 b
                d2K = self.mathess(s, n)
                sol = 2 * KdK @ KdK @ Kb - self.solve(K, d2K) @ Kb
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
    
    def test_solve_matrix_jac_fwd_da(self):
        self.check_solve_jac(self.randmat, jax.jacfwd, da=True)

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
    
    def test_solve_matrix_hess_fwd_fwd(self):
        self.check_solve_jac(self.randmat, lambda f: jax.jacfwd(jax.jacfwd(f)), hess=True)

    def test_solve_matrix_hess_fwd_rev(self):
        self.check_solve_jac(self.randmat, lambda f: jax.jacfwd(jax.jacrev(f)), hess=True)

    def test_solve_matrix_hess_da(self):
        self.check_solve_jac(self.randmat, lambda f: jax.jacfwd(jax.jacrev(f)), hess=True, da=True)

    def solve_matrix_jac_matrix(self, jacfun):
        def fun(s, n, b, A):
            K = self.mat(s, n)
            return A @ self.decompclass(K).solve(b)
        funjac = jacfun(fun)
        for n in self.sizes:
            s = np.exp(rng.uniform(-1, 1))
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
        
        # TODO to compare matrices, use the 2-norm.
        
        diffcov = gvar.evalcov(diff)
        solmax = np.max(linalg.eigvalsh(solcov))
        diffmax = np.max(linalg.eigvalsh(diffcov))
        np.testing.assert_allclose(diffmax / solmax, 0, atol=1e-10)
    
    def randvecgvar(self, n):
        mean = self.randvec(n)
        xcov = np.linspace(0, 3, n)
        cov = np.exp(-(xcov.reshape(-1, 1) - xcov.reshape(1, -1)) ** 2)
        return gvar.gvar(mean, cov)
    
    def test_quad_vec_vec_gvar(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            b = self.randvec(n)
            c = self.randvecgvar(n)
            invK = self.solve(K, np.eye(len(K)))
            sol = b.T @ invK @ c
            result = self.decompclass(K).quad(b, c)
            self.assert_close_gvar(sol, result)

    def test_quad_matrix_vec_gvar(self):
        for n in self.sizes:
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
    
    def check_quad(self, bgen, cgen=lambda n: None, jit=False):
        fun = lambda K, b, c: self.decompclass(K).quad(b, c)
        funjit = jax.jit(fun)
        for n in self.sizes:
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
    
    def check_quad_jac(self, jacfun, bgen, cgen=lambda n: None, jit=False, hess=False, da=False, stopg=False):
        def fun(s, n, b, c):
            K = self.mat(s, n)
            return self.decompclass(K, direct_autodiff=da, stop_hessian=stopg).quad(b, c)
        fungrad = jacfun(fun)
        fungradjit = jax.jit(fungrad, static_argnums=1)
        for n in self.sizes:
            s = np.exp(rng.uniform(-1, 1))
            b = bgen(n)
            c = cgen(n)
            result = fungrad(s, n, b, c)
            if jit:
                result2 = fungradjit(s, n, b, c)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-7)
                continue
            if c is None:
                c = b
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            KdK = self.solve(K, dK)
            Kc = self.solve(K, c)
            if not hess:
                # b.T K^-1 c
                # -b.T K^-1 dK K^-1 c
                sol = -b.T @ KdK @ Kc
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-6)
            else:
                #  b.T K^-1 dK K^-1 dK K^-1 c   +
                # -b.T K^-1 d2K K^-1 c          +
                #  b.T K^-1 dK K^-1 dK K^-1 c
                sol = 2 * b.T @ KdK @ KdK @ Kc
                if not stopg:
                    d2K = self.mathess(s, n)
                    sol -= b.T @ self.solve(K, d2K) @ Kc
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-5)
    
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
    
    def test_quad_matrix_matrix_jac_fwd_da(self):
        self.check_quad_jac(jax.jacfwd, self.randmat, self.randmat, da=True)
    
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

    def test_quad_matrix_matrix_hess_fwd_rev(self):
        self.check_quad_jac(lambda f: jax.jacfwd(jax.jacrev(f)), self.randmat, self.randmat, hess=True)
    
    def test_quad_matrix_matrix_hess_fwd_fwd(self):
        self.check_quad_jac(lambda f: jax.jacfwd(jax.jacfwd(f)), self.randmat, self.randmat, hess=True)
    
    def test_quad_matrix_matrix_hess_fwd_fwd_stopg(self):
        self.check_quad_jac(lambda f: jax.jacfwd(jax.jacfwd(f)), self.randmat, self.randmat, hess=True, stopg=True)
    
    def test_quad_matrix_matrix_hess_da(self):
        self.check_quad_jac(lambda f: jax.jacfwd(jax.jacrev(f)), self.randmat, self.randmat, hess=True, da=True)
    
    def logdet(self, K):
        return np.sum(np.log(linalg.eigvalsh(K)))

    def check_logdet(self, jit=False):
        fun = lambda K: self.decompclass(K).logdet()
        funjit = jax.jit(fun)
        for n in self.sizes:
            K = self.randsymmat(n)
            result = fun(K)
            if jit:
                result2 = funjit(K)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-14)
            else:
                sol = self.logdet(K)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-10)
    
    def check_logdet_jac(self, jacfun, jit=False, hess=False, num=False, da=False, stopg=False):
        def fun(s, n):
            K = self.mat(s, n)
            return self.decompclass(K, direct_autodiff=da, stop_hessian=stopg).logdet()
        fungrad = jacfun(fun)
        fungradjit = jax.jit(fungrad, static_argnums=1)
        for n in self.sizes:
            s = np.exp(rng.uniform(-1, 1))
            if num:
                order = 2 if hess else 1
                test_util.check_grads(lambda s: fun(s, n), (s,), order=order)
                continue
            result = fungrad(s, n)
            if jit:
                result2 = fungradjit(s, n)
                np.testing.assert_allclose(result2, result, atol=1e-15, rtol=1e-9)
                continue
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            KdK = self.solve(K, dK)
            if not hess:
                # tr(K^-1 dK)
                sol = np.trace(KdK)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-4)
                # TODO 1e-4? really? in general probably low tolerances are
                # needed for CholToeplitz and to a lesser extent Chol, not for
                # the diagonalizations
            else:
                # tr(-K^-1 dK K^-1 dK + K d2K)
                sol = -np.trace(KdK @ KdK)
                if not stopg:
                    d2K = self.mathess(s, n)
                    Kd2K = self.solve(K, d2K)
                    sol += np.trace(Kd2K)
                np.testing.assert_allclose(result, sol, atol=1e-15, rtol=1e-5)
    
    def test_logdet(self):
        self.check_logdet()
    
    def test_logdet_jac_rev(self):
        self.check_logdet_jac(jax.jacrev)
    
    def test_logdet_jac_fwd(self):
        self.check_logdet_jac(jax.jacfwd)
        
    def test_logdet_hess_fwd_fwd(self):
        self.check_logdet_jac(lambda f: jax.jacfwd(jax.jacfwd(f)), hess=True)

    def test_logdet_hess_fwd_fwd_stopg(self):
        self.check_logdet_jac(lambda f: jax.jacfwd(jax.jacfwd(f)), hess=True, stopg=True)

    def test_logdet_hess_fwd_rev(self):
        self.check_logdet_jac(lambda f: jax.jacfwd(jax.jacrev(f)), hess=True)

    def test_logdet_hess_da(self):
        self.check_logdet_jac(lambda f: jax.jacfwd(jax.jacrev(f)), hess=True, da=True)

    def test_logdet_jit(self):
        self.check_logdet(True)
    
    def test_logdet_jac_rev_jit(self):
        self.check_logdet_jac(jax.jacrev, True)
    
    def test_logdet_jac_fwd_jit(self):
        self.check_logdet_jac(jax.jacfwd, True)

    def test_inv(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            sol = self.solve(K, np.eye(n))
            result = self.decompclass(K).inv()
            np.testing.assert_allclose(sol, result)
        
    def check_tree_decomp(self, conversion):
        for n in self.sizes:
            K = self.randsymmat(n)
            b = self.randmat(n)
            d1 = self.decompclass(K)
            q1 = d1.quad(b)
            d2 = conversion(d1)
            q2 = d2.quad(b)
            util.assert_equal(q1, q2)
    
    def test_flatten_decomp(self):
        def conversion(d):
            c, a = jax.tree_util.tree_flatten(d)
            return jax.tree_util.tree_unflatten(a, c)
        self.check_tree_decomp(conversion)
    
    def check_decomp_aux(self, op):
        def conversion(d):
            _, d2 = op(lambda x: (x, d), has_aux=True)(0.)
            return d2
        self.check_tree_decomp(conversion)
    
    def test_decomp_aux_jacfwd(self):
        self.check_decomp_aux(jax.jacfwd)
    
    def test_decomp_aux_jacrev(self):
        self.check_decomp_aux(jax.jacrev)
    
    def test_decomp_aux_grad(self):
        self.check_decomp_aux(jax.grad)
    
    # TODO test derivatives w.r.t. b, c
    # reverse gradient is broken in empbayes_fit, so I envisage that reverse
    # derivatives of quad w.r.t. b and c won't work due to the quad_autodiff
    # in the custom jvp

    def test_correlate_eye(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            A = self.decompclass(K).correlate(np.eye(n))
            Q = A @ A.T
            np.testing.assert_allclose(K, Q)
    
    def test_decorrelate_mat(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            b = self.randmat(n)
            x = self.decompclass(K).decorrelate(b)
            result = x.T @ x
            sol = self.decompclass(K).quad(b)
            np.testing.assert_allclose(sol, result)

class TestDiag(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.Diag

class TestEigCutFullRank(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.EigCutFullRank

class TestEigCutLowRank(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.EigCutLowRank

class TestSVDCutFullRank(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.SVDCutFullRank

class TestSVDCutLowRank(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.SVDCutLowRank

class TestChol(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.Chol

class TestCholGersh(DecompTestBase):
    
    @property
    def decompclass(self):
        return _linalg.CholGersh

class ToeplitzBase(DecompTestBase):

    def randsymmat(self, n):
        x = np.arange(n)
        beta = 2 * np.exp(1/3 * rng.standard_normal())
        scale = np.exp(1/3 * rng.standard_normal())
        kernel = _kernels.Cauchy(scale=scale, beta=beta)
        return np.pi * kernel(x[None, :], x[:, None])
    
    def mat(self, s, n):
        x = np.arange(n)
        return np.pi * jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)

class TestCholToeplitz(ToeplitzBase):
    
    @property
    def decompclass(self):
        return _linalg.CholToeplitz
    
class TestCholToeplitzML(ToeplitzBase):
    
    @property
    def decompclass(self):
        return _linalg.CholToeplitzML    
        
class TestReduceRank(DecompTestBase):
    
    def rank(self, n):
        if not hasattr(self, 'ranks'):
            self.ranks = dict()
            # do not use __init__ or __new__ because they shoo away pytest
        return self.ranks.setdefault(n, rng.integers(1, n + 1))
    
    @property
    def decompclass(self):
        return lambda K, **kw: _linalg.ReduceRank(K, rank=self.rank(len(K)), **kw)
    
    def solve(self, K, b):
        invK, rank = linalg.pinv(K, return_rank=True)
        assert rank == self.rank(len(K))
        return invK @ b
    
    def randsymmat(self, n):
        rank = self.rank(n)
        O = stats.ortho_group.rvs(n) if n > 1 else np.atleast_2d(1)
        eigvals = rng.uniform(1e-2, 1e2, size=rank)
        O = O[:, :rank]
        K = (O * eigvals) @ O.T
        np.testing.assert_allclose(K, K.T)
        return K
    
    def mat(self, s, n):
        rank = self.rank(n)
        x = np.arange(n)
        x[:n - rank + 1] = x[0]
        return np.pi * jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
        
    def logdet(self, K):
        return np.sum(np.log(np.sort(linalg.eigvalsh(K))[-self.rank(len(K)):]))
        
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
        def decomp(K, **kw):
            if len(K) == 1:
                return self.subdecompclass(K, **kw)
            p = rng.integers(1, len(K))
            P = K[:p, :p]
            Q = K[:p, p:]
            S = K[p:, p:]
            Pdec = self.subdecompclass(P, **kw)
            subdec = lambda K, **kw: self.subdecompclass(K, **kw)
            return _linalg.BlockDecomp(Pdec, S, Q, subdec, **kw)
        return decomp
    
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
        p = rng.integers(1, n) if n > 1 else 0
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

class TestBlockDiagDiag(BlockDiagDecompTestBase):
    
    @property
    def subdecompclass(self):
        return _linalg.Diag

class SandwichTestBase(DecompTestBase):
    """
    Abstract base class to test Sandwich* decompositions
    """
    
    @property
    @abc.abstractmethod
    def subdecompclass(self):
        """ class to decompose the inner matrix """
        pass
    
    @property
    @abc.abstractmethod
    def sandwichclass(self):
        """ main decomposition class """
        pass
    
    def _init(self):
        # do not use __init__ or __new__ because they shoo away pytest, even
        # if defined in a superclass
        if not hasattr(self, 'ranks'):
            self.ranks = {}
            self.As = {}
            self.Bs = {}
    
    def rank(self, n):
        self._init()
        return self.ranks.setdefault(n, rng.integers(1, n + 1))
    
    def B(self, n):
        self._init()
        return self.Bs.setdefault(n, rng.standard_normal((n, self.rank(n))))
    
    def randA(self, n):
        self._init()
        return self.As.setdefault(n, super().randsymmat(self.rank(n)))
    
    def matA(self, s, n):
        return super().mat(s, self.rank(n))
        
    @property
    def decompclass(self):
        def decomposition(K, **kw):
            if self._afrom == 'randsymmat':
                A = self.As[len(K)]
            elif self._afrom == 'mat':
                A = self.matA(self._sparam, len(K))
            B = self.Bs[len(K)]
            A_decomp = self.subdecompclass(A, **kw)
            return self.sandwichclass(A_decomp, B)
        return decomposition
            
    def randsymmat(self, n):
        A = self.randA(n)
        B = self.B(n)
        K = B @ A @ B.T
        np.testing.assert_allclose(K, K.T)
        self._afrom = 'randsymmat'
        return K
    
    def mat(self, s, n):
        A = self.matA(s, n)
        B = self.B(n)
        K = B @ A @ B.T
        self._afrom = 'mat'
        self._sparam = s
        return K
        
    def solve(self, K, b):
        invK, rank = linalg.pinv(K, return_rank=True)
        assert rank == self.rank(len(K))
        return invK @ b
    
    def logdet(self, K):
        return np.sum(np.log(np.sort(linalg.eigvalsh(K))[-self.rank(len(K)):]))

class TestSandwichQRDiag(SandwichTestBase):
    
    @property
    def subdecompclass(self):
        return _linalg.Diag
    
    @property
    def sandwichclass(self):
        return _linalg.SandwichQR

class TestSandwichSVDDiag(SandwichTestBase):
    
    @property
    def subdecompclass(self):
        return _linalg.Diag
    
    @property
    def sandwichclass(self):
        return _linalg.SandwichSVD

class WoodburyTestBase(DecompTestBase):
    """
    Abstract base class to test Woodbury
    """
    
    @property
    @abc.abstractmethod
    def subdecompclass(self):
        """ class to decompose A, C, and C^-1 - B^T A^-1 B """
        pass
    
    def _init(self):
        # do not use __init__ or __new__ because they shoo away pytest, even
        # if defined in a superclass
        if not hasattr(self, 'ranks'):
            self.ranks = {}
            self.Ms = {}
            self.signs = {}
    
    def rank(self, n):
        self._init()
        return self.ranks.setdefault(n, rng.integers(1, n + 1))
    
    def sign(self, n):
        self._init()
        return self.signs.setdefault(n, -1 + 2 * rng.integers(2))
    
    def randM(self, n):
        self._init()
        return self.Ms.setdefault(n, super().randsymmat(n + self.rank(n)))
    
    def matM(self, s, n):
        return super().mat(s, n + self.rank(n))
    
    def ABC(self, n, M):
        A = M[:n, :n]
        B = M[:n, n:]
        C = M[n:, n:]
        return A, B, C
        
    def randsymmat(self, n):
        A, B, C = self.ABC(n, self.randM(n))
        K = A + self.sign(n) * B @ C @ B.T
        np.testing.assert_allclose(K, K.T)
        self._mfrom = 'randsymmat'
        return K
    
    def mat(self, s, n):
        A, B, C = self.ABC(n, self.matM(s, n))
        K = A + self.sign(n) * B @ C @ B.T
        self._mfrom = 'mat'
        self._sparam = s
        return K
        
    @property
    def decompclass(self):
        def decomposition(K, **kw):
            n = len(K)
            if self._mfrom == 'randsymmat':
                M = self.randM(n)
            elif self._mfrom == 'mat':
                M = self.matM(self._sparam, n)
            A, B, C = self.ABC(n, M)
            A_decomp = self.subdecompclass(A, **kw)
            C_decomp = self.subdecompclass(C, **kw)
            return _linalg.Woodbury(A_decomp, B, C_decomp, self.subdecompclass, sign=self.sign(n), **kw)
        return decomposition

class TestWoodburyDiag(WoodburyTestBase):
    
    @property
    def subdecompclass(self):
        return _linalg.Diag
            
@util.tryagain
def test_solve_triangular():
    for n in DecompTestBase.sizes:
        for ndim in range(4):
            for lower in [True, False]:
                tri = np.tril if lower else np.triu
                A = tri(rng.standard_normal((n, n)))
                diag = np.sqrt(np.sum(rng.standard_normal((n, 2)) ** 2, axis=-1) / 2)
                A[np.diag_indices(n)] = diag
                shape = rng.integers(1, 4, size=ndim)
                B = rng.standard_normal((*shape[:-1], n, *shape[-1:]))
                x = _linalg.solve_triangular(A, B, lower=lower)
                np.testing.assert_allclose(A @ x, B, rtol=1e-10)

@util.tryagain
def test_toeplitz_gershgorin():
    t = rng.standard_normal(100)
    m = linalg.toeplitz(t)
    b1 = _linalg._decomp._gershgorin_eigval_bound(m)
    b2 = _linalg._toeplitz.eigv_bound(t)
    np.testing.assert_allclose(b2, b1, rtol=1e-15)

def check_toeplitz():
    mod = _linalg._toeplitz
    for n in [1, 2, 10]:
        x = np.linspace(0, 3, n)
        t = np.pi * np.exp(-1/2 * x ** 2)
        m = linalg.toeplitz(t)
    
        l1 = mod.chol(t)
        l2 = linalg.cholesky(m, lower=True)
        np.testing.assert_allclose(l1, l2, rtol=1e-8)
    
        b = rng.standard_normal((len(t), 30))
        lb1 = mod.chol_matmul(t, b)
        lb2 = l2 @ b
        np.testing.assert_allclose(lb1, lb2, rtol=1e-7)

        ld1 = mod.logdet(t)
        _, ld2 = np.linalg.slogdet(m)
        np.testing.assert_allclose(ld1, ld2, rtol=1e-9)
    
        ilb1 = mod.chol_solve(t, b)
        ilb2 = linalg.solve_triangular(l2, b, lower=True)
        np.testing.assert_allclose(ilb1, ilb2, rtol=1e-6)
    
        imb1 = mod.solve(t, b)
        imb2 = np.linalg.solve(m, b)
        np.testing.assert_allclose(imb1, imb2, rtol=1e-5)

@util.tryagain
def test_toeplitz_nojit():
    with jax.disable_jit():
        check_toeplitz()

@util.tryagain
def test_toeplitz():
    check_toeplitz()

@util.tryagain
def test_toeplitz_chol_solve_numpy():
    shapes = [
        [(), ()],
        [(10,), (1,)],
        [(1, 2), (3, 1)],
        [(1, 4), (4,)],
        [(3,), (1, 3)],
    ]
    for tshape, bshape in shapes:
        for n in [1, 2, 10]:
            x = np.linspace(0, 3, n)
            gamma = rng.uniform(0, 2, tshape + (1,))
            t = np.pi * np.exp(-1/2 * x ** gamma)
            m = np.empty(tshape + (n, n))
            for i in np.ndindex(*tshape):
                m[i] = linalg.toeplitz(t[i])
            l = np.linalg.cholesky(m)
            for shape in [(), (1,), (2,), (10,)]:
                if bshape and not shape:
                    continue
                b = rng.standard_normal((*bshape, n, *shape))
                ilb = _linalg._toeplitz.chol_solve_numpy(t, b, diageps=1e-12)
                np.testing.assert_allclose(*np.broadcast_arrays(l @ ilb, b), rtol=1e-6)

@mark.parametrize('mode', ['reduced', 'full'])
def test_rq(mode):
    for n in DecompTestBase.sizes:
        a = rng.standard_normal((n, 5))
        r, q = _linalg._decomp._rq(a, mode=mode)
        util.assert_close_matrices(r @ q, a, atol=0, rtol=1e-15)
        util.assert_equal(np.tril(r), r)
        util.assert_close_matrices(  q @ q.T @   q,   q, atol=0, rtol=1e-14)
        util.assert_close_matrices(q.T @   q @ q.T, q.T, atol=0, rtol=1e-14)

@mark.parametrize('decomp', [
    _linalg.CholGersh,
    _linalg.EigCutLowRank,
    _linalg.SVDCutLowRank,
])
def test_degenerate(decomp):
    a = linalg.block_diag(np.eye(5), 0)
    d = decomp(a)
    util.assert_close_matrices(d.quad(a), a, atol=0, rtol=1e-14)

# TODO since the decompositions are pseudoinverses, I should do all tests
# on singular matrices too instead of just here for some of them.

#### XFAILS ####
# keep last to avoid hiding them in wrappings

# TODO second derivatives completing but returning incorrect result
util.xfail(DecompTestBase, 'test_solve_matrix_hess_fwd_rev')
util.xfail(DecompTestBase, 'test_logdet_hess_fwd_rev')
util.xfail(BlockDecompTestBase, 'test_logdet_hess_fwd_fwd_stopg')
util.xfail(BlockDecompTestBase, 'test_quad_matrix_matrix_hess_fwd_fwd_stopg')
util.xfail(WoodburyTestBase, 'test_logdet_hess_fwd_fwd_stopg')
util.xfail(WoodburyTestBase, 'test_quad_matrix_matrix_hess_fwd_fwd_stopg')

# TODO linalg.sparse.eigsh does not have a jax counterpart, but apparently
# they are now making an effort to add sparse support to jax, let's wait
for name, meth in inspect.getmembers(TestReduceRank, inspect.isfunction):
    if name.endswith('_jit') or name.endswith('_da'):
        util.xfail(TestReduceRank, name)

# TODO reverse diff broken because they use quads within other stuff probably.
# Subclassing DecompAutoDiff does not work. Maybe just using quad to compute
# tildeS is too much. The error is two undefined primals in a matrix
# multiplication jvp transpose. => it's probably derivatives w.r.t. the outer
# sides of quad
for cls in [BlockDecompTestBase, WoodburyTestBase]:
    util.xfail(cls, 'test_solve_vec_jac_rev')
    util.xfail(cls, 'test_solve_matrix_jac_rev')
    util.xfail(cls, 'test_solve_vec_jac_rev_jit')
    util.xfail(cls, 'test_solve_matrix_jac_rev_jit')
    util.xfail(cls, 'test_solve_matrix_jac_rev_matrix')
    util.xfail(cls, 'test_quad_vec_jac_rev')
    util.xfail(cls, 'test_quad_matrix_jac_rev')
    util.xfail(cls, 'test_quad_vec_jac_rev_jit')
    util.xfail(cls, 'test_quad_matrix_jac_rev_jit')
    util.xfail(cls, 'test_logdet_jac_rev')
    util.xfail(cls, 'test_logdet_jac_rev_jit')
    util.xfail(cls, 'test_quad_matrix_matrix_hess_fwd_rev')

# TODO I don't know how to implement correlate and decorrelate for Woodbury
util.xfail(WoodburyTestBase, 'test_correlate_eye')
util.xfail(WoodburyTestBase, 'test_decorrelate_mat')

# TODO these work but are quite inaccurate so they can fail
# util.xfail(BlockDecompTestBase, 'test_logdet_hess_da')
# util.xfail(BlockDecompTestBase, 'test_solve_matrix_hess_da')

# TODO why?
# util.xfail(BlockDecompTestBase, 'test_logdet_hess_num')
