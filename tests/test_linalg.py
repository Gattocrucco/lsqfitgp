# lsqfitgp/tests/test_linalg.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

import re
import sys
import abc
import inspect
import functools

import jax
from jax import numpy as jnp
from jax import test_util
from jax.scipy import linalg as jlinalg
import numpy as np
from scipy import linalg, stats
import gvar
from pytest import mark
import pytest

import util

sys.path = ['.'] + sys.path
from lsqfitgp import _linalg, _kernels

TRYAGAIN = True

# TODO rewrite many comparisons to check for closeness of inputs with inverse
# operation applied to solution in 2-norm instead of comparing solutions
# computed in different ways

# TODO since the decompositions are pseudoinverses, I should do all tests on
# singular matrices too instead of just for some of them in test_degenerate. To
# make a degenerate matrix, multiply the current matrix factories by a random
# singular projector. To do:
#   - fix and document pinv behavior in Decomposition
#   - decide which properties to check (just the first MP identity? All four?)

# TODO remove some redundant tests to speed up the suite (_matrix and _vec
# variants?)

jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

s1, s2 = np.random.SeedSequence(202302061416).spawn(2)
rng = np.random.default_rng(s1)
np.random.seed(s2.generate_state(1))

def randortho(n):
    if n > 1:
        return stats.ortho_group.rvs(n, random_state=rng)
    else:
        # stats.ortho_group does not support n < 2
        return np.atleast_2d(2 * rng.integers(2) - 1)

class DecompTestABC(abc.ABC):

    @property
    @abc.abstractmethod
    def decompclass(self):
        """ Decomposition subclass to be tested """
        pass
    
    @property
    @abc.abstractmethod
    def lowrank(self):
        """ Boolean indicating if the tests matrices are singular """
        pass
        
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        
        cls.matjac = jax.jacfwd(cls.mat, 1)
        cls.mathess = jax.jacfwd(cls.matjac, 1)
            
        # wrap all methods to try again in case of failure, to hedge against bad
        # random matrices. vars() only gives methods overwritten by a subclass,
        # so we do not get all members, such that subclasses inherit xfails,
        # which would be hidden by the decorator
        if TRYAGAIN:
            for name, meth in vars(cls).items():
                if name.startswith('test_'):
                        setattr(cls, name, util.tryagain(meth, method=True))
        
        def setattr_ifmis(cls, name, value):
            prev = getattr(cls, name, None)
            sup = getattr(DecompTestABC, name, prev)
            if not hasattr(cls, name) or sup is prev:
                setattr(cls, name, value)
        
        # configure automatically the subclass if it's still abstract
        name = cls.__name__
        if name.startswith('Test') and inspect.isabstract(cls):
            name = name[4:]
            if name.endswith('_LowRank'):
                name = name[:-8]
                cls.lowrank = True
            else:
                cls.lowrank = False
            if '_' in name:
                comp, name = name.split('_')
                setattr_ifmis(cls, 'compositeclass', getattr(_linalg, comp))
                setattr_ifmis(cls, 'subdecompclass', getattr(_linalg, name))
            else:
                setattr_ifmis(cls, 'decompclass', getattr(_linalg, name))

class DecompTestBase(DecompTestABC):
    """
    
    Restrictions:
    
    - self.decompclass can be applied only on a matrix generated either with
      self.mat or self.randsymmat
    
    - in each test method, use only one of self.mat or self.randsymmat
    
    """
    
    sizes = [10, 3, 2, 1]
    
    def randsize(self):
        return rng.integers(1, 11)
    
    _rank = functools.cached_property(lambda self: {})
    def rank(self, n):
        # cache the rank because it needs to stay consistent between invocations
        # of parametric test matrix constructors
        return self._rank.setdefault(n, 1 + rng.integers(n) if self.lowrank else n)
    
    # TODO unificate randsymmat and mat by generating a cached random orthogonal
    # transformation and using a "pseudocasual" nonlinear function of the
    # parameter for the eigenvalues. To generate a random matrix, pass a random
    # parameter. This simplifies the composite decompositions. randsymmat can
    # become a wrapper of mat. Even better for the composite: generate directly
    # the decomposition, i.e., replace decompclass with decomp(n, s) and
    # randdecomp(n), since now I can get the matrix from the decomposition with
    # .matrix(). => Problem: if I fix the orthogonal transf, ∂K will have the
    # same kernel as K, which is not general. I could add a rotation around a
    # random axis after the orthogonal transformation.
        
    def randsymmat(self, n, *, rank=None):
        """ return random nxn symmetric matrix, well conditioned in the
        orthogonal complement to the null space """
        if rank is None:
            rank = self.rank(n)
        eigvals = rng.uniform(1e-2, 1e2, size=rank)
        O = randortho(n)
        O = O[:, :rank]
        K = (O * eigvals) @ O.T
        util.assert_close_matrices(K, K.T, rtol=1e-15)
        return K
    
    def mat(self, s, n, *, rank=None):
        if rank is None:
            rank = self.rank(n)
        x = np.arange(n)
        x[:n - rank + 1] = x[:1]
        return np.pi * jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
        # the π factor is for tests that pass only if the diagonal is 1
        # TODO add some nontrivial dependence on s when n = 1
    
    def randvec(self, n):
        return rng.standard_normal(n)
    
    def randmat(self, m, n=None):
        if n is None:
            n = self.randsize()
        return rng.standard_normal((m, n))
    
    def quad_or_solve(self, K, b=None, c=None, return_rank=False):
        U, s, Vh = linalg.svd(K)
        eps = len(K) * s[0] * np.finfo(K.dtype).eps
        cond = s > eps
        sinv = np.diag(np.where(cond, 1 / np.where(s, s, 1), 0))
        lpd = np.sum(np.log(s[cond]))
        ld = lpd + np.count_nonzero(~cond) * np.log(eps)
        rank = np.count_nonzero(cond)

        if b is None:
            sol = lpd
        elif c is None:
            sol = Vh.T @ (sinv @ (U.T @ b))
        else:
            sol = (Vh @ b).T @ sinv @ (U.T @ c)

        if return_rank:
            return sol, rank
        else:
            return sol
    
    def logdet(self, K):
        return self.quad_or_solve(K)

    def solve(self, K, b):
        """ K^+ b """
        return self.quad_or_solve(K, b)
    
    def quad(self, K, b, c=None):
        """ b^T K^+ b or b^T K^+ c """
        return self.quad_or_solve(K, b, b if c is None else c)
    
    @classmethod
    def clskey(cls, mapping, default):
        for k, v in mapping.items():
            if re.search(k, cls.__name__, flags=re.IGNORECASE):
                return v
        return default
    
    def check_solve(self, bgen, jit=False):
        fun = lambda K, b: self.decompclass(K).solve(b)
        for n in self.sizes:
            K = self.randsymmat(n)
            b = bgen(len(K))
            result = fun(K, b)
            if jit:
                result2 = jax.jit(fun)(K, b)
                util.assert_close_matrices(result2, result, rtol=self.clskey({
                    r'pinv_chol': 1e-8,
                    r'pinv2_chol': 1e-9,
                }, 1e-11))
            else:
                sol = self.solve(K, b)
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'pinv_chol': 1e-3,
                    r'pinv2_chol': 1e-7,
                    r'TestSandwichSVD_EigCutFullRank': 1e-10,
                }, 1e-11), atol=1e-15)
    
    def check_solve_jac(self, bgen, jacfun, jit=False, hess=False, da=False, rtol=1e-7):
        # TODO sometimes the jacobian of fun is practically zero. Why? This
        # gives problems because it needs an higher absolute tolerance. =>
        # because in 1x1 matrices there is only the diagonal, which does not
        # depend on s (but this can't be that problem, because it's exactly 0)
        def fun(s, n, b):
            K = self.mat(s, n)
            return self.decompclass(K, direct_autodiff=da).solve(b)
        funjac = jacfun(fun)
        for n in self.sizes:
            s = np.exp(rng.uniform(-1, 1))
            b = bgen(n)
            result = funjac(s, n, b)
            if jit:
                funjacjit = jax.jit(funjac, static_argnums=1)
                result2 = funjacjit(s, n, b)
                util.assert_close_matrices(result2, result, rtol=self.clskey({
                    r'pinv2_chol': 1e-4,
                    r'pinv_chol': 1e-4,
                }, rtol))
                continue
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            KdK = self.solve(K, dK)
            Kb = self.solve(K, b)
            if not hess:
                # TODO use correct formulas for pseudoinverses
                # -K^-1 dK K^-1 b
                sol = -KdK @ Kb
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'woodbury[^2]': 1e-3,
                    r'woodbury2': 1e-3,
                    r'pinv(2|)_chol': 1e-4,
                    r'chol': 1e-6,
                    r'sandwichsvd': 1e-7,
                }, 1e-8))
            else:
                # TODO use correct formulas for pseudoinverses
                #  K^-1 dK K^-1 dK K^-1 b   +
                # -K^-1 d2K K^-1 b          +
                #  K^-1 dK K^-1 dK K^-1 b
                d2K = self.mathess(s, n)
                sol = 2 * KdK @ KdK @ Kb - self.solve(K, d2K) @ Kb
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'pinv_chol': 1e-4,
                    r'woodbury2': 1e-4,
                    r'woodbury[^2]': 1e-5,
                    r'chol': 1e-6,
                }, 1e-7))
    
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
        self.check_solve_jac(self.randvec, jax.jacfwd, True, rtol=self.clskey({
            r'pinv2_chol': 1e-3,
        }, 1e-6))

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
                # TODO use correct formulas for pseudoinverses
            sol = -A @ self.solve(K.T, self.solve(K, dK).T).T @ b
            result = funjac(s, n, b, A)
            util.assert_close_matrices(sol, result, rtol=self.clskey({
                r'pinv_chol': 1e-2,
                r'pinv2_chol': 1e-3,
            }, 1e-4))
        
    def test_solve_matrix_jac_rev_matrix(self):
        self.solve_matrix_jac_matrix(jax.jacrev)

    def test_solve_matrix_jac_fwd_matrix(self):
        self.solve_matrix_jac_matrix(jax.jacfwd)

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
            util.assert_similar_gvars(sol, result, rtol=self.clskey({
                r'pinv_chol': 1e-3,
                r'pinv2_chol': 1e-7,
            }, 1e-10), atol=1e-15)

    def test_quad_matrix_vec_gvar(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            b = self.randvec(n * 3).reshape(n, 3)
            c = self.randvecgvar(n)
            invK = self.solve(K, np.eye(len(K)))
            sol = b.T @ invK @ c
            result = self.decompclass(K).quad(b, c)
            util.assert_similar_gvars(sol, result, rtol=self.clskey({
                r'pinv2_chol': 1e-6,
                r'pinv_chol': 1e-8,
                r'sandwichqr': 1e-10,
            }, 1e-11), atol=1e-15)
    
    def check_quad(self, bgen, cgen=lambda n: None, jit=False):
        fun = lambda K, b, c: self.decompclass(K).quad(b, c)
        for n in self.sizes:
            K = self.randsymmat(n)
            b = bgen(len(K))
            c = cgen(len(K))
            result = fun(K, b, c)
            if jit:
                funjit = jax.jit(fun)
                result2 = funjit(K, b, c)
                util.assert_close_matrices(result2, result, rtol=self.clskey({
                    r'pinv2_chol': 1e-7,
                    r'pinv_chol': 1e-10,
                }, 1e-11))
            else:
                sol = self.quad(K, b, c)
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'pinv_chol': 1e-6,
                }, 1e-8))
    
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
                util.assert_close_matrices(result2, result, rtol=self.clskey({
                    r'pinv_chol': 1e-2,
                    r'pinv2_chol': 1e-3,
                    r'woodbury[^2]': 1e-4,
                }, 1e-7))
                continue
            if c is None:
                c = b
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            KdK = self.solve(K, dK)
            Kc = self.solve(K, c)
            if not hess:
                # TODO use correct formulas for pseudoinverses
                # b.T K^-1 c
                # -b.T K^-1 dK K^-1 c
                sol = -b.T @ KdK @ Kc
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'pinv_chol': 1e-2,
                    r'woodbury[^2]': 1e-2,
                    r'pinv2_chol': 1e-3,
                    r'chol': 1e-6,
                    r'woodbury2': 1e-7,
                }, 1e-8))
            else:
                # TODO use correct formulas for pseudoinverses
                #  b.T K^-1 dK K^-1 dK K^-1 c   +
                # -b.T K^-1 d2K K^-1 c          +
                #  b.T K^-1 dK K^-1 dK K^-1 c
                sol = 2 * b.T @ KdK @ KdK @ Kc
                if not stopg:
                    d2K = self.mathess(s, n)
                    sol -= b.T @ self.solve(K, d2K) @ Kc
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'pinv_chol': 1e-2,
                    r'pinv2_chol': 1e-4,
                    r'woodbury[^2]': 1e-3,
                }, 1e-6))
    
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
    
    def check_logdet(self, jit=False):
        fun = lambda K: self.decompclass(K).logdet()
        funjit = jax.jit(fun)
        for n in self.sizes:
            K = self.randsymmat(n)
            result = fun(K)
            if jit:
                result2 = funjit(K)
                util.assert_close_matrices(result2, result, rtol=self.clskey({
                    r'pinv_chol': 1e-2,
                    r'pinv2_chol': 1e-7,
                }, 1e-14))
            else:
                sol = self.logdet(K)
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'pinv2_chol': 1e-2,
                    r'pinv_chol': 1e-7,
                }, 1e-11))
    
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
                util.assert_close_matrices(result2, result, rtol=self.clskey({
                    r'pinv_chol': 1e-3,
                    r'woodbury2': 1e-6,
                    r'pinv2_chol': 1e-7,
                }, 1e-9), atol=1e-30)
                continue
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            KdK = self.solve(K, dK)
            if not hess:
                # tr(K^-1 dK)
                sol = np.trace(KdK)
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'pinv2_chol': 1e-3,
                    r'pinv_chol': 1e-4,
                    r'TestWoodbury2_EigCutFullRank': 1e-4,
                }, 1e-7), atol=1e-20)
            else:
                # tr(-K^-1 dK K^-1 dK + K d2K)
                sol = -np.trace(KdK @ KdK)
                if not stopg:
                    d2K = self.mathess(s, n)
                    Kd2K = self.solve(K, d2K)
                    sol += np.trace(Kd2K)
                util.assert_close_matrices(result, sol, rtol=self.clskey({
                    r'pinv_chol': 1e-2,
                    r'pinv2_chol': 1e-3,
                    r'woodbury[^2]': 1e-2,
                }, 1e-5), atol=1e-60)
    
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
            util.assert_close_matrices(result, sol, rtol=self.clskey({
                r'pinv2_chol': 1e-4,
                r'pinv_chol': 1e-5,
            }, 1e-11))
        
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
        self._woodbury_possign = True
        for n in self.sizes:
            K = self.randsymmat(n)
            dec = self.decompclass(K)
            A = dec.correlate(np.eye(dec.m))
            Q = A @ A.T
            util.assert_close_matrices(K, Q, rtol=self.clskey({
                r'pinv2_chol': 1e-4,
                r'pinv_chol': 1e-10,
                r'lobpcg': 1e-12,
            }, 1e-13))
    
    def test_double_correlate(self):
        self._woodbury_possign = True
        for n in self.sizes:
            K = self.randsymmat(n)
            d = self.decompclass(K)
            K2 = d.correlate(d.correlate(np.eye(n), transpose=True))
            util.assert_close_matrices(K, K2, atol=1e-15, rtol=self.clskey({
                r'pinv2_chol': 1e-3,
                r'pinv_chol': 1e-10,
            }, 1e-13))
    
    def test_correlate_transpose(self):
        self._woodbury_possign = True
        for n in self.sizes:
            K = self.randsymmat(n)
            AT = self.decompclass(K).correlate(np.eye(n), transpose=True)
            K2 = AT.T @ AT
            util.assert_close_matrices(K, K2, atol=1e-15, rtol=self.clskey({
                r'pinv_chol': 1e-9,
            }, 1e-13))

    def test_decorrelate_mat(self):
        self._woodbury_possign = True
        for n in self.sizes:
            K = self.randsymmat(n)
            b = self.randmat(n)
            x = self.decompclass(K).decorrelate(b)
            result = x.T @ x
            sol = self.decompclass(K).quad(b)
            util.assert_close_matrices(result, sol, rtol=self.clskey({
                r'pinv2_chol': 1e-6,
            }, 1e-13))
    
    def test_double_decorrelate(self):
        self._woodbury_possign = True
        for n in self.sizes:
            K = self.randsymmat(n)
            d = self.decompclass(K)
            Kinv = d.inv()
            Kinv2 = d.decorrelate(d.decorrelate(np.eye(n)), transpose=True)
            util.assert_close_matrices(Kinv, Kinv2, rtol=self.clskey({
                r'pinv2_chol': 1e-4,
                r'pinv_chol_lowrank': 1e-13,
            }, 1e-14))

class CompositeDecompTestBase(DecompTestBase):
    """ Abstract class to test decompositions which do not take as input
    the matrix itself """
    
    @property
    @abc.abstractmethod
    def compositeclass(self): pass
    
    @property
    @abc.abstractmethod
    def subdecompclass(self): pass
    
    @staticmethod
    def splitrank(n1, n2, rank):
        r = rng.integers(max(1, rank - n2), min(rank, n1 + 1)) if rank > 1 else 0
        assert 0 <= r <= n1 and 0 <= rank - r <= n2
        return r, rank - r

class ToeplitzBase(DecompTestBase):
    
    @staticmethod
    def singular_toeplitz(n, rank, ampl):
        """ produce an n x n real symmetric toeplitz matrix with given rank,
        ampl is an array of input values used deterministically in the
        construction """
        assert 0 <= rank <= n
        assert ampl.shape == (rank // 2 + rank % 2,)
        freqs = jnp.linspace(0, 1, 2 + n)[1 - rank % 2:1 + rank // 2]
        i = jnp.arange(n)[:, None]
        j = jnp.arange(n)[None, :]
        comps = jnp.cos(2 * jnp.pi * (i - j) * freqs[:, None, None]) * jnp.abs(ampl[:, None, None])
        return jnp.sum(comps, axis=0)

    def randsymmat(self, n):
        rank = self.rank(n)
        ampl = rng.gamma(2, 1/2, rank // 2 + rank % 2)
        return self.singular_toeplitz(n, rank, ampl)
    
    def mat(self, s, n):
        rank = self.rank(n)
        ampl = 1 + jnp.cos(s + jnp.arange(rank // 2 + rank % 2))
        return self.singular_toeplitz(n, rank, ampl)

class BlockDecompTestBase(CompositeDecompTestBase):
    """
    Abstract class for testing BlockDecomp.
    """
    
    def decompclass(self, K, **kw):
        if len(K) == 1:
            return self.subdecompclass(K, **kw)
        p = rng.integers(1, len(K))
        P = K[:p, :p]
        Q = K[:p, p:]
        S = K[p:, p:]
        Pdec = self.subdecompclass(P, **kw)
        subdec = lambda K, **kw: self.subdecompclass(K, **kw)
        return _linalg.Block(Pdec, S, Q, subdec, **kw)
    
class BlockDiagDecompTestBase(CompositeDecompTestBase):
    """
    Abstract class for testing BlockDiagDecomp.
    """
    
    _p = functools.cached_property(lambda self: {})
    def p(self, n):
        return self._p.setdefault(n, rng.integers(1, n) if n > 1 else 0)
    
    _subranks = functools.cached_property(lambda self: {})
    def subranks(self, n, rank):
        p = self.p(n)
        return self._subranks.setdefault(n, self.splitrank(p, n - p, rank))
    
    def randsymmat(self, n):
        rank = self.rank(n)
        p = self.p(n)
        r1, r2 = self.subranks(n, rank)
        assert 0 <= p <= n and 0 <= r1 <= p and 0 <= r2 <= n - p and r1 + r2 == rank
        K = np.zeros((n, n))
        if p > 0:
            K[:p, :p] = super().randsymmat(p, rank=r1)
        K[p:, p:] = super().randsymmat(n - p, rank=r2)
        return K
    
    def mat(self, s, n):
        rank = self.rank(n)
        p = self.p(n)
        r1, r2 = self.subranks(n, rank)
        A = super().mat(s, p, rank=r1)
        B = super().mat(s, n - p, rank=r2)
        return jlinalg.block_diag(A, B)
                
    def decompclass(self, K, **kw):
        if len(K) == 1:
            return self.subdecompclass(K, **kw)
        p = self.p(len(K))
        A = K[:p, :p]
        B = K[p:, p:]
        args = (self.subdecompclass(A, **kw), self.subdecompclass(B, **kw))
        return _linalg.BlockDiag(*args, **kw)

class SandwichTestBase(CompositeDecompTestBase):
    """
    Abstract base class to test Sandwich* decompositions
    """
    
    _inner = functools.cached_property(lambda self: {})
    def inner(self, n):
        return self._inner.setdefault(n, 1 + rng.integers(n))
            
    _rank = functools.cached_property(lambda self: {})
    def rank(self, n):
        return self._rank.setdefault(n, 1 + rng.integers(self.inner(n)) if self.lowrank else self.inner(n))
        # TODO guaranteed low rank
        
    _B = functools.cached_property(lambda self: {})
    def B(self, n):
        return self._B.setdefault(n, rng.standard_normal((n, self.rank(n))))
    
    _randA = functools.cached_property(lambda self: {})
    def randA(self, n):
        return self._randA.setdefault(n, super().randsymmat(self.inner(n), rank=self.rank(n)))
    
    def matA(self, s, n):
        return super().mat(s, self.inner(n), rank=self.rank(n))
    
    def randsymmat(self, n):
        A = self.randA(n)
        B = self.B(n)
        K = B @ A @ B.T
        util.assert_close_matrices(K, K.T, rtol=1e-15)
        self._afrom = 'randsymmat'
        return K
    
    def mat(self, s, n):
        A = self.matA(s, n)
        B = self.B(n)
        K = B @ A @ B.T
        self._afrom = 'mat'
        self._sparam = s
        return K
        
    def decompclass(self, K, **kw):
        if self._afrom == 'randsymmat':
            A = self.randA(len(K))
        elif self._afrom == 'mat':
            A = self.matA(self._sparam, len(K))
        B = self.B(len(K))
        A_decomp = self.subdecompclass(A, **kw)
        return self.compositeclass(A_decomp, B, **kw)
            
    def quad_or_solve(self, K, *args, **kw):
        sol, rank = super().quad_or_solve(K, *args, return_rank=True, **kw)
        assert rank == self.rank(len(K))
        return sol
    
class WoodburyTestBase(CompositeDecompTestBase):
    """
    Abstract base class to test Woodbury*
    """
    
    # TODO need a way to specify I want the low rank only on the inner matrix
    # to test woodbury2
    
    _inner = functools.cached_property(lambda self: {})
    def inner(self, n):
        return self._inner.setdefault(n, rng.integers(1, n + 1))
    
    _sign = functools.cached_property(lambda self: {})
    def sign(self, n):
        if getattr(self, '_woodbury_possign', False):
            return 1
        return self._sign.setdefault(n, -1 + 2 * rng.integers(2))
    
    _randM = functools.cached_property(lambda self: {})
    def randM(self, n):
        return self._randM.setdefault(n, super().randsymmat(n + self.inner(n)))
    
    def matM(self, s, n):
        return super().mat(s, n + self.inner(n))
        
    _subranks = functools.cached_property(lambda self: {})
    def subranks(self, n):
        return self._subranks.setdefault(n, self.splitrank(n, self.inner(n), self.rank(n + self.inner(n))))
    
    _proj = functools.cached_property(lambda self: {})
    def proj(self, n):
        u1 = randortho(n)
        u2 = randortho(self.inner(n))
        r1, r2 = self.subranks(n)
        w1 = np.zeros(len(u1))
        w2 = np.zeros(len(u2))
        w1[:r1] = 1
        w2[:r2] = 1
        p1 = (u1 * w1) @ u1.T
        p2 = (u2 * w2) @ u2.T
        return self._proj.setdefault(n, (p1, p2))
    
    def ABC(self, n, M):
        A = M[:n, :n]
        B = M[:n, n:]
        C = M[n:, n:]
        pa, pc = self.proj(n)
        A = pa @ A @ pa
        B = pa @ B @ pc
        C = pc @ C @ pc
        C = jnp.linalg.pinv(C) # such that A - BCB^T is p.s.d.
        return A, B, C
        
    def randsymmat(self, n):
        A, B, C = self.ABC(n, self.randM(n))
        K = A + self.sign(n) * B @ C @ B.T
        util.assert_close_matrices(K, K.T, rtol=1e-14)
        self._mfrom = 'randsymmat'
        return K
    
    def mat(self, s, n):
        A, B, C = self.ABC(n, self.matM(s, n))
        K = A + self.sign(n) * B @ C @ B.T
        self._mfrom = 'mat'
        self._sparam = s
        return K
    
    def decompclass(self, K, **kw):
        n = len(K)
        if self._mfrom == 'randsymmat':
            M = self.randM(n)
        elif self._mfrom == 'mat':
            M = self.matM(self._sparam, n)
        A, B, C = self.ABC(n, M)
        A_decomp = self.subdecompclass(A, **kw)
        C_decomp = self.subdecompclass(C, **kw)
        return self.compositeclass(A_decomp, B, C_decomp, self.subdecompclass, sign=self.sign(n), **kw)

class PinvTestBase(CompositeDecompTestBase):
    
    def decompclass(self, K, **kw):
        return self.compositeclass(K, self.subdecompclass, **kw)

class Pinv2TestBase(CompositeDecompTestBase):
    
    def decompclass(self, K, **kw):
        decompcls = lambda *args, **kw: self.subdecompclass(*args, epsrel=0, **kw)
        return self.compositeclass(K, decompcls, epsrel=1e-10, N=2, **kw)

class TestWoodbury2_EigCutFullRank(WoodburyTestBase): pass
class TestWoodbury_EigCutFullRank(WoodburyTestBase): pass

class TestSandwichSVD_EigCutFullRank(SandwichTestBase): pass
class TestSandwichQR_EigCutFullRank(SandwichTestBase): pass

class TestBlockDiag_EigCutFullRank(BlockDiagDecompTestBase): pass

class TestBlock_EigCutFullRank(BlockDecompTestBase): pass

class TestCholToeplitz(ToeplitzBase): pass

class TestChol(DecompTestBase): pass
class TestSVDCutLowRank(DecompTestBase): pass
class TestSVDCutFullRank(DecompTestBase): pass
class TestEigCutLowRank(DecompTestBase): pass
class TestEigCutFullRank(DecompTestBase): pass
        
class TestLanczos(DecompTestBase):
    def decompclass(self, K, **kw):
        return _linalg.Lanczos(K, rank=self.rank(len(K)), **kw)
    
class TestLOBPCG(DecompTestBase):
    def decompclass(self, K, **kw):
        return _linalg.LOBPCG(K, rank=self.rank(len(K)), **kw)

@mark.skip # Pinv and Pinv2 are crap, don't bother testing them for now
class TestPinv_Chol(PinvTestBase): pass
@mark.skip
class TestPinv2_Chol(Pinv2TestBase): pass

class TestSVDCutLowRank_LowRank(DecompTestBase): pass
class TestEigCutLowRank_LowRank(DecompTestBase): pass
class TestLanczos_LowRank(TestLanczos): pass
class TestLOBPCG_LowRank(TestLOBPCG): pass
@mark.skip
class TestPinv_Chol_LowRank(PinvTestBase): pass
@mark.skip
class TestPinv2_Chol_LowRank(Pinv2TestBase): pass
# class TestEigCutFullRank_LowRank(DecompTestBase): pass # fails as expected

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
                lhs = (A @ x).reshape(-1, B.shape[-1])
                util.assert_close_matrices(lhs, B.reshape(lhs.shape), rtol=1e-12)

@util.tryagain
def test_toeplitz_gershgorin():
    t = rng.standard_normal(100)
    m = linalg.toeplitz(t)
    b1 = _linalg._decomp._gershgorin_eigval_bound(m)
    b2 = _linalg._toeplitz.eigv_bound(t)
    util.assert_close_matrices(b2, b1, rtol=1e-15)

def check_toeplitz():
    mod = _linalg._toeplitz
    for n in [10, 2, 1]:
        x = np.linspace(0, 3, n)
        t = np.pi * np.exp(-1/2 * x ** 2)
        m = linalg.toeplitz(t)
    
        l1 = mod.chol(t)
        l2 = linalg.cholesky(m, lower=True)
        util.assert_close_matrices(l1, l2, rtol=1e-10)
    
        b = rng.standard_normal((len(t), 30))
        lb1 = mod.chol_matmul(t, b)
        lb2 = l2 @ b
        util.assert_close_matrices(lb1, lb2, rtol=1e-10)

        ld1 = mod.logdet(t)
        _, ld2 = np.linalg.slogdet(m)
        util.assert_close_matrices(ld1, ld2, rtol=1e-9)
    
        ilb1 = mod.chol_solve(t, b)
        ilb2 = linalg.solve_triangular(l2, b, lower=True)
        util.assert_close_matrices(ilb1, ilb2, rtol=1e-8)
    
        imb1 = mod.solve(t, b)
        imb2 = np.linalg.solve(m, b)
        util.assert_close_matrices(imb1, imb2, rtol=1e-8)

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
        for n in [0, 1, 2, 10]:
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
                lhs, rhs = np.broadcast_arrays(l @ ilb, b)
                lhs = lhs.reshape(-1 if lhs.size else 0, lhs.shape[-1])
                rhs = rhs.reshape(lhs.shape)
                util.assert_close_matrices(lhs, rhs, rtol=1e-7)
    with pytest.raises(np.linalg.LinAlgError):
        _linalg._toeplitz.chol_solve_numpy([-1], [1])
    with pytest.raises(np.linalg.LinAlgError):
        _linalg._toeplitz.chol_solve_numpy([1, 2], [1, 1])
    with pytest.raises(np.linalg.LinAlgError):
        _linalg._toeplitz.chol_solve_numpy([1, 0.5, 2], [1, 1, 1])

@mark.parametrize('mode', ['reduced', 'full'])
def test_rq(mode):
    for n in DecompTestBase.sizes:
        a = rng.standard_normal((n, 5))
        r, q = _linalg._decomp._rq(a, mode=mode)
        util.assert_close_matrices(r @ q, a, rtol=1e-15)
        util.assert_equal(np.tril(r), r)
        util.assert_close_matrices(  q @ q.T @   q,   q, rtol=1e-14)
        util.assert_close_matrices(q.T @   q @ q.T, q.T, rtol=1e-14)

@mark.parametrize('decomp', [
    _linalg.Chol,
    _linalg.EigCutLowRank,
    _linalg.SVDCutLowRank,
])
def test_degenerate(decomp):
    a = linalg.block_diag(np.eye(5), 0)
    d = decomp(a)
    util.assert_close_matrices(d.quad(a), a, rtol=1e-14)

def test_acausal_alg():
    with pytest.raises(ValueError):
        _linalg._seqalg.sequential_algorithm(2, [_linalg._seqalg.Stack(0)])

#### XFAILS ####
# keep last to avoid hiding them in wrappings

# TODO second derivatives completing but returning incorrect result
util.xfail(DecompTestBase, 'test_solve_matrix_hess_fwd_rev') # works only for TestSandwichSVDDiag
util.xfail(DecompTestBase, 'test_logdet_hess_fwd_rev') # since on quad it works, maybe I can solve it by using quad in the custom jvp
util.xfail(WoodburyTestBase, 'test_logdet_hess_fwd_fwd_stopg')
util.xfail(WoodburyTestBase, 'test_quad_matrix_matrix_hess_fwd_fwd_stopg')

# TODO fail on ubuntu-latest since jax 0.4.6 (???????), it's all the _da methods
# for Chol (completing but incorrect result, no inf/nan)
util.xfail(TestChol, 'test_solve_matrix_jac_fwd_da')
util.xfail(TestChol, 'test_solve_matrix_hess_da')
util.xfail(TestChol, 'test_quad_matrix_matrix_jac_fwd_da')
util.xfail(TestChol, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestChol, 'test_logdet_hess_da')

# TODO linalg.sparse.eigsh/lobpcg are not implemented in jax
for testcls in [TestLanczos, TestLOBPCG]:
    for name, meth in inspect.getmembers(testcls, inspect.isfunction):
        if name.endswith('_da'):
            util.xfail(testcls, name)

# TODO these derivatives are a full nan, no idea, but it depends on the values,
# and on the sign, so they xpass at random. really no idea
util.xfail(TestWoodbury2_EigCutFullRank, 'test_solve_matrix_jac_fwd_da')
util.xfail(TestWoodbury2_EigCutFullRank, 'test_solve_matrix_hess_da')
util.xfail(TestWoodbury2_EigCutFullRank, 'test_quad_matrix_matrix_jac_fwd_da')
util.xfail(TestWoodbury2_EigCutFullRank, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestWoodbury2_EigCutFullRank, 'test_logdet_hess_da')
util.xfail(TestBlock_EigCutFullRank, 'test_quad_vec_jac_fwd_jit')

# TODO fail with full nan
util.xfail(TestSVDCutLowRank_LowRank, 'test_solve_matrix_hess_da')
util.xfail(TestSVDCutLowRank_LowRank, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestSVDCutLowRank_LowRank, 'test_logdet_hess_da')
util.xfail(TestEigCutLowRank_LowRank, 'test_solve_matrix_hess_da')
util.xfail(TestEigCutLowRank_LowRank, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestEigCutLowRank_LowRank, 'test_logdet_hess_da')

# TODO sometimes very inaccurate. Maybe all these problems would go away if I
# tried to understand well what I'm doing with the parametric matrix to avoid
# exploding derivatives. Possibly the proposed unification of randsymmat() and
# mat() would do it.
util.xfail(TestWoodbury_EigCutFullRank, 'test_solve_matrix_jac_rev_matrix')
util.xfail(TestWoodbury_EigCutFullRank, 'test_solve_matrix_jac_rev_jit')
util.xfail(TestWoodbury_EigCutFullRank, 'test_solve_matrix_hess_fwd_fwd')
util.xfail(TestWoodbury_EigCutFullRank, 'test_quad_vec_jac_rev_jit')
util.xfail(TestWoodbury_EigCutFullRank, 'test_quad_vec_jac_rev')
util.xfail(TestWoodbury_EigCutFullRank, 'test_quad_vec_jac_fwd')
util.xfail(TestWoodbury_EigCutFullRank, 'test_logdet_jac_rev_jit')
util.xfail(TestWoodbury_EigCutFullRank, 'test_logdet_jac_rev')
util.xfail(TestWoodbury_EigCutFullRank, 'test_logdet_hess_fwd_fwd')

# TODO this fails because Pinv(2).logdet is not a pseudodeterminant.
util.xfail(TestPinv_Chol_LowRank, 'test_logdet')
util.xfail(TestPinv2_Chol_LowRank, 'test_logdet')

# TODO wildly inaccurate derivatives when the result is large, may xpass
util.xfail(TestPinv_Chol_LowRank, 'test_solve_vec_jac_rev')
util.xfail(TestPinv_Chol_LowRank, 'test_solve_vec_jac_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_solve_matrix_jac_rev')
util.xfail(TestPinv_Chol_LowRank, 'test_solve_matrix_jac_fwd_matrix')
util.xfail(TestPinv_Chol_LowRank, 'test_solve_matrix_jac_fwd_da')
util.xfail(TestPinv_Chol_LowRank, 'test_solve_matrix_jac_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_solve_matrix_hess_fwd_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_solve_matrix_hess_da')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_vec_jac_rev')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_vec_jac_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_matrix_matrix_jac_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_matrix_matrix_hess_fwd_rev')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_matrix_matrix_hess_fwd_fwd_stopg')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_matrix_matrix_hess_fwd_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_matrix_jac_rev')
util.xfail(TestPinv_Chol_LowRank, 'test_quad_matrix_jac_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_logdet_jac_rev')
util.xfail(TestPinv_Chol_LowRank, 'test_logdet_jac_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_logdet_hess_fwd_fwd')
util.xfail(TestPinv_Chol_LowRank, 'test_logdet_hess_da')
util.xfail(TestPinv_Chol, 'test_solve_vec_jac_rev')
util.xfail(TestPinv_Chol, 'test_solve_vec_jac_fwd')
util.xfail(TestPinv_Chol, 'test_solve_matrix_jac_rev_matrix')
util.xfail(TestPinv_Chol, 'test_solve_matrix_jac_rev')
util.xfail(TestPinv_Chol, 'test_solve_matrix_jac_fwd_matrix')
util.xfail(TestPinv_Chol, 'test_solve_matrix_jac_fwd_da')
util.xfail(TestPinv_Chol, 'test_solve_matrix_jac_fwd')
util.xfail(TestPinv_Chol, 'test_solve_matrix_hess_fwd_fwd')
util.xfail(TestPinv_Chol, 'test_solve_matrix_hess_da')
util.xfail(TestPinv_Chol, 'test_quad_vec_jac_rev')
util.xfail(TestPinv_Chol, 'test_quad_vec_jac_fwd')
util.xfail(TestPinv_Chol, 'test_quad_matrix_matrix_jac_fwd_da')
util.xfail(TestPinv_Chol, 'test_quad_matrix_matrix_jac_fwd')
util.xfail(TestPinv_Chol, 'test_quad_matrix_matrix_hess_fwd_rev')
util.xfail(TestPinv_Chol, 'test_quad_matrix_matrix_hess_fwd_fwd_stopg')
util.xfail(TestPinv_Chol, 'test_quad_matrix_matrix_hess_fwd_fwd')
util.xfail(TestPinv_Chol, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestPinv_Chol, 'test_quad_matrix_jac_rev')
util.xfail(TestPinv_Chol, 'test_quad_matrix_jac_fwd')
util.xfail(TestPinv_Chol, 'test_logdet_jac_rev')
util.xfail(TestPinv_Chol, 'test_logdet_jac_fwd')
util.xfail(TestPinv_Chol, 'test_logdet_hess_fwd_fwd_stopg')
util.xfail(TestPinv_Chol, 'test_logdet_hess_fwd_fwd')
util.xfail(TestPinv_Chol, 'test_logdet_hess_da')
util.xfail(TestPinv2_Chol_LowRank, 'test_solve_vec_jac_fwd')
util.xfail(TestPinv2_Chol_LowRank, 'test_solve_matrix_jac_rev_matrix')
util.xfail(TestPinv2_Chol_LowRank, 'test_solve_matrix_jac_rev')
util.xfail(TestPinv2_Chol_LowRank, 'test_solve_matrix_jac_fwd_matrix')
util.xfail(TestPinv2_Chol_LowRank, 'test_solve_matrix_jac_fwd_da')
util.xfail(TestPinv2_Chol_LowRank, 'test_solve_matrix_jac_fwd')
util.xfail(TestPinv2_Chol_LowRank, 'test_solve_matrix_hess_fwd_fwd')
util.xfail(TestPinv2_Chol_LowRank, 'test_quad_vec_jac_rev')
util.xfail(TestPinv2_Chol_LowRank, 'test_quad_matrix_matrix_jac_fwd')
util.xfail(TestPinv2_Chol_LowRank, 'test_quad_matrix_matrix_hess_fwd_rev')
util.xfail(TestPinv2_Chol_LowRank, 'test_quad_matrix_matrix_hess_fwd_fwd_stopg')
util.xfail(TestPinv2_Chol_LowRank, 'test_quad_matrix_matrix_hess_fwd_fwd')
util.xfail(TestPinv2_Chol_LowRank, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestPinv2_Chol_LowRank, 'test_quad_matrix_jac_rev')
util.xfail(TestPinv2_Chol_LowRank, 'test_logdet_jac_rev')
util.xfail(TestPinv2_Chol_LowRank, 'test_logdet_hess_fwd_fwd_stopg')
util.xfail(TestPinv2_Chol_LowRank, 'test_logdet_hess_fwd_fwd')
util.xfail(TestPinv2_Chol_LowRank, 'test_logdet_hess_da')
util.xfail(TestPinv2_Chol, 'test_solve_vec_jac_rev')
util.xfail(TestPinv2_Chol, 'test_solve_vec_jac_fwd')
util.xfail(TestPinv2_Chol, 'test_solve_matrix_jac_rev_matrix')
util.xfail(TestPinv2_Chol, 'test_solve_matrix_jac_rev')
util.xfail(TestPinv2_Chol, 'test_solve_matrix_jac_fwd')
util.xfail(TestPinv2_Chol, 'test_solve_matrix_hess_fwd_fwd')
util.xfail(TestPinv2_Chol, 'test_solve_matrix_hess_da')
util.xfail(TestPinv2_Chol, 'test_quad_vec_jac_fwd')
util.xfail(TestPinv2_Chol, 'test_quad_matrix_matrix_jac_fwd_da')
util.xfail(TestPinv2_Chol, 'test_quad_matrix_matrix_jac_fwd')
util.xfail(TestPinv2_Chol, 'test_quad_matrix_matrix_hess_fwd_rev')
util.xfail(TestPinv2_Chol, 'test_quad_matrix_matrix_hess_fwd_fwd_stopg')
util.xfail(TestPinv2_Chol, 'test_quad_matrix_matrix_hess_fwd_fwd')
util.xfail(TestPinv2_Chol, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestPinv2_Chol, 'test_quad_matrix_jac_rev')
util.xfail(TestPinv2_Chol, 'test_quad_matrix_jac_fwd')
util.xfail(TestPinv2_Chol, 'test_logdet_jac_rev')
util.xfail(TestPinv2_Chol, 'test_logdet_jac_fwd')
util.xfail(TestPinv2_Chol, 'test_logdet_hess_fwd_fwd_stopg')
util.xfail(TestPinv2_Chol, 'test_logdet_hess_fwd_fwd')
util.xfail(TestPinv2_Chol, 'test_logdet_hess_da')
util.xfail(TestPinv2_Chol_LowRank, 'test_logdet_jac_fwd')

# TODO Pinv2's correlate is a mess!
util.xfail(TestPinv2_Chol_LowRank, 'test_double_decorrelate')
util.xfail(TestPinv2_Chol_LowRank, 'test_double_correlate')
util.xfail(TestPinv2_Chol_LowRank, 'test_decorrelate_mat')
util.xfail(TestPinv2_Chol_LowRank, 'test_correlate_transpose')
util.xfail(TestPinv2_Chol_LowRank, 'test_correlate_eye')
util.xfail(TestPinv2_Chol, 'test_decorrelate_mat')
util.xfail(TestPinv2_Chol, 'test_correlate_transpose')

# TODO inaccurate jit (??)
util.xfail(TestPinv_Chol_LowRank, 'test_logdet_jit')
util.xfail(TestPinv_Chol, 'test_solve_vec_jac_fwd_jit')
util.xfail(TestPinv_Chol, 'test_solve_matrix_jac_fwd_jit')
util.xfail(TestPinv_Chol, 'test_quad_vec_jac_fwd_jit')

# TODO why?
# util.xfail(BlockDecompTestBase, 'test_logdet_hess_num')
