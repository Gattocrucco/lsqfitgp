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

import sys
import abc
import inspect
import functools

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

# TODO since the decompositions are pseudoinverses, I should do all tests on
# singular matrices too instead of just for some of them in test_degenerate. To
# make a degenerate matrix, multiply the current matrix factories by a random
# singular projector. To do:
#   - fix and document pinv behavior in Decomposition
#   - decide which properties to check (just the first MP identity? All four?)

rng = np.random.default_rng(202208091144)

class DecompTestABC(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def decompclass(self):
        pass
            
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        
        cls.matjac = jax.jacfwd(cls.mat, 1)
        cls.mathess = jax.jacfwd(cls.matjac, 1)
            
        # wrap all methods to try again in case of failure,
        # to hedge against bad random matrices
        for name, meth in vars(cls).items():
            # do not get all members, or all xfails would be needed to be
            # marked on all subclasses
            if name.startswith('test_'):
                setattr(cls, name, util.tryagain(meth, method=True))
        
        # assign automatically decomposition classes, if not defined
        if cls.__name__.startswith('Test') and inspect.isabstract(cls):
            name = cls.__name__[4:]
            if '_' in name:
                comp, name = name.split('_')
                cls.compositeclass = getattr(_linalg, comp)
                cls.subdecompclass = getattr(_linalg, name)
            else:
                cls.decompclass = getattr(_linalg, name)
        
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
        
    def randsymmat(self, n):
        O = stats.ortho_group.rvs(n) if n > 1 else np.atleast_2d(1)
        eigvals = rng.uniform(1e-2, 1e2, size=n)
        K = (O * eigvals) @ O.T
        util.assert_close_matrices(K, K.T, rtol=1e-15)
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
                util.assert_close_matrices(result2, result, rtol=1e-13)
            else:
                sol = self.solve(K, b)
                util.assert_close_matrices(result, sol, rtol=1e-11, atol=1e-15)
    
    def check_solve_jac(self, bgen, jacfun, jit=False, hess=False, da=False, rtol=1e-7):
        # TODO sometimes the jacobian of fun is practically zero. Why? This
        # gives problems because it needs an higher absolute tolerance. =>
        # because in 1x1 matrices there is only the diagonal, which does not
        # depend on s (but this can't be that problem, because it's exactly 0)
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
                util.assert_close_matrices(result2, result, rtol=rtol)
                continue
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            KdK = self.solve(K, dK)
            Kb = self.solve(K, b)
            if not hess:
                # -K^-1 dK K^-1 b
                sol = -KdK @ Kb
                util.assert_close_matrices(result, sol, rtol=1e-3)
                # 1e-3 is for Woodbury, otherwise 1e-11
            else:
                #  K^-1 dK K^-1 dK K^-1 b   +
                # -K^-1 d2K K^-1 b          +
                #  K^-1 dK K^-1 dK K^-1 b
                d2K = self.mathess(s, n)
                sol = 2 * KdK @ KdK @ Kb - self.solve(K, d2K) @ Kb
                util.assert_close_matrices(result, sol, rtol=1e-7)
    
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
        self.check_solve_jac(self.randvec, jax.jacfwd, True, rtol=1e-6)

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
            util.assert_close_matrices(sol, result, rtol=1e-4)
        
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
            util.assert_similar_gvars(sol, result, rtol=1e-10, atol=1e-15)

    def test_quad_matrix_vec_gvar(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            b = self.randvec(n * 3).reshape(n, 3)
            c = self.randvecgvar(n)
            invK = self.solve(K, np.eye(len(K)))
            sol = b.T @ invK @ c
            result = self.decompclass(K).quad(b, c)
            util.assert_similar_gvars(sol, result, rtol=1e-11, atol=1e-15)
    
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
                util.assert_close_matrices(result2, result, rtol=1e-13)
            else:
                sol = self.quad(K, b, c)
                util.assert_close_matrices(result, sol, rtol=1e-11)
    
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
                util.assert_close_matrices(result2, result, rtol=1e-8)
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
                util.assert_close_matrices(result, sol, rtol=1e-4)
                # 1e-4 is for Woodbury, otherwise 1e-9
            else:
                #  b.T K^-1 dK K^-1 dK K^-1 c   +
                # -b.T K^-1 d2K K^-1 c          +
                #  b.T K^-1 dK K^-1 dK K^-1 c
                sol = 2 * b.T @ KdK @ KdK @ Kc
                if not stopg:
                    d2K = self.mathess(s, n)
                    sol -= b.T @ self.solve(K, d2K) @ Kc
                util.assert_close_matrices(result, sol, rtol=1e-6)
    
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
                util.assert_close_matrices(result2, result, rtol=1e-15)
            else:
                sol = self.logdet(K)
                util.assert_close_matrices(result, sol, rtol=1e-11)
    
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
                util.assert_close_matrices(result2, result, rtol=1e-9, atol=1e-30)
                continue
            K = self.mat(s, n)
            dK = self.matjac(s, n)
            KdK = self.solve(K, dK)
            if not hess:
                # tr(K^-1 dK)
                sol = np.trace(KdK)
                util.assert_close_matrices(result, sol, rtol=1e-7)
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
                util.assert_close_matrices(result, sol, rtol=1e-5, atol=1e-100)
    
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
            util.assert_close_matrices(result, sol, rtol=1e-11)
        
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
            util.assert_close_matrices(K, Q, rtol=1e-13)
    
    def test_double_correlate(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            d = self.decompclass(K)
            K2 = d.correlate(d.correlate(np.eye(n), transpose=True))
            util.assert_close_matrices(K, K2, atol=1e-15, rtol=1e-14)
    
    def test_correlate_transpose(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            AT = self.decompclass(K).correlate(np.eye(n), transpose=True)
            K2 = AT.T @ AT
            util.assert_close_matrices(K, K2, atol=1e-15, rtol=1e-14)

    def test_decorrelate_mat(self):
        for n in self.sizes:
            K = self.randsymmat(n)
            b = self.randmat(n)
            x = self.decompclass(K).decorrelate(b)
            result = x.T @ x
            sol = self.decompclass(K).quad(b)
            util.assert_close_matrices(result, sol, rtol=1e-14)
            
class CompositeDecompTestBase(DecompTestBase):
    
    @property
    @abc.abstractmethod
    def compositeclass(self): pass
    
    @property
    @abc.abstractmethod
    def subdecompclass(self): pass

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

class BlockDecompTestBase(CompositeDecompTestBase):
    """
    Abstract class for testing BlockDecomp. Concrete subclasses must
    overwrite `subdecompclass`.
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
    Abstract class for testing BlockDiagDecomp. Concrete subclasses must
    overwrite `subdecompclass`.
    """
    
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
                
    def decompclass(self, K, **kw):
        if len(K) == 1:
            return self.subdecompclass(K, **kw)
        p = self._p
        assert p < len(K)
        A = K[:p, :p]
        B = K[p:, p:]
        args = (self.subdecompclass(A, **kw), self.subdecompclass(B, **kw))
        return _linalg.BlockDiag(*args, **kw)

class SandwichTestBase(CompositeDecompTestBase):
    """
    Abstract base class to test Sandwich* decompositions
    """
    
    ranks = functools.cached_property(lambda self: {})
    As = functools.cached_property(lambda self: {})
    Bs = functools.cached_property(lambda self: {})
    
    def rank(self, n):
        return self.ranks.setdefault(n, rng.integers(1, n + 1))
    
    def B(self, n):
        return self.Bs.setdefault(n, rng.standard_normal((n, self.rank(n))))
    
    def randA(self, n):
        return self.As.setdefault(n, super().randsymmat(self.rank(n)))
    
    def matA(self, s, n):
        return super().mat(s, self.rank(n))
        
    def decompclass(self, K, **kw):
        if self._afrom == 'randsymmat':
            A = self.As[len(K)]
        elif self._afrom == 'mat':
            A = self.matA(self._sparam, len(K))
        B = self.Bs[len(K)]
        A_decomp = self.subdecompclass(A, **kw)
        return self.compositeclass(A_decomp, B, **kw)
            
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
        
    def solve(self, K, b):
        invK, rank = linalg.pinv(K, return_rank=True)
        assert rank == self.rank(len(K))
        return invK @ b
    
    def logdet(self, K):
        return np.sum(np.log(np.sort(linalg.eigvalsh(K))[-self.rank(len(K)):]))

class WoodburyTestBase(CompositeDecompTestBase):
    """
    Abstract base class to test Woodbury*
    """
    
    ranks = functools.cached_property(lambda self: {})
    Ms = functools.cached_property(lambda self: {})
    signs = functools.cached_property(lambda self: {})
    
    def rank(self, n):
        return self.ranks.setdefault(n, rng.integers(1, n + 1))
    
    def sign(self, n):
        return self.signs.setdefault(n, -1 + 2 * rng.integers(2))
    
    def randM(self, n):
        return self.Ms.setdefault(n, super().randsymmat(n + self.rank(n)))
    
    def matM(self, s, n):
        return super().mat(s, n + self.rank(n))
    
    def ABC(self, n, M):
        A = M[:n, :n]
        B = M[:n, n:]
        C = jnp.linalg.inv(M[n:, n:]) # such that A - BCB^T is p.s.d.
        return A, B, C
        
    def randsymmat(self, n):
        A, B, C = self.ABC(n, self.randM(n))
        K = A + self.sign(n) * B @ C @ B.T
        util.assert_close_matrices(K, K.T, rtol=1e-15)
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

class TestWoodbury2_EigCutFullRank(WoodburyTestBase): pass
class TestWoodbury_EigCutFullRank(WoodburyTestBase): pass

class TestSandwichSVD_EigCutFullRank(SandwichTestBase): pass
class TestSandwichQR_EigCutFullRank(SandwichTestBase): pass

class TestBlockDiag_EigCutFullRank(BlockDiagDecompTestBase): pass

class TestBlock_EigCutFullRank(BlockDecompTestBase): pass

class TestCholToeplitz(ToeplitzBase): pass

class TestCholGersh(DecompTestBase): pass
class TestSVDCutLowRank(DecompTestBase): pass
class TestSVDCutFullRank(DecompTestBase): pass
class TestEigCutLowRank(DecompTestBase): pass
class TestEigCutFullRank(DecompTestBase): pass
        
class TestReduceRank(DecompTestBase):
    
    ranks = functools.cached_property(lambda self: {})
    
    def rank(self, n):
        return self.ranks.setdefault(n, rng.integers(1, n + 1))
    
    def decompclass(self, K, **kw):
        return _linalg.ReduceRank(K, rank=self.rank(len(K)), **kw)
    
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
        util.assert_close_matrices(K, K.T, rtol=1e-15)
        return K
    
    def mat(self, s, n):
        rank = self.rank(n)
        x = np.arange(n)
        x[:n - rank + 1] = x[0]
        return np.pi * jnp.exp(-1/2 * (x[:, None] - x[None, :]) ** 2 / s ** 2)
        
    def logdet(self, K):
        return np.sum(np.log(np.sort(linalg.eigvalsh(K))[-self.rank(len(K)):]))
        
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
                util.assert_close_matrices(lhs, B.reshape(lhs.shape), rtol=1e-13)

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
        util.assert_close_matrices(lb1, lb2, rtol=1e-11)

        ld1 = mod.logdet(t)
        _, ld2 = np.linalg.slogdet(m)
        util.assert_close_matrices(ld1, ld2, rtol=1e-10)
    
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
                lhs, rhs = np.broadcast_arrays(l @ ilb, b)
                lhs = lhs.reshape(-1, lhs.shape[-1])
                rhs = rhs.reshape(lhs.shape)
                util.assert_close_matrices(lhs, rhs, rtol=1e-8)

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
    _linalg.CholGersh,
    _linalg.EigCutLowRank,
    _linalg.SVDCutLowRank,
])
def test_degenerate(decomp):
    a = linalg.block_diag(np.eye(5), 0)
    d = decomp(a)
    util.assert_close_matrices(d.quad(a), a, rtol=1e-14)

#### XFAILS ####
# keep last to avoid hiding them in wrappings

# TODO second derivatives completing but returning incorrect result
util.xfail(DecompTestBase, 'test_solve_matrix_hess_fwd_rev') # works only for TestSandwichSVDDiag
util.xfail(DecompTestBase, 'test_logdet_hess_fwd_rev') # since on quad it works, maybe I can solve it by using quad in the custom jvp
util.xfail(WoodburyTestBase, 'test_logdet_hess_fwd_fwd_stopg')
util.xfail(WoodburyTestBase, 'test_quad_matrix_matrix_hess_fwd_fwd_stopg')

# TODO linalg.sparse.eigsh is not implemented in jax
for name, meth in inspect.getmembers(TestReduceRank, inspect.isfunction):
    if name.endswith('_da'):
        util.xfail(TestReduceRank, name)

# reverse diff broken because they use quads within other stuff probably.
# Subclassing DecompAutoDiff does not work. Maybe just using quad to compute
# tildeS is too much. The error is two undefined primals in a matrix
# multiplication jvp transpose. => it's probably derivatives w.r.t. the outer
# sides of quad
# for cls in [BlockDecompTestBase, WoodburyTestBase]:
    # a jax update between 0.3.13 and 0.3.17 solved all these, dunno why
    # util.xfail(cls, 'test_solve_vec_jac_rev')
    # util.xfail(cls, 'test_solve_matrix_jac_rev')
    # util.xfail(cls, 'test_solve_vec_jac_rev_jit')
    # util.xfail(cls, 'test_solve_matrix_jac_rev_jit')
    # util.xfail(cls, 'test_solve_matrix_jac_rev_matrix')
    # util.xfail(cls, 'test_quad_vec_jac_rev')
    # util.xfail(cls, 'test_quad_matrix_jac_rev')
    # util.xfail(cls, 'test_quad_vec_jac_rev_jit')
    # util.xfail(cls, 'test_quad_matrix_jac_rev_jit')
    # util.xfail(cls, 'test_logdet_jac_rev')
    # util.xfail(cls, 'test_logdet_jac_rev_jit')

# TODO I don't know how to implement correlate and decorrelate for Woodbury
# => do it with a larger i.i.d. space, using the pseudoinverse of the factor
# (change the definition of correlate to use the pinv)
for name, meth in inspect.getmembers(WoodburyTestBase, inspect.isfunction):
    if 'correlate' in name:
        util.xfail(WoodburyTestBase, name)

# TODO these derivatives are a full nan, no idea, but it depends on the values,
# and on the sign, so they xpass at random. really no idea
util.xfail(TestWoodbury2_EigCutFullRank, 'test_solve_matrix_jac_fwd_da')
util.xfail(TestWoodbury2_EigCutFullRank, 'test_solve_matrix_hess_da')
util.xfail(TestWoodbury2_EigCutFullRank, 'test_quad_matrix_matrix_jac_fwd_da')
util.xfail(TestWoodbury2_EigCutFullRank, 'test_quad_matrix_matrix_hess_da')
util.xfail(TestWoodbury2_EigCutFullRank, 'test_logdet_hess_da')

# TODO why?
# util.xfail(BlockDecompTestBase, 'test_logdet_hess_num')

# TODO this sometimes is very inaccurate
# TestWoodburyEigCutFullRank.test_logdet_hess_fwd_fwd
