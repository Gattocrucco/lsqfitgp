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
DecompAutoDiff
    Abstract subclass that adds JAX support.
Diag
    Diagonalization.
EigCutFullRank
    Diagonalization rounding up small (or negative) eigenvalues.
EigCutLowRank
    Diagonalization removing small (or negative) eigenvalues.
SVDCutFullRank
    Diagonalization rounding up small eigenvalues, keeping their sign.
SVDCutLowRank
    Diagonalization removing small eigenvalues.
ReduceRank
    Partial diagonalization with higher eigenvalues only.
Chol
    Cholesky decomposition.
CholReg
    Abstract class for regularized Cholesky decomposition.
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
import warnings

import jax
from jax import numpy as jnp
import numpy
from scipy import linalg
from scipy import sparse
from jax.scipy import linalg as jlinalg
import gvar

from . import _toeplitz_linalg

# TODO optimize the matrix multiplication with gvars. Use these gvar internals:
# gvar.svec(int size)
# gvar.svec._assign(float[] values, int[] indices)
# gvar.GVar(float mean, svec derivs, smat cov)
# it may require cython to be fast since it's not vectorized

def choose_numpy(*args):
    if any(isinstance(x, jnp.ndarray) for x in args):
        return jnp
    else:
        return numpy

def notracer(x):
    """
    Unpack a JAX tracer.
    """
    if isinstance(x, jax.interpreters.ad.JVPTracer):
        return notracer(x.primal)
    elif isinstance(x, jax.interpreters.batching.BatchTracer):
        return notracer(x.val)
    else:
        return x

def asinexact(dtype):
    """
    Return dtype if it is inexact, else float64.
    """
    if jnp.issubdtype(dtype, jnp.inexact):
        return dtype
    else:
        return jnp.float64

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
    def __init__(self, K): # pragma: no cover
        """
        Decompose matrix K.
        """
        pass
        
    @abc.abstractmethod
    def solve(self, b): # pragma: no cover
        """
        Solve the linear system K @ x = b.
        """
        pass
    
    def quad(self, b, c=None):
        """
        Compute the quadratic form b.T @ inv(K) @ b if c is not specified, else
        b.T @ inv(K) @ c. `c` can be an array of gvars.
        """
        if c is None:
            c = b
        return b.T @ self.solve(c)
    
    @abc.abstractmethod
    def logdet(self): # pragma: no cover
        """
        Compute log(det(K)).
        """
        pass
    
    @abc.abstractmethod
    def correlate(self, b): # pragma: no cover
        """
        Compute A @ b where K = A @ A.T. If b represents iid variables with
        unitary variance, A @ b has covariance matrix K.
        """
        pass
    
    @abc.abstractmethod
    def decorrelate(self, b): # pragma: no cover
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
        
        return self.quad(jnp.eye(self.n))
    
    @property
    def n(self):
        """
        Return n where the decomposed matrix is n x n.
        """
        return len(self._K)

class DecompAutoDiff(Decomposition):
    """
    Abstract subclass adding JAX support to subclasses of Decomposition.
    Moreover, it fixes the convention of tensor multiplication by passing
    the concrete class methods only 2d matrices and then reshaping back the
    result.
    """
        
    def __init_subclass__(cls, **kw):
        
        # For __init__ I can't use an _autodiff flag like below to avoid double
        # wrapping because the wrapper is called as super().__init__ in
        # subclasses, so I assign self._K *after* calling old__init__.
        
        old__init__ = cls.__init__
        
        @functools.wraps(old__init__)
        def __init__(self, K, **kw):
            old__init__(self, notracer(K), **kw)
            self._K = K
        
        cls.__init__ = __init__
        
        for name in 'solve', 'quad', 'logdet':
            meth = getattr(cls, name)
            if not hasattr(meth, '_autodiff'):
                # print(f'defining {cls.__name__}.{name}')
                newmeth = getattr(DecompAutoDiff, '_make_' + name)(meth)
                newmeth = functools.wraps(meth)(newmeth)
                if not hasattr(newmeth, '_autodiff'):
                    newmeth._autodiff = True
                setattr(cls, name, newmeth)
        
        super().__init_subclass__(**kw)

    @staticmethod
    def _make_solve(oldsolve):
        
        # def solve_vjp_K(ans, self, K, b):
        #     assert ans.shape == b.shape
        #     assert b.shape[0] == K.shape[0] == K.shape[1]
        #     def vjp(g):
        #         assert g.shape[-len(b.shape):] == b.shape
        #         g = jnp.moveaxis(g, -len(b.shape), 0)
        #         A = solve_autodiff(self, K, g)
        #         B = jnp.moveaxis(ans, 0, -1)
        #         AB = jnp.tensordot(A, B, len(b.shape) - 1)
        #         AB = jnp.moveaxis(AB, 0, -2)
        #         assert AB.shape == g.shape[:-len(b.shape)] + K.shape
        #         return -AB
        #     return vjp
        #
        # def solve_vjp_b(ans, self, K, b):
        #     assert ans.shape == b.shape
        #     assert b.shape[0] == K.shape[0] == K.shape[1]
        #     def vjp(g):
        #         assert g.shape[-len(b.shape):] == b.shape
        #         g = jnp.moveaxis(g, -len(b.shape), 0)
        #         gj = solve_autodiff(self, K, g)
        #         gj = jnp.moveaxis(gj, 0, -len(b.shape))
        #         assert gj.shape == g.shape
        #         return gj
        #     return vjp
        #
        # def solve_fwd(self, K, b):
        #     ans = solve_autodiff(self, K, b)
        #     return ans, (ans, self, K, b)
        #
        # def solve_bwd(res, g):
        #     return None, solve_vjp_K(*res)(g), solve_vjp_b(*res)(g)
        #
        # solve_autodiff.defvjp(solve_fwd, solve_bwd)
        
        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def solve_autodiff(self, K, b):
            b2d = b.reshape(b.shape[0], -1)
            return oldsolve(self, b2d).reshape(b.shape)
        
        def solve_jvp_K(g, ans, self, K, b):
            assert ans.shape == b.shape
            assert ans.shape[0] == K.shape[0] == K.shape[1]
            assert g.shape[:2] == K.shape
            invKg = solve_autodiff(self, K, g)
            ans2D = ans.reshape(ans.shape[0], -1)
            jvp = -jnp.tensordot(invKg, ans2D, axes=(1, 0))
            jvp = jnp.moveaxis(jvp, -1, 1)
            return jvp.reshape(ans.shape + g.shape[2:])
        
        def solve_jvp_b(g, ans, self, K, b):
            assert ans.shape == b.shape
            assert b.shape[0] == K.shape[0] == K.shape[1]
            assert g.shape[:len(b.shape)] == b.shape
            return solve_autodiff(self, K, g)
        
        @solve_autodiff.defjvp
        def solve_vjp(self, primals, tangents):
            K, b = primals
            K_dot, b_dot = tangents
            ans = solve_autodiff(self, K, b)
            return ans, (
                solve_jvp_K(K_dot, ans, self, K, b) +
                solve_jvp_b(b_dot, ans, self, K, b)
            )
        
        def solve(self, b):
            return solve_autodiff(self, self._K, b)
        
        # solve_autodiff is used by other methods
        solve._autodiff = solve_autodiff
        
        return solve
    
    @staticmethod
    def _make_quad(oldquad):
        
        # def quad_vjp_K(ans, self, K, b, c):
        #     assert b.shape[0] == K.shape[0] == K.shape[1]
        #     bshape = b.shape[1:]
        #     if c is None:
        #         cshape = bshape
        #     else:
        #         assert c.shape[0] == b.shape[0]
        #         cshape = c.shape[1:]
        #     assert ans.shape == tuple(reversed(bshape)) + cshape
        #
        #     def vjp(g):
        #         assert g.shape[len(g.shape) - len(ans.shape):] == ans.shape
        #
        #         invKb = self.solve._autodiff(self, K, b)
        #         if c is None:
        #             invKc = invKb
        #         else:
        #             invKc = self.solve._autodiff(self, K, c)
        #
        #         axes = 2 * (tuple(range(-len(cshape), 0)),)
        #         ginvKc = jnp.tensordot(g, invKc, axes)
        #
        #         axes = 2 * (tuple(range(-len(bshape) - 1, -1)),)
        #         ginvKcb = jnp.tensordot(ginvKc, invKb.T, axes)
        #
        #         assert ginvKcb.shape == g.shape[:len(g.shape) - len(ans.shape)] + K.shape
        #         return -ginvKcb
        #     return vjp
        #
        # def quad_vjp_b(ans, self, K, b, c):
        #     assert b.shape[0] == K.shape[0] == K.shape[1]
        #     bshape = b.shape[1:]
        #     if c is None:
        #         cshape = bshape
        #     else:
        #         assert c.shape[0] == b.shape[0]
        #         cshape = c.shape[1:]
        #     assert ans.shape == tuple(reversed(bshape)) + cshape
        #
        #     if c is None:
        #         def vjp(g):
        #             glen = len(g.shape) - len(ans.shape)
        #             assert g.shape[glen:] == ans.shape
        #
        #             invKb = self.solve._autodiff(self, K, b)
        #
        #             axes = 2 * (range(-len(bshape), 0),)
        #             gj1 = jnp.tensordot(g, invKb, axes)
        #
        #             axes = tuple(range(glen))
        #             axes += tuple(reversed(range(glen, len(gj1.shape))))
        #             gj1 = jnp.transpose(gj1, axes)
        #             assert gj1.shape == g.shape[:glen] + b.shape
        #
        #             axes = (
        #                 tuple(range(-len(bshape) - 1, -2 * len(bshape) - 1, -1)),
        #                 tuple(range(-len(bshape), 0))
        #             )
        #             gj2 = jnp.tensordot(g, invKb, axes)
        #
        #             gj2 = jnp.moveaxis(gj2, -1, -len(bshape) - 1)
        #             assert gj2.shape == gj1.shape
        #
        #             return gj1 + gj2
        #     else:
        #         def vjp(g):
        #             glen = len(g.shape) - len(ans.shape)
        #             assert g.shape[glen:] == ans.shape
        #
        #             invKc = self.solve._autodiff(self, K, c)
        #
        #             axes = 2 * (range(-len(cshape), 0),)
        #             gj = jnp.tensordot(g, invKc, axes)
        #
        #             axes = tuple(range(glen))
        #             axes += tuple(reversed(range(glen, len(gj.shape))))
        #             gj = jnp.transpose(gj, axes)
        #             assert gj.shape == g.shape[:glen] + b.shape
        #             return gj
        #
        #     return vjp
        #
        # def quad_vjp_c(ans, self, K, b, c):
        #     assert b.shape[0] == K.shape[0] == K.shape[1]
        #     assert c.shape[0] == b.shape[0]
        #     bshape = b.shape[1:]
        #     cshape = c.shape[1:]
        #     assert ans.shape == tuple(reversed(bshape)) + cshape
        #
        #     def vjp(g):
        #         assert g.shape[len(g.shape) - len(ans.shape):] == ans.shape
        #
        #         invKb = self.solve._autodiff(self, K, b)
        #
        #         axes = (
        #             tuple(range(-len(cshape) - 1, -len(cshape) - len(bshape) - 1, -1)),
        #             tuple(range(-len(bshape), 0))
        #         )
        #         gj = jnp.tensordot(g, invKb, axes)
        #
        #         gj = jnp.moveaxis(gj, -1, -len(cshape) - 1)
        #         assert gj.shape == g.shape[:len(g.shape) - len(ans.shape)] + c.shape
        #         return gj
        #
        #     return vjp
        #
        # def quad_fwd(ans, self, K, b, c):
        #     ans = quad_autodiff(self, K, b, c)
        #     return ans, (ans, self, K, b, c)
        #
        # def quad_bwd(res, g):
        #     return None, quad_vjp_K(*res)(g), quad_vjp_b(*res)(g), quad_vjp_c(*res)(g)
        #
        # quad_autodiff.defvjp(quad_fwd, quad_bwd)
        
        # TODO currently backward derivatives don't work because when
        # transposing quad's jvp into a vjp somehow it's like a nonlinear
        # function of the tangent is evaluated. Probably happens in
        # quad_jvp_K but I don't know how.
        
        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def quad_autodiff(self, K, b, c):
            b2d = b.reshape(b.shape[0], -1)
            c2d = c.reshape(c.shape[0], -1)
            outshape = b.shape[1:][::-1] + c.shape[1:]
            return oldquad(self, b2d, c2d).reshape(outshape)

        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def quad_autodiff_cnone(self, K, b):
            b2d = b.reshape(b.shape[0], -1)
            outshape = b.shape[1:][::-1] + b.shape[1:]
            return oldquad(self, b2d).reshape(outshape)

        def quad_jvp_K(g, ans, self, K, b, c):
            if c is None:
                c = b
            bshape = b.shape[1:]
            cshape = c.shape[1:]
            assert ans.shape == tuple(reversed(bshape)) + cshape
            assert g.shape[:2] == K.shape
            g3d = g.reshape(K.shape + (-1,))
            ginvKc = quad_autodiff(self, K, g3d, c)
            ginvKc = jnp.moveaxis(ginvKc, 0, -1)
            ginvKc = ginvKc.reshape(c.shape + g.shape[2:])
            jvp = -quad_autodiff(self, K, b, ginvKc)
            assert jvp.shape == ans.shape + g.shape[2:]
            return jvp
        
        def quad_jvp_b(g, ans, self, K, b, c):
            assert g.shape[:len(b.shape)] == b.shape
            g1d = g.reshape(b.shape + (-1,))
            if c is None:
                jvp = quad_autodiff(self, K, b, g1d)
                jvpT = jnp.moveaxis(jvp.T, 0, -1)
                jvp = jvp + jvpT
            else:
                jvp = quad_autodiff(self, K, g1d, c)
                jvp = jnp.moveaxis(jvp, 0, -1)
            return jvp.reshape(ans.shape + g.shape[len(b.shape):])
        
        def quad_jvp_c(g, ans, self, K, b, c):
            assert g.shape[:len(c.shape)] == c.shape
            jvp = quad_autodiff(self, K, b, g)
            assert jvp.shape == ans.shape + g.shape[len(c.shape):]
            return jvp
        
        @quad_autodiff.defjvp
        def quad_jvp(self, primals, tangents):
            K, b, c = primals
            K_dot, b_dot, c_dot = tangents
            ans = quad_autodiff(self, K, b, c)
            return ans, (
                quad_jvp_K(K_dot, ans, self, K, b, c)
                + quad_jvp_b(b_dot, ans, self, K, b, c)
                + quad_jvp_c(c_dot, ans, self, K, b, c)
            )
            
        @quad_autodiff_cnone.defjvp
        def quad_cnone_jvp(self, primals, tangents):
            K, b = primals
            K_dot, b_dot = tangents
            ans = quad_autodiff_cnone(self, K, b)
            return ans, (
                quad_jvp_K(K_dot, ans, self, K, b, None) +
                quad_jvp_b(b_dot, ans, self, K, b, None)
            )

        def quad(self, b, c=None):
            if c is None:
                return quad_autodiff_cnone(self, self._K, b)
            else:
                return quad_autodiff(self, self._K, b, c)
        
        quad._autodiff = quad_autodiff

        return quad
    
    @staticmethod
    def _make_logdet(oldlogdet):
        
        # def logdet_vjp(ans, self, K):
        #     assert ans.shape == ()
        #     assert K.shape[0] == K.shape[1]
        #     def vjp(g):
        #         invK = self.quad._autodiff(self, K, jnp.eye(len(K)), None)
        #         return g[..., None, None] * invK
        #     return vjp
        #
        # def logdet_fwd(self, K):
        #     ans = logdet_autodiff(self, K)
        #     return ans, (ans, self, K)
        #
        # def logdet_bwd(res, g):
        #     return None, logdet_vjp(*res)(g)
        #
        # logdet_autodiff.defvjp(logdet_fwd, logdet_bwd)

        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def logdet_autodiff(self, K):
            return oldlogdet(self)
                        
        def logdet_jvp_K(g, ans, self, K):
            assert ans.shape == ()
            assert K.shape[0] == K.shape[1]
            assert g.shape[:2] == K.shape
            return jnp.trace(self.solve._autodiff(self, K, g))
        
        @logdet_autodiff.defjvp
        def logdet_jvp(self, primals, tangents):
            K, = primals
            K_dot, = tangents
            ans = logdet_autodiff(self, K)
            return ans, logdet_jvp_K(K_dot, ans, self, K)
        
        def logdet(self):
            return logdet_autodiff(self, self._K)
        
        return logdet
    
class Diag(DecompAutoDiff):
    """
    Diagonalization.
    """
    
    def __init__(self, K):
        self._w, self._V = jlinalg.eigh(K, check_finite=False)
    
    def solve(self, b):
        return (self._V / self._w) @ (self._V.T @ b)
    
    def quad(self, b, c=None):
        VTb = self._V.T @ b
        VTbw = VTb.T / self._w
        if c is None:
            VTc = VTb
        elif c.dtype == object:
            VTc = numpy.array(self._V).T @ c
            VTbw = numpy.array(VTbw)
        else:
            VTc = self._V.T @ c
        return VTbw @ VTc
    
    def logdet(self):
        return jnp.sum(jnp.log(self._w))
    
    def correlate(self, b):
        return (self._V * jnp.sqrt(self._w)) @ b
    
    def decorrelate(self, b):
        return (self._V / jnp.sqrt(self._w)).T @ b
    
    def _eps(self, eps):
        w = self._w
        if eps is None:
            eps = len(w) * jnp.finfo(asinexact(w.dtype)).eps
        assert jnp.isscalar(eps) and 0 <= eps < 1
        return eps * jnp.max(w)

class EigCutFullRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are set to `eps`, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None):
        super().__init__(K)
        eps = self._eps(eps)
        self._w = jnp.where(self._w < eps, eps, self._w)
            
class EigCutLowRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are removed, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None):
        super().__init__(K)
        eps = self._eps(eps)
        subset = slice(jnp.sum(self._w < eps), None) # w is sorted ascending
        self._w = self._w[subset]
        self._V = self._V[:, subset]
        # TODO jax's jit won't like this. fix: set entries to zero.
        
class SVDCutFullRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` in absolute value are set to
    `eps` with their sign, where `eps` is relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None):
        super().__init__(K)
        eps = self._eps(eps)
        cond = jnp.abs(self._w) < eps
        self._w = jnp.where(cond, eps * jnp.sign(self._w), self._w)
        
class SVDCutLowRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` in absolute value are removed,
    where `eps` is relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None):
        super().__init__(K)
        eps = self._eps(eps)
        subset = jnp.abs(self._w) >= eps
        self._w = self._w[subset]
        self._V = self._V[:, subset]

class ReduceRank(Diag):
    """
    Keep only the first `rank` higher eigenmodes.
    """
    
    def __init__(self, K, rank=1):
        assert isinstance(rank, (int, jnp.integer)) and rank >= 1
        self._w, self._V = sparse.linalg.eigsh(numpy.asarray(K), k=rank, which='LM')
    
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
    x = numpy.copy(b)
    a = numpy.asarray(a).reshape(a.shape + (1,) * len(x.shape[1:]))
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
    elif isinstance(a, jnp.ndarray) or isinstance(b, jnp.ndarray):
        return jlinalg.solve_triangular(a, b, lower=lower, check_finite=False)
    else:
        return linalg.solve_triangular(a, b, lower=lower, check_finite=False)

class Chol(DecompAutoDiff):
    """
    Cholesky decomposition.
    """
    
    def __init__(self, K):
        self._L = jlinalg.cholesky(K, lower=True, check_finite=False)
        if not jnp.all(jnp.isfinite(self._L)):
            raise numpy.linalg.LinAlgError('cholesky decomposition not finite, probably matrix not pos def numerically')
    
    def solve(self, b):
        invLb = solve_triangular_auto(self._L, b, lower=True)
        return solve_triangular_auto(self._L.T, invLb, lower=False)
    
    def quad(self, b, c=None):
        invLb = solve_triangular_auto(self._L, b, lower=True)
        if c is None:
            invLc = invLb
        else:
            invLc = solve_triangular_auto(self._L, c, lower=True)
            if c.dtype == object:
                invLb = numpy.asarray(invLb)
        return invLb.T @ invLc
    
    def logdet(self):
        return 2 * jnp.sum(jnp.log(jnp.diag(self._L)))
    
    def correlate(self, b):
        return self._L @ b
    
    def decorrelate(self, b):
        return solve_triangular_auto(self._L, b, lower=True)

    def _eps(self, eps, mat, maxeigv):
        if eps is None:
            eps = len(mat) * jnp.finfo(asinexact(mat.dtype)).eps
        assert 0 <= eps < 1
        return eps * maxeigv

def _scale(a):
    """
    Compute a vector s of powers of 2 such that diag(a / outer(s, s)) ~ 1.
    """
    return jnp.exp2(jnp.rint(0.5 * jnp.log2(jnp.diag(a))))

class CholReg(Chol):
    """
    Cholesky decomposition correcting for roundoff. Abstract class.
    """
    
    def __init__(self, K, eps=None):
        s = _scale(K)
        J = K / jnp.outer(s, s)
        J = self._regularize(J, eps)
        super().__init__(J)
        self._L = self._L * s[:, None]
    
    @abc.abstractmethod
    def _regularize(self, mat, eps): # pragma: no cover
        """Modify mat to make it numerically positive definite."""
        pass
    
def _gershgorin_eigval_bound(mat):
    """
    Upper bound on the largest magnitude eigenvalue of the matrix.
    """
    return jnp.max(jnp.sum(jnp.abs(mat), axis=1))

class CholGersh(CholReg):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff
    by adding to the diagonal a small number relative to the maximum eigenvalue.
    `eps` multiplies this number. The maximum eigenvalue is estimated
    with Gershgorin's theorem.
    """
    
    def _regularize(self, mat, eps):
        maxeigv = _gershgorin_eigval_bound(mat)
        return mat + jnp.diag(jnp.broadcast_to(self._eps(eps, mat, maxeigv), len(mat)))

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
    
    # TODO is it more efficient to make this a subclass of DecompAutoDiff?
    
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
        S_decomp_class : subclass of Decomposition
            A subclass of Decomposition used to decompose S - Q.T P^-1 Q.
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
        return jnp.concatenate([x, y])
    
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
        # Block Cholesky decomposition:
        # K = [P    Q] = L L^T
        #     [Q^T  S]
        # L = [A   ]
        #     [B  C]
        # AA^T = P
        # AB^T = Q  ==>  B^T = A^-1Q
        # BB^T + CC^T = S  ==>  CC^T = S - Q^T P^-1 Q
        invP = self._invP
        tildeS = self._tildeS
        Q = self._Q
        f = b[:len(Q)]
        g = b[len(Q):]
        x = invP.correlate(f)
        y = jnp.tensordot(invP.decorrelate(Q).T, f, 1) + tildeS.correlate(g)
        return jnp.concatenate([x, y])
    
    def decorrelate(self, b):
        # L^-1 = [   A^-1         ]
        #        [-C^-1BA^-1  C^-1]
        invP = self._invP
        tildeS = self._tildeS
        Q = self._Q
        f = b[:len(Q)]
        g = b[len(Q):]
        x = invP.decorrelate(f)
        y = tildeS.decorrelate(g - invP.quad(Q, f))
        return jnp.concatenate([x, y])
        
    @property
    def n(self):
        return sum(self._Q.shape)

class BlockDiagDecomp(Decomposition):
    
    # TODO allow NxN instead of 2x2?
    
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
        return jnp.concatenate([A.solve(f), B.solve(g)])
    
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
        return jnp.concatenate([A.correlate(f), B.correlate(g)])
    
    def decorrelate(self, b):
        A = self._A
        B = self._B
        An = A.n
        f = b[:An]
        g = b[An:]
        return jnp.concatenate([A.decorrelate(f), B.decorrelate(g)])

    @property
    def n(self):
        return self._A.n + self._B.n
