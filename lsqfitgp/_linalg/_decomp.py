# lsqfitgp/_linalg/_decomp.py
#
# Copyright (c) 2020, 2022, 2023 Giacomo Petrillo
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

Decompositions of positive definite matrices.

A decomposition object is initialized with a square symmetric matrix and then
can solve linear systems and do other stuff with that matrix. The matrix is
assumed to be positive semidefinite in principle but not numerically,
decompositions differ on how the eventual numerical degeneracy is handled.

It is intended that the matrix inverse is a Moore-Penrose pseudoinverse in case
of (numerically) singular matrices. A different pseudoinverse or lack of
support for singular matrices is considered a bug.

These classes never check for infs/nans in the matrices.

Classes
-------
Decomposition
    Abstract base class.
DecompPyTree
    Abstract subclass that register subclasses as jax pytrees.
DecompAutoDiff
    Abstract subclass that adds decomposition-independent jax-autodiff support.
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
    of the maximum eigenvalue. Uses Schur's algorithm.
CholToeplitzML
    Like CholToeplitz but does not store the cholesky factor in memory.
BlockDecomp
    Decompose a 2x2 block matrix.
BlockDiagDecomp
    Decompose a 2x2 block diagonal matrix.
SandwichQR, SandwichSVD
    Decompose B A B^T.
Woodbury
    Decompose A ± B C B^T.

"""

import abc
import functools
import warnings
import inspect

import jax
from jax import numpy as jnp
from jax.scipy import linalg as jlinalg
import numpy
from scipy import linalg
from scipy import sparse
import gvar

from . import _toeplitz
from .. import _patch_jax
from . import _pytree

# TODO optimize matrix multiplication with gvars. Use these gvar internals:
# gvar.svec(int size)
# gvar.svec._assign(float[] values, int[] indices)
# gvar.GVar(float mean, svec derivs, smat cov)
# it may require cython to be fast since it's not vectorized

def _transpose(x):
    """ swap the last two axes of array x, corresponds to matrix tranposition
    with the broadcasting convention of matmul """
    if x.ndim < 2:
        return x
    elif isinstance(x, jnp.ndarray):
        return jnp.swapaxes(x, -2, -1)
    else:
        # need to support numpy because this function is used with gvars
        return numpy.swapaxes(x, -2, -1)

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
    
    @abc.abstractmethod
    def quad(self, b, c=None): # pragma: no cover
        """
        Compute the quadratic form b.T @ inv(K) @ b if c is not specified, else
        b.T @ inv(K) @ c. `c` can be an array of gvars.
        """
        pass
        
    # TODO to compute efficiently the predictive variance I need a new method
    # diagquad(b). And for things like toeplitzml and other memoryless solvers
    # to be added in the future, quad_and_logdet for the marginal likelihood.
    
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
    @abc.abstractmethod
    def n(self): # pragma: no cover
        """
        Return n where the decomposed matrix is n x n.
        """
        pass

class DecompPyTree(Decomposition, _pytree.AutoPyTree):
    pass

class DecompAutoDiffBase(DecompPyTree):
    """
    Abstract subclass adding JAX autodiff support to subclasses of
    Decomposition, even if the decomposition algorithm is not supported by JAX.
    """
    
    @abc.abstractmethod
    def matrix(self, *args, **kw):
        """ Given the initialization arguments, returns the matrix to be
        decomposed """
        return K
            
    @abc.abstractmethod
    def map_traceable_init_args(self, fun, *args, **kw):
        """ Given a function Any -> Any, and the initialization arguments,
        apply the function only to those arguments which are considered inputs
        to the definition of the matrix """
        return mapped_args, mapped_kw
        
    def _dad_matrix_(self):
        return self.matrix(*self._dad_args_, **self._dad_kw_)
    
    def _popattr(self, kw, attr, default):
        if not hasattr(self, attr):
            setattr(self, attr, kw.pop(attr, default))
    
    # TODO try using jax.linearize and jax.convert_closure. Or: try to pass
    # the decomposition products as variadic arguments to the intermediate
    # wrapper, maybe this solves all reverse-mode autodiff problems like
    # the double undefined primals.
        
    def __init_subclass__(cls, **kw):
        
        super().__init_subclass__(**kw)
        
        if inspect.isabstract(cls):
            # do not wrap the class methods if it's still abstract, because
            # __init_subclass__ will necessarily be invoked again on the
            # concrete subclasses
            return

        old__init__ = cls.__init__
        
        @functools.wraps(old__init__)
        def __init__(self, *args, **kw):
            self._popattr(kw, 'direct_autodiff', False)
            self._popattr(kw, 'stop_hessian', False)
            if self.direct_autodiff and self.stop_hessian:
                nargs, nkw = self.map_traceable_init_args(_patch_jax.stop_hessian, *args, **kw)
                old__init__(self, *nargs, **nkw)
            elif self.direct_autodiff:
                old__init__(self, *args, **kw)
            else:
                nargs, nkw = self.map_traceable_init_args(jax.lax.stop_gradient, *args, **kw)
                old__init__(self, *nargs, **nkw)
                if self.stop_hessian:
                    args, kw = self.map_traceable_init_args(_patch_jax.stop_hessian, *args, **kw)
            self._dad_args_ = args
            self._dad_kw_ = kw
            # For __init__ I can't use an _autodiff flag like below to avoid
            # double wrapping because the wrapper is called as super().__init__
            # in subclasses, so I assign _dad_args_ and _dad_kw *after* calling
            # old__init__.
        
        cls.__init__ = __init__
        
        for name in 'solve', 'quad', 'logdet':
            meth = getattr(cls, name)
            if not hasattr(meth, '_autodiff'):
                newmeth = getattr(cls, '_make_' + name)(meth)
                newmeth = functools.wraps(meth)(newmeth)
                if not hasattr(newmeth, '_autodiff'):
                    newmeth._autodiff = True
                setattr(cls, name, newmeth)
    
    @staticmethod
    def _make_solve(oldsolve):
        
        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def solve_autodiff(self, K, b):
            return oldsolve(self, b)
        
        @solve_autodiff.defjvp
        def solve_vjp(self, primals, tangents):
            K, b = primals
            K_dot, b_dot = tangents
            primal = solve_autodiff(self, K, b)
            tangent_K = -solve_autodiff(self, K, K_dot) @ primal
            tangent_b = solve_autodiff(self, K, b_dot)
            return primal, tangent_K + tangent_b
        
        def solve(self, b):
            if self.direct_autodiff:
                return oldsolve(self, b)
            else:
                return solve_autodiff(self, self._dad_matrix_(), b)
        
        # solve_autodiff is used by logdet_jvp and quad_jvp.
        solve._autodiff = solve_autodiff
        
        return solve
    
    @staticmethod
    def _make_quad(oldquad):
        
        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def quad_autodiff(self, K, b, c):
            return oldquad(self, b, c)

        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def quad_autodiff_cnone(self, K, b):
            return oldquad(self, b)
        
        # TODO the tangent_K calculation with quad_autodiff breaks backward
        # derivatives, I don't know why. Maybe I should define a new
        # decomposition function sandwitch(b, X, c) which computes b.T K^-1 X
        # K^-1 c and use it both for numerical accuracy (I'm guessing that
        # avoiding the two concatenated solves within solve is good) and to make
        # the jvp work. For things like ToeplitzML that stream-decompose each
        # time, it would be more efficient to have versions of solve and quad
        # with an arbitrary number of right-hand sides.

        @quad_autodiff.defjvp
        def quad_jvp(self, primals, tangents):
            K, b, c = primals
            K_dot, b_dot, c_dot = tangents
            primal = quad_autodiff(self, K, b, c)
            # tangent_K = -quad_autodiff(self, K, b, quad_autodiff(self, K, K_dot, c))
            Kb = self.solve._autodiff(self, K, b)
            Kc = self.solve._autodiff(self, K, c)
            tangent_K = -_transpose(Kb) @ K_dot @ Kc
            tangent_b = quad_autodiff(self, K, b_dot, c)
            # tangent_b = _transpose(b_dot) @ self.solve._autodiff(self, K, c)
            tangent_c = quad_autodiff(self, K, b, c_dot)
            # tangent_c = _transpose(b) @ self.solve._autodiff(self, K, c_dot)
            return primal, tangent_K + tangent_b + tangent_c
            
        @quad_autodiff_cnone.defjvp
        def quad_cnone_jvp(self, primals, tangents):
            K, b = primals
            K_dot, b_dot = tangents
            primal = quad_autodiff_cnone(self, K, b)
            # tangent_K = -quad_autodiff(self, K, b, quad_autodiff(self, K, K_dot, b))
            Kb = self.solve._autodiff(self, K, b)
            tangent_K = -_transpose(Kb) @ K_dot @ Kb
            tangent_b = quad_autodiff(self, K, b_dot, b)
            return primal, tangent_K + tangent_b + _transpose(tangent_b)

        def quad(self, b, c=None):
            if self.direct_autodiff:
                return oldquad(self, b, c)
            elif c is None:
                return quad_autodiff_cnone(self, self._dad_matrix_(), b)
            elif c.dtype == object:
                return oldquad(self, b, c)
                # if c contains gvars, the output is an array of gvars, and
                # quad_autodiff would hang because custom_jvp functions are
                # checked not to return object dtypes, so I have to call
                # oldquad
            else:
                return quad_autodiff(self, self._dad_matrix_(), b, c)
        
        return quad
    
    @staticmethod
    def _make_logdet(oldlogdet):
        
        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def logdet_autodiff(self, K):
            return oldlogdet(self)
                        
        @logdet_autodiff.defjvp
        def logdet_jvp(self, primals, tangents):
            K, = primals
            K_dot, = tangents
            primal = logdet_autodiff(self, K)
            tangent = jnp.trace(self.solve._autodiff(self, K, K_dot))
            # TODO for efficiency I should define a new method
            # tracesolve.
            return primal, tangent
        
        def logdet(self):
            if self.direct_autodiff:
                return oldlogdet(self)
            else:
                return logdet_autodiff(self, self._dad_matrix_())
        
        return logdet

class DecompAutoDiff(DecompAutoDiffBase):
    
    @property
    def n(self):
        return len(self._dad_args_[0])
    
    def matrix(self, K, *args, **kw):
        return K
    
    def map_traceable_init_args(self, fun, K, *args, **kw):
        return (fun(K), *args), kw
        
class Diag(DecompAutoDiff):
    """
    Diagonalization.
    """
    
    def __init__(self, K):
        self._w, self._V = jlinalg.eigh(K)
    
    def solve(self, b):
        return (self._V / self._w) @ (self._V.T @ b)
    
    def quad(self, b, c=None):
        VTb = self._V.T @ b
        VTbw = _transpose(VTb) / self._w
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
            eps = len(w) * jnp.finfo(w.dtype).eps
        assert 0 <= eps < 1
        return eps * jnp.max(jnp.abs(w))

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
        cond = self._w < eps
        self._w = jnp.where(cond, 1, self._w)
        self._V = jnp.where(cond, 0, self._V)

class _SVD(Diag):
    """ Like Diag but supports negative values in w """
    
    def logdet(self):
        return jnp.sum(jnp.log(jnp.abs(self._w)))
    
    def correlate(self, b):
        return (self._V * jnp.sqrt(jnp.abs(self._w))) @ b
    
    def decorrelate(self, b):
        return (self._V / jnp.sqrt(jnp.abs(self._w))).T @ b
        
class SVDCutFullRank(_SVD):
    """
    Diagonalization. Eigenvalues below `eps` in absolute value are set to
    `eps` with their sign, where `eps` is relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None):
        super().__init__(K)
        eps = self._eps(eps)
        cond = jnp.abs(self._w) < eps
        self._w = jnp.where(cond, eps * jnp.sign(self._w), self._w)
        
class SVDCutLowRank(_SVD):
    """
    Diagonalization. Eigenvalues below `eps` in absolute value are removed,
    where `eps` is relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None):
        super().__init__(K)
        eps = self._eps(eps)
        cond = jnp.abs(self._w) < eps
        self._w = jnp.where(cond, 1, self._w)
        self._V = jnp.where(cond, 0, self._V)

class ReduceRank(Diag):
    """
    Keep only the first `rank` higher eigenmodes.
    """
    
    def __init__(self, K, rank=1):
        class wdummy:
            dtype = K.dtype
            shape = (rank,)
        class Vdummy:
            dtype = K.dtype
            shape = (len(K), rank)
        self._w, self._V = jax.pure_callback(
            lambda K: sparse.linalg.eigsh(numpy.asarray(K), k=rank, which='LM'),
            (wdummy, Vdummy), K,
        )
        
        # TODO try using jnp.matmul instead of passing the matrix and check
        # if it improves performance due to lower default comput precision
        
        # TODO try out lobpcg, which is also supported by jax (experimental)
    
    def correlate(self, b):
        return super().correlate(b[:len(self._w)])

def solve_triangular(a, b, lower=False):
    """
    Pure python implementation of scipy.linalg.solve_triangular for when
    a or b are object arrays.
    """
    # TODO maybe commit this to gvar.linalg
    a = numpy.asarray(a)
    x = numpy.copy(b)
    
    vec = x.ndim < 2
    if vec:
        x = x[:, None]

    n = a.shape[-1]
    assert x.shape[-2] == n
    
    if not lower:
        a = a[..., ::-1, ::-1]
        x = x[..., ::-1, :]

    x[..., 0, :] /= a[..., 0, 0, None]
    for i in range(1, n):
        x[..., i:, :] -= x[..., None, i - 1, :] * a[..., i:, i - 1, None]
        x[..., i, :] /= a[..., i, i, None]
    
    if not lower:
        x = x[..., ::-1, :]
    
    if vec:
        x = numpy.squeeze(x, -1)
    return x

def solve_triangular_auto(a, b, lower=False):
    """Works with b both object and non-object array"""
    if b.dtype == object:
        return solve_triangular(a, b, lower=lower)
    else:
        return jlinalg.solve_triangular(a, b, lower=lower)

class CholEps:
    
    def _eps(self, eps, mat, maxeigv):
        if eps is None:
            eps = len(mat) * jnp.finfo(_patch_jax.float_type(mat)).eps
        assert 0 <= eps < 1
        return eps * maxeigv    

class Chol(DecompAutoDiff, CholEps):
    """
    Cholesky decomposition.
    """
    
    def __init__(self, K):
        self._L = jlinalg.cholesky(K, lower=True, check_finite=False)
        with _patch_jax.skipifabstract():
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
        return _transpose(invLb) @ invLc
    
    def logdet(self):
        return 2 * jnp.sum(jnp.log(jnp.diag(self._L)))
    
    def correlate(self, b):
        return self._L @ b
    
    def decorrelate(self, b):
        return solve_triangular_auto(self._L, b, lower=True)

def _scale(a):
    """
    Compute a vector s of powers of 2 such that diag(a / outer(s, s)) ~ 1.
    """
    d = jnp.diag(a)
    return jnp.where(d, jnp.exp2(jnp.rint(0.5 * jnp.log2(d))), 1)

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
        return mat.at[jnp.diag_indices(len(mat))].add(self._eps(eps, mat, maxeigv))

class CholToeplitz(Chol):
    """
    Cholesky decomposition of a Toeplitz matrix. Only the first row of the
    matrix is read.
    """
    
    def __init__(self, K, eps=None):
        t = jnp.asarray(K[0])
        m = _toeplitz.eigv_bound(t)
        eps = self._eps(eps, t, m)
        t = t.at[0].add(eps)
        self._L = _toeplitz.chol(t)

class CholToeplitzML(DecompAutoDiff, CholEps):
    """
    Cholesky decomposition of a Toeplitz matrix. Only the first row of the
    matrix is read. It does not store the decomposition in memory, it is
    evaluated each time column by column during operations.
    """
    
    def __init__(self, K, eps=None):
        t = jnp.asarray(K[0, :])
        m = _toeplitz.eigv_bound(t)
        eps = self._eps(eps, t, m)
        self.t = t.at[0].add(eps)
    
    def solve(self, b):
        return _toeplitz.solve(self.t, b)
    
    def quad(self, b, c=None):
        t = self.t
        if c is not None and c.dtype == object:
            ilb = numpy.array(_toeplitz.chol_solve(t, b))
            ilc = _toeplitz.chol_solve_numpy(t, c)
        elif c is None:
            ilb = _toeplitz.chol_solve(t, b)
            ilc = ilb
        else:
            ilb, ilc = _toeplitz.chol_solve(t, b, c)
        return _transpose(ilb) @ ilc
    
    def logdet(self):
        return _toeplitz.logdet(self.t)
    
    def correlate(self, b):
        return _toeplitz.chol_matmul(self.t, b)
    
    def decorrelate(self, b):
        return _toeplitz.chol_solve(self.t, b)

class BlockDecomp(DecompPyTree):
    """
    Decomposition of a 2x2 symmetric block matrix using decompositions of the
    diagonal blocks.
    
    Reference: Gaussian Processes for Machine Learning, A.3, p. 201.
    """
    
    # This class can be used only starting from a seed block and adding
    # other blocks one at a time. Would a divide et impera approach be useful
    # for my case?
        
    def __init__(self, P_decomp, S, Q, S_decomp_class, **kw):
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
        **kw :
            Additional keyword arguments are passed to S_decomp_class.
        """
        self._Q = Q
        self._invP = P_decomp
        self._tildeS = S_decomp_class(S - P_decomp.quad(Q), **kw)
    
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
            return invP.quad(f) + tildeS.quad(QTinvPf) - fTinvPQtildeSg - _transpose(fTinvPQtildeSg) + tildeS.quad(g)
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
        y = _transpose(invP.decorrelate(Q)) @ f + tildeS.correlate(g)
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

class BlockDiagDecomp(DecompPyTree):
    
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

class SandwichQR(DecompPyTree):
    
    def __init__(self, A_decomp, B):
        """
        Decompose M = B A B^T with the QR decomposition of B.
        
        Parameters
        ----------
        A_decomp : Decomposition
            Instantiated decomposition of A.
        B : array
            The B matrix. Must be tall or square.
        
        """
        self._A = A_decomp
        assert B.shape[0] >= B.shape[1]
        self._B = B
        self._q, self._r = jnp.linalg.qr(B, mode='reduced')
    
    def solve(self, b):
        A, q, r = self._A, self._q, self._r
        rqb = jlinalg.solve_triangular(r, q.T @ b, lower=False)
        arqb = A.solve(rqb)
        return q @ jlinalg.solve_triangular(r.T, arqb, lower=True)
    
    def quad(self, b, c=None):
        A, q, r = self._A, self._q, self._r
        rqb = jlinalg.solve_triangular(r, q.T @ b, lower=False)
        if c is None:
            return A.quad(rqb)
        if c.dtype != object:
            rqc = jlinalg.solve_triangular(r, q.T @ c, lower=False)
        else:
            rqc = solve_triangular(r, numpy.matmul(q.T, c), lower=False)
        return A.quad(rqb, rqc)
    
    def logdet(self):
        A, r = self._A, self._r
        B_logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(r))))
        return A.logdet() + 2 * B_logdet
    
    def correlate(self, b):
        A, B = self._A, self._B
        return B @ A.correlate(b[:A.n])
    
    def decorrelate(self, b):
        A, q, r = self._A, self._q, self._r
        rqb = jlinalg.solve_triangular(r, q.T @ b, lower=False)
        return A.decorrelate(rqb)
    
    @property
    def n(self):
        return len(self._B)

def _rq(a, **kw):
    """
    Decompose a as r q, with r lower-triangular and q orthogonal
    Return: r, q
    """
    qt, rt = jnp.linalg.qr(_transpose(a), **kw)
    return _transpose(rt), _transpose(qt)

# TODO I am not able to define an efficient pinv of B A B^T with short B.
#
# class SandwichRQ(DecompPyTree):
#
#     def __init__(self, A_decomp, B):
#         """
#         Decompose M = B A B^T with the RQ decomposition of B.
#
#         Parameters
#         ----------
#         A_decomp : Decomposition
#             Instantiated decomposition of A.
#         B : array
#             The B matrix. Must be short or square.
#
#         """
#         self._A = A_decomp
#         assert B.shape[0] <= B.shape[1]
#         self._B = B
#         self._r, self._q = _rq(B, mode='reduced')
#
#     def solve(self, b):
#         A, q, r = self._A, self._q, self._r
#         qrb = q.T @ jlinalg.solve_triangular(r, b, lower=True)
#         aqrb = A.solve(qrb)
#         return jlinalg.solve_triangular(r.T, q @ aqrb, lower=False)
#
#     def quad(self, b, c=None):
#         A, q, r = self._A, self._q, self._r
#         qrb = q.T @ jlinalg.solve_triangular(r, b, lower=True)
#         if c is None:
#             return A.quad(qrb)
#         if c.dtype != object:
#             qrc = q.T @ jlinalg.solve_triangular(r, c, lower=True)
#         else:
#             qrc = numpy.matmul(q.T, solve_triangular(r, c, lower=True))
#         return A.quad(qrb, qrc)
#
#     def logdet(self):
#         A, r = self._A, self._r
#         B_logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(r))))
#         return A.logdet() + 2 * B_logdet
#         # I don't know if this really makes sense
#
#     def correlate(self, b):
#         A, B = self._A, self._B
#         return B @ A.correlate(b[:A.n])
#
#     def decorrelate(self, b):
#         A, q, r = self._A, self._q, self._r
#         qrb = q.T @ jlinalg.solve_triangular(r, b, lower=True)
#         return A.decorrelate(qrb)
#
#     @property
#     def n(self):
#         return len(self._B)

class SandwichSVD(DecompPyTree):
    
    def __init__(self, A_decomp, B):
        """
        Decompose M = B A B^T with the SVD decomposition of B.
        
        Parameters
        ----------
        A_decomp : Decomposition
            Instantiated decomposition of A.
        B : array
            The B matrix, must be tall or square.
        
        """
        self._A = A_decomp
        self._B = B
        self._u, self._s, self._vh = jnp.linalg.svd(B, full_matrices=False)
    
    def solve(self, b):
        A, u, s, vh = self._A, self._u, self._s, self._vh
        vsub = jnp.linalg.multi_dot([vh.T / s, u.T, b])
        vsu = (vh.T / s) @ u.T
        return A.quad(vsu, vsub)
    
    def quad(self, b, c=None):
        A, u, s, vh = self._A, self._u, self._s, self._vh
        vsub = jnp.linalg.multi_dot([vh.T / s, u.T, b])
        if c is None:
            return A.quad(vsub)
        if c.dtype != object:
            vsuc = jnp.linalg.multi_dot([vh.T / s, u.T, c])
        else:
            vsuc = numpy.linalg.multi_dot([vh.T / s, u.T, c])
        return A.quad(vsub, vsuc)
    
    def logdet(self):
        A, s = self._A, self._s
        B_logdet = jnp.sum(jnp.log(s))
        return A.logdet() + 2 * B_logdet
    
    def correlate(self, b):
        A, B = self._A, self._B
        return B @ A.correlate(b[:A.n])
    
    def decorrelate(self, b):
        A, u, s, vh = self._A, self._u, self._s, self._vh
        vsub = jnp.linalg.multi_dot([vh.T / s, u.T, b])
        return A.decorrelate(vsub)
    
    @property
    def n(self):
        return len(self._B)

class Woodbury(DecompPyTree):
    
    def __init__(self, A_decomp, B, C_decomp, decompcls, sign=1, **kw):
        """
        Decompose M = A ± B C B^T using Woodbury's formula, with M, A, and C
        positive definite. Very inaccurate if A and/or C are ill-conditioned.
        
        Parameters
        ----------
        A_decomp : Decomposition
            Instantiated decomposition of A.
        B : array
            The B matrix, can be rectangular.
        C_decomp : Decomposition
            Instantiated decomposition of C.
        decompcls : type
            Decomposition class used to decompose C^-1 ± B^T A^-1 B.
        sign : {-1, 1}
            The sign in the expression above, default 1.
        **kw :
            Keyword arguments are passed to `decompcls`'s initialization.
        
        """
        assert B.shape == (A_decomp.n, C_decomp.n)
        self._A = A_decomp
        self._B = B
        self._C = C_decomp
        X = C_decomp.inv() + sign * A_decomp.quad(B)
        self._X = decompcls(X, **kw)
        self._sign = sign
    
    def solve(self, b):
        A, B, X, s = self._A, self._B, self._X, self._sign
        ab = A.solve(b)
        AB = A.solve(B)
        bab = A.quad(B, b)
        xb = X.quad(AB.T, bab)
        return ab - s * xb
    
    def quad(self, b, c=None):
        A, B, X, s = self._A, self._B, self._X, self._sign
        if c is None:
            bab = A.quad(b)
            BAB = A.quad(B, b)
            bxb = X.quad(BAB)
            return bab - s * bxb
        else:
            bac = A.quad(b, c)
            BAB = A.quad(B, b)
            BAC = A.quad(B, c)
            bxc = X.quad(BAB, BAC)
            return bac - s * bxc
    
    def logdet(self):
        A, C, X = self._A, self._C, self._X
        return A.logdet() + C.logdet() + X.logdet()
    
    def correlate(self, b):
        raise NotImplementedError
    
    def decorrelate(self, b):
        raise NotImplementedError
    
    @property
    def n(self):
        return self._A.n
