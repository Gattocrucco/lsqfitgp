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

Decompositions of nonnegative definite matrices.

A decomposition object is initialized with a square symmetric matrix and then
can solve linear systems and do other stuff with that matrix. The matrix is
assumed to be positive semidefinite in principle but not numerically,
decompositions differ on how the eventual numerical degeneracy is handled.

It is intended that the matrix inverse is a Moore-Penrose pseudoinverse in case
of (numerically) singular matrices. Some decompositions do not support
pseudoinversion, i.e., the matrix must be well-conditioned.

Abstract classes
----------------
Decomposition
    Abstract base class.
DecompPyTree
    Abstract subclass that register subclasses as jax pytrees.
DecompAutoDiffBase
    Abstract subclass that adds decomposition-independent jax-autodiff support.
DecompAutoDiff
    Specialization of DecompAutoDiffBase for classes which take as
    initialization input a single matrix object.
Diag
    Diagonalization.
LowRankDiag
    Diagonalization with spectrum truncation.

Concrete classes
----------------
EigCutFullRank
    Diagonalization rounding up small (or negative) eigenvalues.
EigCutLowRank
    Diagonalization removing small (or negative) eigenvalues.
SVDCutFullRank
    Diagonalization rounding up small eigenvalues, keeping their sign.
SVDCutLowRank
    Diagonalization removing small eigenvalues.
Lanczos, LOBPCG
    Partial diagonalization with higher eigenvalues only.
Chol
    Cholesky regularized using an estimate of the maximum eigenvalue.
CholToeplitz
    Cholesky decomposition of a Toeplitz matrix regularized using an estimate
    of the maximum eigenvalue. Uses Schur's and Levisons' algorithms. Does not
    store the Cholesky factor in memory.

Composite decompositions
------------------------
Block
    Decompose a 2x2 partitioned matrix.
BlockDiag
    Decompose a 2x2 block diagonal matrix.
SandwichQR, SandwichSVD
    Decompose BAB'.
Woodbury, Woodbury2
    Decompose A ± BCB'.
Pinv, Pinv2
    Turn any decomposition into one that supports pseudoinversion.

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
    
    Abstract base class for nonnegative definite symmetric matrix decomposition.
    
    Methods
    -------
    solve
    quad
    diagquad
    logdet
    tracesolve
    correlate
    decorrelate
    inv
    
    Properties
    ----------
    n
    m
    
    """
    
    @abc.abstractmethod
    def __init__(self, *args, **kw): # pragma: no cover
        """
        Decompose a symmetric nonnegative definite matrix K.
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def solve(self, b): # pragma: no cover
        """
        Compute K⁺b.
        """
        raise NotImplementedError
    
    def quad(self, b, c=None):
        """
        Compute the quadratic form b'K⁺b if c is not specified, else b'K⁺c.
        `c` can be an array of gvars.
        """
        if c is None:
            return _transpose(b) @ self.solve(b)
        elif c.dtype == object:
            return numpy.asarray(_transpose(self.solve(b))) @ c
        else:
            return _transpose(b) @ self.solve(c)
    
    def diagquad(self, b):
        """
        Compute diag(b'K⁺b).
        """
        result = self.quad(b)
        if result.ndim >= 2:
            result = jnp.diagonal(result, 0, -2, -1)
        return result
        
    # TODO to compute efficiently the likelihood with things like toeplitz and
    # other memoryless solvers to be added in the future, add quad_and_logdet.
    
    @abc.abstractmethod
    def logdet(self): # pragma: no cover
        """
        Compute log det K.
        """
        raise NotImplementedError
        
        # TODO adopt a clear convention: if the matrix is degenerate, it's
        # log pdet + regularization to make it continuous => problem: when the
        # regularization changes with the maximum eigenvalue, this gives a
        # large logdet variation which does not make sense. The regularization
        # should be fixed. Whatevs, the derivatives won't take this variation
        # into account. Maybe I should use a fixed eps? Could that be one of
        # the sources of problems in the hyperparameter fit, e.g., the periodic
        # fit in the tests? Alternative: use a separate fixed eps for the
        # regularization. This does not give continuity, but should give
        # coherent comparability both at the same and different ranks, if the
        # fixed eps is small enough. Alternative: is there a way to correct
        # dynamically the regularization based on the previous ones? The decomp
        # could provide some opaque reg info object, that is to be optionally
        # fed to logdet. Maybe it can be made decomposition-agnostic.
        #
        # => change the current methods' name to logpdet, then make a new one
    
    def tracesolve(self, b):
        """
        Compute tr(K⁺b).
        """
        # return jnp.trace(self.solve(b))
        return jnp.einsum('ij,ji', self.inv(), b)
    
    @abc.abstractmethod
    def correlate(self, b, *, transpose=False): # pragma: no cover
        """
        Compute Ab where K = AA', with A n x m. If b represents iid
        variables with unitary variance, Ab has covariance matrix K. If
        transpose=True, compute A'b.
        """
        raise NotImplementedError
    
    def decorrelate(self, b, *, transpose=False):
        """
        Compute A⁺b, where K = AA', with A n x m. If b represents variables
        with covariance matrix K, A⁺b has idempotent covariance A⁺A. If
        `transpose=True`, compute A⁺'b.
        """
        # K = AA'
        # A⁺ = A'(AA')⁺ = A'K⁺
        # A⁺' = K⁺A
        if transpose:
            return self.solve(self.correlate(b))
        else:
            return self.correlate(self.solve(b), transpose=True)
    
    def inv(self):
        """
        Invert the matrix.
        """
        
        # TODO currently no subclass overrides this, some should probably do.
        
        return self.quad(jnp.eye(self.n))
    
    @property
    def n(self):
        """
        The size of the matrix.
        """
        return len(self.matrix())
    
    @property
    def m(self):
        """
        The inner size of the decomposition used by `correlate` and
        `decorrelate`.
        """
        return self.n
    
    @abc.abstractmethod
    def matrix(self): # pragma: no cover
        """
        Return K.
        """
        raise NotImplementedError
    
    # TODO new properties: rank is the rank for low-rank or rank revealing
    # decompositions, otherwise n.
    
    def _parseeps(self, mat, epsrel, epsabs, maxeigv=None):
        machine_eps = jnp.finfo(_patch_jax.float_type(mat)).eps
        if epsrel == 'auto':
            epsrel = len(mat) * machine_eps
        if epsabs == 'auto':
            epsabs = machine_eps
        if maxeigv is None:
            maxeigv = _gershgorin_eigval_bound(mat)
        self._eps = epsrel * maxeigv + epsabs
        return self._eps
    
    @property
    def eps(self):
        """
        The threshold below which eigenvalues are too small to be determined.
        """
        return self._eps

class DecompPyTree(Decomposition, _pytree.AutoPyTree):
    """
    Marks a decomposition as JAX PyTree, with automatic detection of which
    members are to be traced by JAX.
    """
    pass

class DecompAutoDiffBase(DecompPyTree):
    """
    Abstract subclass adding JAX autodiff support to subclasses of
    Decomposition, even if the decomposition algorithm is not supported by JAX.
    Concrete subclasses have to define _matrix and _map_traceable_init_args.
    """
    
    @abc.abstractmethod
    def _matrix(self, *args, **kw): # pragma: no cover
        """ Given the initialization arguments, return the matrix to be
        decomposed """
        raise NotImplementedError
            
    @abc.abstractmethod
    def _map_traceable_init_args(self, fun, *args, **kw): # pragma: no cover
        """
        
        Given a function Any -> Any, and the initialization arguments,
        apply the function only to those arguments which are considered inputs
        to the definition of the matrix. The return layout must be stable
        under repeated application, i.e., after
        
            a1, k1 = self._map_traceable_init_args(lambda x: x, *a0, **k0)
            a2, k2 = self._map_traceable_init_args(lambda x: x, *a1, **k1)
        
        it must be a1 == a2 and k1 == k2.
        
        """
        raise NotImplementedError
        
    def _store_args(self, args, kw):
        """ store initialization arguments """
        class Marker: pass
        args, kw = self._map_traceable_init_args(lambda x: x, *args, **kw)
        # apply _map_traceable_init_args twice to have a consistent placement
        # of arguments between args, kw and margs, mkw
        margs, mkw = self._map_traceable_init_args(lambda _: Marker, *args, **kw)
        self._dad_jax_args_ = {
            i: v for i, (v, m) in enumerate(zip(args, margs)) if m is Marker
        }
        self._dad_other_args_ = {
            i: v for i, (v, m) in enumerate(zip(args, margs)) if m is not Marker
        }
        self._dad_jax_kw_ = {
            k: v for k, v in kw.items() if mkw[k] is Marker
        }
        self._dad_other_kw_ = {
            k: v for k, v in kw.items() if mkw[k] is not Marker
        }
    
    def _jax_vars(self):
        """ tell AutoPyTree about additional variables to be considered
        children """
        return super()._jax_vars() + ['_dad_jax_args_', '_dad_jax_kw_']
    
    def _get_args(self):
        """ get initialization arguments saved by _store_args """
        args = dict(self._dad_jax_args_)
        args.update(self._dad_other_args_)
        args = tuple(args[i] for i in sorted(args))
        kw = dict(**self._dad_jax_kw_, **self._dad_other_kw_)
        return args, kw
    
    def matrix(self):
        args, kw = self._get_args()
        return self._matrix(*args, **kw)
        
    def _getarg(self, kw, attr, default, nextguy):
        """ saves an initialization argument as instance attribute, and decide
        if the argument is to be passed to the subclass __init__ """
        if not hasattr(self, attr):
            setattr(self, attr, kw.get(attr, default))
            if not self._hasarg(nextguy, attr):
                kw.pop(attr, None)
    
    @staticmethod
    def _hasarg(func, arg):
        """ check if callable func accepts argument keyword argument arg """
        try:
            inspect.signature(func).bind_partial(**{arg: None})
        except TypeError:
            return False
        else:
            return True
    
    # TODO try using jax.linearize and jax.convert_closure. Or: try to pass
    # the decomposition products as variadic arguments to the intermediate
    # wrapper, maybe this solves all reverse-mode autodiff problems like
    # the double undefined primals.
        
    def __init_subclass__(cls, **kw):
        """ wrap the subclass methods to implement automatic differentiation """
        
        super().__init_subclass__(**kw)
        
        old__init__ = cls.__init__

        if inspect.isabstract(cls):
            # Do not wrap the class methods if it's still abstract, because
            # __init_subclass__ will necessarily be invoked again on the
            # concrete subclasses. But wrap the __init__ to drop
            # DecompAutoDiffBase additional arguments eventually passed to
            # super().__init__ from the subclass.
            
            argnames = ['direct_autodiff', 'stop_hessian']
            missing_names = [
                name for name in argnames
                if not cls._hasarg(old__init__, name)
            ]
            if missing_names:
                @functools.wraps(old__init__)
                def __init__(self, *args, **kw):
                    for name in missing_names:
                        kw.pop(name, None)
                    old__init__(self, *args, **kw)
                cls.__init__ = __init__
            
            return
        
        @functools.wraps(old__init__)
        def __init__(self, *args, **kw):
            
            # I can't avoid double wrapping because the superclass __init__ is
            # wrapped and may get called with super(), so I use a flag to check
            # if we are an __init__ called by the old__init__ of a subclass
            if getattr(self, '_dad_initialized_', False):
                return old__init__(self, *args, **kw)
            self._dad_initialized_ = True
            
            self._getarg(kw, 'direct_autodiff', False, old__init__)
            self._getarg(kw, 'stop_hessian', False, old__init__)
            
            if self.direct_autodiff and self.stop_hessian:
                nargs, nkw = self._map_traceable_init_args(_patch_jax.stop_hessian, *args, **kw)
                old__init__(self, *nargs, **nkw)
            elif self.direct_autodiff:
                old__init__(self, *args, **kw)
            else:
                nargs, nkw = self._map_traceable_init_args(jax.lax.stop_gradient, *args, **kw)
                old__init__(self, *nargs, **nkw)
                if self.stop_hessian:
                    args, kw = self._map_traceable_init_args(_patch_jax.stop_hessian, *args, **kw)
            
            self._store_args(args, kw)
        
        cls.__init__ = __init__
        
        for name in 'solve', 'quad', 'logdet', 'tracesolve':
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
            assert not self.direct_autodiff
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
                return solve_autodiff(self, self.matrix(), b)
        
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
        # decomposition function sandwich(b, X, c) which computes b.T K^-1 X
        # K^-1 c and use it both for numerical accuracy (I'm guessing that
        # avoiding the two concatenated solves within solve is good) and to make
        # the jvp work. For things like ToeplitzML that stream-decompose each
        # time, it would be more efficient to have versions of solve and quad
        # with an arbitrary number of right-hand sides. => I could have
        # a generic operation sandwich(b, X1, X2, ..., c) that computes
        # b K^-1 X1 K^-1 X2 K^-1 ... K^-1 c, such that its derivatives can
        # be defined in terms of itself, but handling repeated arguments is
        # cumbersome.

        @quad_autodiff.defjvp
        def quad_jvp(self, primals, tangents):
            assert not self.direct_autodiff
            K, b, c = primals
            K_dot, b_dot, c_dot = tangents
            primal = quad_autodiff(self, K, b, c)
            # tangent_K = -quad_autodiff(self, K, b, quad_autodiff(self, K, K_dot, c))
            Kb = self.solve._autodiff(self, K, b)
            Kc = self.solve._autodiff(self, K, c)
            tangent_K = -_transpose(Kb) @ K_dot @ Kc
            tangent_b = quad_autodiff(self, K, b_dot, c)
            # tangent_b = _transpose(b_dot) @ Kc
            tangent_c = quad_autodiff(self, K, b, c_dot)
            # tangent_c = _transpose(Kb) @ c_dot
            return primal, tangent_K + tangent_b + tangent_c
            
        @quad_autodiff_cnone.defjvp
        def quad_cnone_jvp(self, primals, tangents):
            assert not self.direct_autodiff
            K, b = primals
            K_dot, b_dot = tangents
            primal = quad_autodiff_cnone(self, K, b)
            # tangent_K = -quad_autodiff(self, K, b, quad_autodiff(self, K, K_dot, b))
            Kb = self.solve._autodiff(self, K, b)
            tangent_K = -_transpose(Kb) @ K_dot @ Kb
            # using solve + matmul instead of double for tangent_K quad does
            # only O(n^2) matrix-vector multiplications instead of O(n^3)
            # matrix-matrix, but my intuition suggests it could be less
            # numerically accurate
            tangent_b = quad_autodiff(self, K, b_dot, b)
            # tangent_b = _transpose(b_dot) @ Kb
            return primal, tangent_K + tangent_b + _transpose(tangent_b)

        def quad(self, b, c=None):
            if self.direct_autodiff:
                return oldquad(self, b, c)
            elif c is None:
                return quad_autodiff_cnone(self, self.matrix(), b)
            elif c.dtype == object:
                return oldquad(self, b, c)
                # if c contains gvars, the output is an array of gvars, and
                # quad_autodiff would hang because custom_jvp functions are
                # checked not to return object dtypes, so I have to call
                # oldquad
            else:
                return quad_autodiff(self, self.matrix(), b, c)

        quad._autodiff = quad_autodiff
        
        return quad
    
    @staticmethod
    def _make_logdet(oldlogdet):
        
        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def logdet_autodiff(self, K):
            return oldlogdet(self)
                        
        @logdet_autodiff.defjvp
        def logdet_jvp(self, primals, tangents):
            assert not self.direct_autodiff
            K, = primals
            K_dot, = tangents
            primal = logdet_autodiff(self, K)
            # tangent = jnp.trace(self.solve._autodiff(self, K, K_dot))
            tangent = self.tracesolve._autodiff(self, K, K_dot)
            return primal, tangent
        
        def logdet(self):
            if self.direct_autodiff:
                return oldlogdet(self)
            else:
                return logdet_autodiff(self, self.matrix())
        
        return logdet

    @staticmethod
    def _make_tracesolve(oldtracesolve):

        @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
        def tracesolve_autodiff(self, K, b):
            return oldtracesolve(self, b)
                        
        @tracesolve_autodiff.defjvp
        def tracesolve_jvp(self, primals, tangents):
            assert not self.direct_autodiff
            K, b = primals
            K_dot, b_dot = tangents
            primal = tracesolve_autodiff(self, K, b)
            tangent_K = -tracesolve_autodiff(self, K, self.quad._autodiff(self, K, K_dot, b))
            tangent_b = tracesolve_autodiff(self, K, b_dot)
            return primal, tangent_K + tangent_b
        
        def tracesolve(self, b):
            if self.direct_autodiff:
                return oldtracesolve(self, b)
            else:
                return tracesolve_autodiff(self, self.matrix(), b)

        tracesolve._autodiff = tracesolve_autodiff
        
        return tracesolve

class DecompAutoDiff(DecompAutoDiffBase):
    """ Specialization of DecompAutoDiffBase for classes which take as
    initialization input a single matrix object """
    
    def _matrix(self, K, *_, **__):
        return K
    
    def _map_traceable_init_args(self, fun, K, *args, **kw):
        return (fun(K), *args), kw
        
class Diag(DecompAutoDiff):
    """
    Diagonalization.
    """
    
    @abc.abstractmethod
    def __init__(self, K, *, epsrel='auto', epsabs=0):
        self._w, self._V = jlinalg.eigh(K)
        self._parseeps(K, epsrel, epsabs, jnp.max(jnp.abs(self._w)))
    
    def solve(self, b):
        return (self._V / self._w) @ (self._V.T @ b)
    
    def quad(self, b, c=None):
        VTb = self._V.T @ b
        bTVw = _transpose(VTb) / self._w
        if c is None:
            VTc = VTb
        elif c.dtype == object:
            VTc = numpy.array(self._V).T @ c
            bTVw = numpy.array(bTVw)
        else:
            VTc = self._V.T @ c
        return bTVw @ VTc
    
    def logdet(self):
        return jnp.sum(jnp.log(jnp.abs(self._w)))
    
    def correlate(self, b, *, transpose=False):
        A = self._V * jnp.sqrt(jnp.abs(self._w))
        if transpose:
            A = A.T
        return A @ b
    
    def decorrelate(self, b, *, transpose=False):
        A_1 = (self._V / jnp.sqrt(jnp.abs(self._w))).T
        if transpose:
            A_1 = A_1.T
        return A_1 @ b
        
# TODO merge the diagonalizations into a single class Diag with options
# rank=None, if specified applies before epsilon cut, inflate: bool, signed:
# bool. Default inflate=True (to avoid ignoring incompatible y and breaking the
# continuity of the logdet), signed=True (to provide an accurate solve and
# quad). In the GP interface, make a single method name 'diag'.
    
class EigCutFullRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are set to `eps`, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        self._w = jnp.where(self._w < self.eps, self.eps, self._w)
            
class EigCutLowRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are removed, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        cond = self._w < self.eps
        self._w = jnp.where(cond, 1, self._w)
        self._V = jnp.where(cond, 0, self._V)
        
        # TODO rank parameter to fix regularization

class SVDCutFullRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` in absolute value are set to
    `eps` with their sign, where `eps` is relative to the largest eigenvalue.
    """
    
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        cond = jnp.abs(self._w) < self.eps
        self._w = jnp.where(cond, self.eps * jnp.sign(self._w), self._w)
        
class SVDCutLowRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` in absolute value are removed,
    where `eps` is relative to the largest eigenvalue.
    """
    
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        cond = jnp.abs(self._w) < self.eps
        self._w = jnp.where(cond, 1, self._w)
        self._V = jnp.where(cond, 0, self._V)

class LowRankDiag(Diag):
    """
    Diagonalization with a fixed size truncation of the spectrum.
    """
    
    @abc.abstractmethod
    def __init__(self, K, *, rank=None):
        if rank is None:
            raise ValueError('rank not specified')
        self._rank = rank
    
    @property
    def m(self):
        return self._rank

class Lanczos(LowRankDiag):
    """
    Keep only the first `rank` higher eigenmodes.
    """
    
    def __init__(self, K, *, rank=None):
        super().__init__(K, rank=rank)
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
        # if it improves performance due to lower default comput precision =>
        # behavior under jit not clear
        
        # TODO try using linalg.blas.get_blas_func('symv', [K]) for the matrix
        # vector product. sparse.linalg does not seem to take into account
        # that the matrix is symmetric to do the matvec product.
        
class LOBPCG(LowRankDiag):
    """
    Keep only the first `rank` higher eigenmodes.
    """
    
    def __init__(self, K, *, rank=None):
        super().__init__(K, rank=rank)
        class wdummy:
            dtype = K.dtype
            shape = (rank,)
        class Vdummy:
            dtype = K.dtype
            shape = (len(K), rank)
        self._w, self._V = jax.pure_callback(
            lambda K: sparse.linalg.lobpcg(numpy.asarray(K), numpy.random.randn(*Vdummy.shape)),
            (wdummy, Vdummy), K,
        )
        
        # TODO try using jax's own implementation of LOBPCG

# TODO look into the pivoted Cholesky decomposition from Gardner et al. 2018

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

def _gershgorin_eigval_bound(mat):
    """
    Upper bound on the largest magnitude eigenvalue of the matrix.
    """
    return jnp.max(jnp.sum(jnp.abs(mat), axis=1))

# TODO consider scipy.linalg.lapack.dpstrf that does cholesky pivoted up to a
# remainder nondef block. on 1000x1000 it takes 2.1x cholesky worst case, i.e.,
# when it decomposes the full matrix, so it's fast. 

class Chol(DecompAutoDiff):
    """
    Cholesky decomposition.
    """
    
    @staticmethod
    def _scale(K):
        """
        Compute a vector s of powers of 2 such that diag(K / outer(s, s)) ~ 1.
        """
        d = jnp.diag(K)
        return jnp.where(d, jnp.exp2(jnp.rint(0.5 * jnp.log2(d))), 1)

    def __init__(self, K, *, epsrel='auto', epsabs=0):
        s = self._scale(K)
        K = K / s / s[:, None]
        eps = self._parseeps(K, epsrel, epsabs)
        K = K.at[jnp.diag_indices(len(K))].add(eps)
        L = jlinalg.cholesky(K, lower=True)
        with _patch_jax.skipifabstract():
            if not jnp.all(jnp.isfinite(L)):
                raise numpy.linalg.LinAlgError('cholesky decomposition not finite, probably matrix not pos def numerically')
        self._L = L * s[:, None]
    
    def solve(self, b):
        invLb = jlinalg.solve_triangular(self._L, b, lower=True)
        return jlinalg.solve_triangular(self._L.T, invLb, lower=False)
    
    def quad(self, b, c=None):
        invLb = jlinalg.solve_triangular(self._L, b, lower=True)
        if c is None:
            invLc = invLb
        elif c.dtype == object:
            invLc = solve_triangular(self._L, c, lower=True)
            invLb = numpy.asarray(invLb)
        else:
            invLc = jlinalg.solve_triangular(self._L, c, lower=True)
        return _transpose(invLb) @ invLc
    
    def logdet(self):
        return 2 * jnp.sum(jnp.log(jnp.diag(self._L)))
    
    def correlate(self, b, *, transpose=False):
        return (self._L.T if transpose else self._L) @ b
    
    def decorrelate(self, b, *, transpose=False):
        L = self._L.T if transpose else self._L
        return jlinalg.solve_triangular(L, b, lower=not transpose)

class CholToeplitz(DecompAutoDiff):
    """
    Cholesky decomposition of a Toeplitz matrix. Only the first row of the
    matrix is read. It does not store the decomposition in memory, it is
    evaluated each time column by column during operations.
    """
    
    def __init__(self, K, *, epsrel='auto', epsabs=0):
        t = jnp.asarray(K[0, :])
        m = _toeplitz.eigv_bound(t)
        eps = self._parseeps(t, epsrel, epsabs, m)
        self.t = t.at[0].add(eps)
    
    def solve(self, b):
        return _toeplitz.solve(self.t, b)
    
    def quad(self, b, c=None):
        t = self.t
        if c is None:
            ilb = _toeplitz.chol_solve(t, b)
            ilc = ilb
        elif c.dtype == object:
            ilb = numpy.array(_toeplitz.chol_solve(t, b))
            ilc = _toeplitz.chol_solve_numpy(t, c)
        else:
            ilb, ilc = _toeplitz.chol_solve(t, b, c)
        return _transpose(ilb) @ ilc
    
    def logdet(self):
        return _toeplitz.logdet(self.t)
    
    def correlate(self, b, *, transpose=False):
        if transpose:
            return _toeplitz.chol_transp_matmul(self.t, b)
        else:
            return _toeplitz.chol_matmul(self.t, b)
    
    def decorrelate(self, b, *, transpose=False):
        if transpose:
            return _toeplitz.chol_transp_solve(self.t, b)
        else:
            return _toeplitz.chol_solve(self.t, b)

class Block(DecompAutoDiffBase):
    """
    Decomposition of a 2x2 symmetric block matrix using decompositions of the
    diagonal blocks.
    """
    
    # For A- it works, see schott 2017, theorem 7.12, page 298
    # For A+, it needs to be modified, see schott 2017, theorem 7.13, page 300
    # For a sufficient condition for working with A+ as written, see schott
    # 2017, exercise 7.28, page 311
    
    def _matrix(self, P_decomp, S, Q, S_decomp_class, **kw):
        assert not self.direct_autodiff
        return jnp.block([[P_decomp.matrix(), Q], [Q.T, S]])
            
    def _map_traceable_init_args(self, fun, P_decomp, S, Q, S_decomp_class, **kw):
        return (fun(P_decomp), fun(S), fun(Q), S_decomp_class), kw

    def __init__(self, P_decomp, S, Q, decompcls, **kw):
        """
        The matrix to be decomposed is
        
            K = [P   Q]
                [Q'  S]
        
        Parameters
        ----------
        P_decomp : Decomposition
            An instantiated decomposition of P.
        S, Q : matrices
            The other blocks.
        decompcls : type
            A subclass of Decomposition used to decompose S - Q'P⁻¹Q.
        **kw :
            Additional keyword arguments are passed to `decompcls`.
        """
        self._Q = Q
        self._invP = P_decomp
        self._tildeS = decompcls(S - P_decomp.quad(Q), **kw)
        
        # TODO does it make sense to allow to use woodbury on the schur
        # complement of P? Maybe to be general I should take an instantiated
        # decomposition schurP_decomp instead of a class. Or maybe detect
        # a woodbury class with a woodbury ABC and pass arguments appropriately.
    
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
        # TODO Is this valid for the pseudodeterminant? It isn't in general, but
        # it could be for p.s.d. matrices. schott 2017, theorem 7.10, page 297
        # shows that the rank decomposes in this way, so maybe the pdet does
        # too.
    
    @property
    def eps(self):
        return jnp.maximum(self._invP.eps, self._tildeS.eps())
    
    def correlate(self, b, *, transpose=False):
        # Block Cholesky decomposition:
        # K = [P   Q] = LL'
        #     [Q'  S]
        # L = [A   ]        L' = [A'  B']
        #     [B  C]             [    C']
        # AA' = P
        # AB' = Q  ==>  B' = AQ
        # BB' + CC' = S  ==>  CC' = S - Q'P⁻¹Q = S̃
        invP = self._invP
        tildeS = self._tildeS
        Q = self._Q
        split = len(Q) if transpose else invP.m
        f = b[:split]
        g = b[split:]
        if transpose:
            x = invP.correlate(f, transpose=True) + invP.decorrelate(Q @ g)
            y = tildeS.correlate(g, transpose=True)
        else:
            x = invP.correlate(f)
            y = Q.T @ invP.decorrelate(f, transpose=True) + tildeS.correlate(g)
        return jnp.concatenate([x, y], axis=max(0, b.ndim - 2))
    
    def decorrelate(self, b, *, transpose=False):
        # L⁻¹ = [   A⁻¹       ]     L'⁻¹ = [A'⁻¹  -A'⁻¹B'C'⁻¹]
        #       [-C⁻¹BA⁻¹  C⁻¹]            [          C'⁻¹   ]
        invP = self._invP
        tildeS = self._tildeS
        Q = self._Q
        split = invP.m if transpose else len(Q)
        f = b[:split]
        g = b[split:]
        if transpose:
            y = tildeS.decorrelate(g, transpose=True)
            x = invP.decorrelate(f - invP.decorrelate(Q @ y), transpose=True)
        else:
            x = invP.decorrelate(f)
            y = tildeS.decorrelate(g - invP.quad(Q, f))
        return jnp.concatenate([x, y], axis=max(0, b.ndim - 2))
        
    @property
    def n(self):
        return sum(self._Q.shape)
    
    @property
    def m(self):
        return self._invP.m + self._tildeS.m
    
    # to compute the rank, use schott 2017, theorem 7.10, page 297
    
class BlockDiag(DecompAutoDiffBase):
    
    # TODO allow NxN instead of 2x2?
    
    def _matrix(self, A_decomp, B_decomp):
        assert not self.direct_autodiff
        return jlinalg.block_diag(A_decomp.matrix(), B_decomp.matrix())
            
    def _map_traceable_init_args(self, fun, A_decomp, B_decomp):
        return (fun(A_decomp), fun(B_decomp)), {}

    def __init__(self, A_decomp, B_decomp):
        """
        
        Decomposition of a 2x2 block diagonal matrix.
        
        The matrix is
        
            K = [A   ]
                [   B]
        
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
    
    @property
    def eps(self):
        return jnp.maximum(self._A.eps, self._B.eps())
    
    def correlate(self, b, *, transpose=False):
        A = self._A
        B = self._B
        An = A.n if transpose else A.m
        f = b[:An]
        g = b[An:]
        return jnp.concatenate([
            A.correlate(f, transpose=transpose),
            B.correlate(g, transpose=transpose),
        ], axis=max(0, b.ndim - 2))
    
    def decorrelate(self, b, *, transpose=False):
        A = self._A
        B = self._B
        An = A.m if transpose else A.n
        f = b[:An]
        g = b[An:]
        return jnp.concatenate([
            A.decorrelate(f, transpose=transpose),
            B.decorrelate(g, transpose=transpose),
        ], axis=max(0, b.ndim - 2))

    @property
    def n(self):
        return self._A.n + self._B.n
    
    @property
    def m(self):
        return self._A.m + self._B.m

class SandwichQR(DecompAutoDiffBase):
    
    def _matrix(self, A_decomp, B):
        assert not self.direct_autodiff
        return B @ A_decomp.matrix() @ B.T
            
    def _map_traceable_init_args(self, fun, A_decomp, B):
        return (fun(A_decomp), fun(B)), {}

    def __init__(self, A_decomp, B):
        """
        Decompose K = BAB' with the QR decomposition of B.
        
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
        # Do the QR of A.correlate(B.T, T).T and drop _A
    
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
    
    @property
    def eps(self):
        return self._A.eps # TODO take B into account somehow
    
    def correlate(self, b, *, transpose=False):
        A, B = self._A, self._B
        if transpose:
            return A.correlate(B.T @ b, transpose=True)
        else:
            return B @ A.correlate(b)
    
    def decorrelate(self, b, *, transpose=False):
        # BAB' = BL(BL)'
        # BL = QRL
        # (QRL)⁺ = L⁻¹R⁻¹Q' (valid because Q has orthonormal columns)
        # (QRL)⁺' = QR'⁻¹L'⁻¹
        A, q, r = self._A, self._q, self._r
        if transpose:
            lb = A.decorrelate(b, transpose=True)
            return q @ jlinalg.solve_triangular(r.T, lb, lower=True)
        else:
            rqb = jlinalg.solve_triangular(r, q.T @ b, lower=False)
            return A.decorrelate(rqb)
    
    @property
    def n(self):
        return len(self._B)
    
    @property
    def m(self):
        return self._A.m
    
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

class SandwichSVD(DecompAutoDiffBase):
    
    # TODO to compute the pseudoinverse, compute C = A.correlate(B.T,
    # transpose=True).T to have M = CCt, then compute C+ to have M+ = C+t C+.
    
    # TODO in a new class: use C+ = (CtC)+ Ct, and decompose CtC with a
    # user-provided class. Make A optional, if A=None, then C = B and M = BBt.
    # If C is short, use C+ = Ct (CCt)+.
    
    def _matrix(self, A_decomp, B):
        assert not self.direct_autodiff
        return B @ A_decomp.matrix() @ B.T
            
    def _map_traceable_init_args(self, fun, A_decomp, B):
        return (fun(A_decomp), fun(B)), {}

    def __init__(self, A_decomp, B):
        """
        Decompose K = BAB' with the SVD decomposition of B.
        
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
        # TODO regularize s
    
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
        return A.logdet() + 2 * B_logdet # makes sense if A invertible, in
        # general after I svd an empty sandwich
    
    @property
    def eps(self):
        return self._A.eps # TODO take B into account somehow
        
    def correlate(self, b, *, transpose=False):
        A, B = self._A, self._B
        if transpose:
            return A.correlate(B.T @ b, transpose=True)
        else:
            return B @ A.correlate(b)
    
    def decorrelate(self, b, *, transpose=False):
        # BAB' = BL(BL)'
        # BL = USV'L
        # (BL)⁺ = L⁻¹VS⁻¹U'
        # (BL)⁺' = US⁻¹V'L'⁻¹
        A, u, s, vh = self._A, self._u, self._s, self._vh
        if transpose:
            lb = A.decorrelate(b, transpose=True)
            return (u / s) @ (vh @ lb)
        else:
            vsub = (vh.T / s) @ (u.T @ b)
            return A.decorrelate(vsub)
    
    @property
    def n(self):
        return len(self._B)
    
    @property
    def m(self):
        return self._A.m
    
class Woodbury(DecompAutoDiffBase):
    
    def _matrix(self, A_decomp, B, C_decomp, decompcls, sign=1, **kw):
        assert not self.direct_autodiff
        return A_decomp.matrix() + sign * B @ C_decomp.matrix() @ B.T
            
    def _map_traceable_init_args(self, fun, A_decomp, B, C_decomp, decompcls, sign=1, **kw):
        return (fun(A_decomp), fun(B), fun(C_decomp), decompcls, sign), kw

    def __init__(self, A_decomp, B, C_decomp, decompcls, sign=1, **kw):
        """

        Decompose K = A ± BCB' using Woodbury's formula. K, A, C > 0.
        
        Parameters
        ----------
        A_decomp : Decomposition
            Instantiated decomposition of A.
        B : array
            The B matrix, can be rectangular.
        C_decomp : Decomposition
            Instantiated decomposition of C.
        decompcls : type
            Decomposition class used to decompose C⁻¹ ± B'A⁻¹B'.
        sign : {-1, 1}
            The sign in the expression above, default 1.
        **kw :
            Keyword arguments are passed to the initialization of `decompcls`.
        
        """
        assert B.shape == (A_decomp.n, C_decomp.n)
        assert sign in (-1, 1)
        self._A = A_decomp
        self._B = B
        self._C = C_decomp
        X = C_decomp.inv() + sign * A_decomp.quad(B)
        self._X = decompcls(X, **kw)
        self._sign = sign
    
    # (A ± BCB')⁻¹
    #   = A⁻¹ ∓ A⁻¹B (C⁻¹ ± B'A⁻¹B)⁻¹ B'A⁻¹
    #                 ~~~~~~~~~~~~
    #                    =: X
    #   = A⁻¹ ∓ A⁻¹B X⁻¹ B'A⁻¹
    
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
    
    @property
    def eps(self):
        A, C, X = self._A, self._C, self._X
        return jnp.max(jnp.array([A.eps, C.eps, X.eps]))
    
    def correlate(self, b, *, transpose=False):
        # A = LL'
        # C = NN'
        # A + BCB' = LL' + BNN'B' = [L BN] [L BN]'
        assert self._sign > 0
        A, B, C = self._A, self._B, self._C
        if transpose:
            return jnp.concatenate([
                A.correlate(b, transpose=True),
                C.correlate(B.T @ b, transpose=True)
            ], axis=max(0, b.ndim - 2))
        else:
            return A.correlate(b[:A.m]) + B @ C.correlate(b[A.m:])
    
    def decorrelate(self, b, *, transpose=False):
        # V = [L BN]
        # V⁺ = V'(VV')⁻¹
        #    = V'K⁻¹ =
        #    = [L'; N'B'] (A⁻¹ - A⁻¹B X⁻¹ B'A⁻¹) =
        #    = [ L⁻¹ (I - B X⁻¹ B'A⁻¹) ]
        #      [ N'(I - B'A⁻¹B X⁻¹) B'A⁻¹ ]
        # V⁺' = [ (I - A⁻¹B X⁻¹ B') L'⁻¹
        #         A⁻¹ (I - X⁻¹ B'A⁻¹) BN ]
        assert self._sign > 0
        A, B, C, X = self._A, self._B, self._C, self._X
        if transpose:
            f, g = b[:A.m], b[A.m:]
            BtA_1 = A.solve(B).T
            Lt_1f = A.decorrelate(f, transpose=True)
            x = Lt_1f - X.quad(BtA_1, B.T @ Lt_1f)
            BNg = B @ C.correlate(g)
            y = A.solve(BNg - X.quad(B.T, A.quad(B, BNg)))
            return x + y
        else:
            BtA_1b = A.quad(B, b)
            BtA_1B = A.quad(B)
            x = A.decorrelate(b - X.quad(B.T, BtA_1b))
            y = C.correlate(BtA_1b - X.quad(BtA_1B, BtA_1b), transpose=True)
            return jnp.concatenate([x, y], axis=max(0, b.ndim - 2))
    
    @property
    def n(self):
        return self._A.n
    
    @property
    def m(self):
        return self._A.m + self._C.m

class Woodbury2(DecompAutoDiffBase):
    
    def _matrix(self, A_decomp, B, C_decomp, decompcls, sign=1, **kw):
        assert not self.direct_autodiff
        return A_decomp.matrix() + sign * B @ C_decomp.matrix() @ B.T
            
    def _map_traceable_init_args(self, fun, A_decomp, B, C_decomp, decompcls, sign=1, **kw):
        return (fun(A_decomp), fun(B), fun(C_decomp), decompcls, sign), kw

    def __init__(self, A_decomp, B, C_decomp, decompcls, sign=1, **kw):
        """

        Decompose K = A ± BCB' using Woodbury's formula. K, A > 0, C ≥ 0.
                
        Parameters
        ----------
        A_decomp : Decomposition
            Instantiated decomposition of A.
        B : array
            The B matrix, can be rectangular.
        C_decomp : Decomposition
            Instantiated decomposition of C.
        decompcls : type
            Decomposition class used to decompose I ± L'B'A⁻¹BL, where
            C = LL'.
        sign : {-1, 1}
            The sign in the expression above, default 1.
        **kw :
            Keyword arguments are passed to the initialization of `decompcls`.
        
        """
        assert B.shape == (A_decomp.n, C_decomp.n)
        assert sign in (-1, 1)
        self._A = A_decomp
        self._V = C_decomp.correlate(B.T, transpose=True).T
        I = jnp.eye(C_decomp.m)
        self._X = decompcls(I + sign * A_decomp.quad(self._V), **kw)
        self._sign = sign
    
    # C = LL'
    # (A ± BCB')⁻¹ =
    #   = (A ± B L L' B')⁻¹ =
    #          ~~~
    #          =:V
    #   = (A ± VV')⁻¹ =
    #   = A⁻¹ ∓ A⁻¹ V (I ± V' A⁻¹ V)⁻¹ V' A⁻¹
    #                  ~~~~~~~~~~~~
    #                     =: X
    #   = A⁻¹ ∓ A⁻¹ V X⁻¹ V' A⁻¹
    
    # TODO: when the sign is negative, can I replace X⁻¹ with X+ to get K+?
    
    def solve(self, b):
        A, X, V, s = self._A, self._X, self._V, self._sign
        A_1b = A.solve(b)
        VtA_1b = A.quad(V, b)
        A_1V = A.solve(V)
        return A_1b - s * X.quad(A_1V.T, VtA_1b)
    
    def quad(self, b, c=None):
        A, X, V, s = self._A, self._X, self._V, self._sign
        if c is None:
            bTA_1b = A.quad(b)
            VtA_1b = A.quad(V, b)
            return bTA_1b - s * X.quad(VtA_1b)
        else:
            bTA_1c = A.quad(b, c)
            VtA_1c = A.quad(V, c)
            VtA_1b = A.quad(V, b)
            return bTA_1c - s * X.quad(VtA_1b, VtA_1c)
    
    def logdet(self):
        A, X = self._A, self._X
        # det(A ± VV') = det(A) det(I ± V' A⁻¹ V)
        return A.logdet() + X.logdet()
    
    @property
    def eps(self):
        A, X = self._A, self._X
        return jnp.maximum(A.eps, X.eps)
    
    def correlate(self, b, *, transpose=False):
        # A = NN'
        # A + BCB' = NN' + VV' = [N V] [N V]'
        assert self._sign > 0
        A, V = self._A, self._V
        if transpose:
            return jnp.concatenate([
                A.correlate(b, transpose=True),
                V.T @ b,
            ], axis=max(0, b.ndim - 2))
        else:
            return A.correlate(b[:A.m]) + V @ b[A.m:]
    
    def decorrelate(self, b, *, transpose=False):
        # A = NN'
        # A + BCB' = [N V] [N V]'
        #            ~~~~~
        #             =: Q
        # Q+ = Q' (QQ')⁻¹ =
        #    = Q' (A + BCB')⁻¹ =
        #    = [N'; V'] (A⁻¹ - A⁻¹ V X⁻¹ V' A⁻¹) =
        #    = [ N' N'⁻¹ N⁻¹ - N' N'⁻¹ N⁻¹ V X⁻¹ V' A⁻¹ ] =
        #      [ V' A⁻¹ - V' A⁻¹ V X⁻¹ V' A⁻¹           ]
        #    = [ N⁻¹ - N⁻¹ V X⁻¹ V' A⁻¹    ] =
        #      [ (I - V' A⁻¹ V X⁻¹) V' A⁻¹ ]
        #    = [ (I - N⁻¹ V X⁻¹ V' N'⁻¹) N⁻¹         ]
        #      [ (I - V' N'⁻¹ N⁻¹ V X⁻¹) V' N'⁻¹ N⁻¹ ]
        # Q+' = [ N'⁻¹ - A⁻¹ V X⁻¹ V' N'⁻¹
        #         A⁻¹ V - A⁻¹ V X⁻¹ V' A⁻¹ V ] =
        #     = [ (I - A⁻¹ V X⁻¹ V') N'⁻¹
        #         (I - A⁻¹ V X⁻¹ V') A⁻¹ V ]
        assert self._sign > 0
        A, V, X = self._A, self._V, self._X
        if transpose:
            f, g = b[:A.m], b[A.m:]
            Nt_1f = A.decorrelate(f, transpose=True)
            A_1V = A.solve(V)
            x = Nt_1f - X.quad(A_1V.T, V.T @ Nt_1f)
            Vg = V @ g
            VtA_1Vg = A.quad(V, Vg)
            A_1Vg = A.solve(Vg)
            y = A_1Vg - X.quad(A_1V.T, VtA_1Vg)
            return x + y
        else:
            N_1V = A.decorrelate(V)
            N_1b = A.decorrelate(b)
            VtA_1b = N_1V.T @ N_1b
            x = N_1b - X.quad(N_1V.T, VtA_1b)
            VtA_1V = N_1V.T @ N_1V
            y = VtA_1b - X.quad(VtA_1V, VtA_1b)
            return jnp.concatenate([x, y], axis=max(0, b.ndim - 2))
        
    @property
    def n(self):
        return self._A.n
    
    @property
    def m(self):
        return self._A.m + self._V.shape[1]

class Pinv(DecompAutoDiff): # pragma: no cover
    
    # K3 = K³ + εI = MMt
    # K+ ≈ K K3^-1 K = (M^-1 K)t (M^-1 K)
    # K ≈ (K+ M) (K+ M)t
    
    # K.solve(b) = K3.quad(K, K b)
    # K.quad(b) = K3.quad(K b)
    # K.quad(b, c) = K3.quad(Kb, Kc)
    # K.logdet = 1/3 K3.logdet
    # K.correlate(b) = K3.quad(K, K K3.correlate(b))
    # K.correlate(b, T) = K3.correlate(K3.quad(K, K b), T)
    # K.decorrelate(b) = K3.decorrelate(K b)
    # K.decorrelate(b, T) = K @ K3.decorrelate(b, T)
    
    # TODO pass directly the instantiated decomposition of K^3 along with K?
    
    def __init__(self, K, decompcls, **kw):
        """
        
        Approximate a pseudoinverse using an inversion.
        
        Parameters
        ----------
        K :
            The matrix to be decomposed.
        decompcls : type
            Class used to decompose K^3, need not support ill-conditioned
            matrices.
        
        """
    
        self._K3 = decompcls(K @ K @ K, **kw)
        self._K = K
    
    def solve(self, b):
        K, K3 = self._K, self._K3
        return K3.quad(K, K @ b)
    
    def quad(self, b, c=None):
        K, K3 = self._K, self._K3
        if c is None:
            return K3.quad(K @ b)
        elif c.dtype == object:
            return K3.quad(K @ b, numpy.asarray(K) @ c)
        else:
            return K3.quad(K @ b, K @ c)
    
    def logdet(self):
        return self._K3.logdet() / 3
        # TODO this is not a pseudodeterminant
    
    def correlate(self, b, *, transpose=False):
        K, K3 = self._K, self._K3
        if transpose:
            return K3.correlate(K3.quad(K, K @ b), transpose=True)
        else:
            return K3.quad(K, K @ K3.correlate(b))
    
    def decorrelate(self, b, *, transpose=False):
        K, K3 = self._K, self._K3
        if transpose:
            return K @ K3.decorrelate(b, transpose=True)
        else:
            return K3.decorrelate(K @ b)
    
    @property
    def m(self):
        return self._K3.m
    
    @property
    def eps(self):
        return jnp.cbrt(self._K3.eps)

class Pinv2(DecompAutoDiff): # pragma: no cover
    
    # K3 = K³ + εI = MM'
    # K⁺ = ∑ n=1^∞ εⁿ⁻¹ K K3⁻ⁿ K =
    #    = ∑ n=1^∞ εⁿ⁻¹ (M⁻ⁿ K)' (M⁻ⁿ K)
    #    = A⁺' A⁺
    # A⁺ = [M⁻¹ K; √ε M⁻² K; ...]
    # A = (A⁺' A⁺)⁺ A⁺' = K A⁺' = K² [M'⁻¹, √ε M'⁻², ...]
    # K = AA'
        
    # TODO pass directly the instantiated decomposition of K^3 along with K?
    
    def __init__(self, K, decompcls, *, N=1, epsrel='auto', epsabs=0, **kw):
        """
        
        Approximate K⁺ with the truncated series
        
            K⁺ ≈ ∑ n=1^N εⁿ⁻¹ K (K³ + εI)⁻ⁿ K.
        
        Parameters
        ----------
        K : (n, n) array
            The matrix to be decomposed.
        decompcls : callable
            Class used to decompose K³ + εI, need not support ill-conditioned
            matrices. It should not add its own diagonal regularization.
        N : int
            Truncation order of the series.
        eps : float
            Positive regularization used to make K³ invertible, relative to
            an upper bound on the spectral radius of K³.
        **kw :
            Keyword arguments are passed to `decompcls`.
        
        """
        
        assert N >= 1
        K = jnp.asarray(K)
        K3 = K @ K @ K
        eps = self._parseeps(K3, epsrel, epsabs)
        K3 = K3.at[jnp.diag_indices_from(K3)].add(eps)
        self._K3 = decompcls(K3, **kw)
        self._K = K
        self._N = N
        self._eps = eps
    
    # K.solve(b) =
    # ∑ n=1^N ε^(n-1) K3.quad(K3.solve^n/2-1(K), K3.solve^n/2(Kb))       n even
    #                 K3.quad(K3.solve^(n-1)/2(K), K3.solve^(n-1)/2(Kb)) n odd
    # n  p1 p2
    # 1  0  0
    # 2  0  1
    # 3  1  1
    # 4  1  2
    # 5  2  2
    # etc.
    def solve(self, b):
        eps, K, K3, N = self._eps, self._K, self._K3, self._N
        K3solveK = K
        K3solveKb = K @ b
        coef = 1
        acc = K3.quad(K3solveK, K3solveKb)
        for n in range(2, N + 1):
            coef *= eps
            if n % 2:
                K3solveK = K3.solve(K3solveK)
            else:
                K3solveKb = K3.solve(K3solveKb)
            acc += coef * K3.quad(K3solveK, K3solveKb)
        return acc
    
    # K.quad(b) =
    # ∑ n=1^N ε^(n-1) K3.quad(K3.solve^n/2-1(Kb), K3.solve^n/2(Kb))   n even
    #                 K3.quad(K3.solve^(n-1)/2(Kb))                   n odd
    # K.quad(b, c) =
    # ∑ n=1^N ε^(n-1) K3.quad(K3.solve^n/2-1(Kb), K3.solve^n/2(Kc))       n even
    #                 K3.quad(K3.solve^(n-1)/2(Kb), K3.solve^(n-1)/2(Kc)) n odd
    # K.quad(b, c_obj) =
    # ∑ n=1^N ε^(n-1) K3.quad(K3.solve^(n-1)(Kb), Kc)
    #
    # n  p1 p2
    # 1  0  0
    # 2  0  1
    # 3  1  1
    # 4  1  2
    # 5  2  2
    # etc.
    def quad(self, b, c=None):
        eps, K, K3, N = self._eps, self._K, self._K3, self._N
        coef = 1
        
        if c is None:
            K3solveKb = K @ b
            acc = K3.quad(K3solveKb)
            for n in range(2, N + 1):
                coef *= eps
                if n % 2:
                    acc += coef * K3.quad(K3solveKb)
                else:
                    K3solveKb_next = K3.solve(K3solveKb)
                    acc += coef * K3.quad(K3solveKb, K3solveKb_next)
                    K3solveKb = K3solveKb_next

        elif c.dtype == object:
            K3solveKb = K @ b
            Kc = numpy.asarray(K) @ c
            acc = K3.quad(K3solveKb, Kc)
            eps = numpy.asarray(eps)
            for n in range(2, N + 1):
                coef *= eps
                K3solveKb = K3.solve(K3solveKb)
                acc += coef * K3.quad(K3solveKb, Kc)

        else:
            K3solveKc = K @ c
            K3solveKb = K @ b
            acc = K3.quad(K3solveKb, K3solveKc)
            for n in range(2, N + 1):
                coef *= eps
                if n % 2:
                    K3solveKb = K3.solve(K3solveKb)
                else:
                    K3solveKc = K3.solve(K3solveKc)
                acc += coef * K3.quad(K3solveKb, K3solveKc)

        return acc
    
    # K.logdet = 1/3 K3.logdet
    def logdet(self):
        return self._K3.logdet() / 3
        # TODO this is not a pseudodeterminant
    
    # K.correlate(b) =
    # K^2 ∑ n=1^N ε^(n-1)/2 K3.decorrelate^n(b_n, T)
    
    # K.correlate(b, T) = [
    #   ε^(n-1)/2 K3.decorrelate^n(K^2 b)
    #   for n in 1 ... N
    # ]
    def correlate(self, b, *, transpose=False):
        seps, K2, K3, N = jnp.sqrt(self._eps), self._K @ self._K, self._K3, self._N
        coef = 1
        
        if transpose:
            K3decK2b = K2 @ b
            parts = []
            for n in range(1, N + 1):
                K3decK2b = K3.decorrelate(K3decK2b)
                parts.append(coef * K3decK2b)
                coef *= seps
            return jnp.concatenate(parts, axis=max(0, b.ndim - 2))
        
        else:
            acc = 0
            for n in range(1, N + 1):
                K3decbn = b[K3.m * (n - 1):K3.m * n]
                for k in range(n):
                    K3decbn = K3.decorrelate(K3decbn, transpose=True)
                acc += coef * K3decbn
                coef *= seps
            return K2 @ acc
    
    # K.decorrelate(b) = [
    #   ε^(n-1)/2 K3.decorrelate^n(K b)
    #   for n in 1 ... N
    # ]
    
    # K.decorrelate(b, T) =
    # K ∑ n=1^N ε^(n-1)/2 K3.decorrelate^n(b_n, T)
    def decorrelate(self, b, *, transpose=False):
        seps, K, K3, N = jnp.sqrt(self._eps), self._K, self._K3, self._N
        coef = 1
        
        if transpose:
            acc = 0
            for n in range(1, N + 1):
                K3decbn = b[K3.m * (n - 1):K3.m * n]
                for k in range(n):
                    K3decbn = K3.decorrelate(K3decbn, transpose=True)
                acc += coef * K3decbn
                coef *= seps
            return K @ acc

        else:
            K3decKb = K @ b
            parts = []
            for n in range(1, N + 1):
                K3decKb = K3.decorrelate(K3decKb)
                parts.append(coef * K3decKb)
                coef *= seps
            return jnp.concatenate(parts, axis=max(0, b.ndim - 2))
    
    @property
    def m(self):
        return self._K3.m * self._N

    @property
    def eps(self):
        return jnp.cbrt(self._K3.eps)
