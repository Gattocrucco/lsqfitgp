import abc

from ._imports import numpy as np
from ._imports import linalg
from ._imports import autograd
from ._imports import sparse

__doc__ = """

Decompositions of positive definite matrices. A decomposition object is
initialized with a matrix and then can solve linear systems for that matrix.
These classes never check for infs/nans in the matrices.

Classes
-------
DecompMeta :
    Metaclass that adds autograd support.
Decomposition :
    Abstract base class.
Diag :
    Diagonalization.
EigCutFullRank :
    Diagonalization rounding up small eigenvalues.
EigCutLowRank :
    Diagonalization removing small eigenvalues.
ReduceRank :
    Partial diagonalization with higher eigenvalues only.
Chol :
    Cholesky decomposition.
CholMaxEig :
    Cholesky regularized using the maximum eigenvalue.
CholGersh :
    Cholesky regularized using an estimate of the maximum eigenvalue.
BlockDecomp :
    Decompose a block matrix.

"""

# TODO check solve and quad work with >2D b. Probably I need to reshape
# b to 2D, do the calculation and then reshape back. When it works, write
# clearly the contraction convention in the docstrings. (Low priority, I never
# use this in the GP code.)

# TODO add an optional argument c to quad to compute b.T @ inv(K) @ c.

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

class DecompMeta(abc.ABCMeta):
    """
    Metaclass for adding autograd support to subclasses of Decomposition.
    """
    
    # TODO add jvps for solve and quad for forward mode
    
    def __init__(cls, *args):
        
        # For __init__ I can't use an _autograd flag like below to avoid double
        # wrapping because the wrapper is called as super().__init__ in
        # subclasses, so I assign self._K *after* calling old__init__.
        old__init__ = cls.__init__
        def __init__(self, K, **kw):
            old__init__(self, noautograd(K), **kw)
            self._K = K
        cls.__init__ = __init__
        
        oldsolve = cls.solve
        if not hasattr(oldsolve, '_autograd'):
            @autograd.extend.primitive
            def solve_autograd(self, K, b):
                return oldsolve(self, b)
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
            def solve(self, b):
                return solve_autograd(self, self._K, b)
            # solve_autograd is used by logdet_vjp, so I store it here in case
            # logdet but not solve need wrapping in a subclass
            solve._autograd = solve_autograd
            cls.solve = solve
        
        # TODO write vjp/jvp optimized for quad instead of relying on solve
        oldquad = cls.quad
        if not hasattr(oldquad, '_autograd'):
            def quad(self, b):
                if isinstance(self._K, np.numpy_boxes.ArrayBox):
                    return b.T @ self.solve(b)
                else:
                    return oldquad(self, b)
            quad._autograd = True
            cls.quad = quad
        
        oldlogdet = cls.logdet
        if not hasattr(oldlogdet, '_autograd'):
            @autograd.extend.primitive
            def logdet_autograd(self, K):
                return oldlogdet(self)
            def logdet_vjp(ans, self, K):
                assert ans.shape == ()
                assert K.shape[0] == K.shape[1]
                def vjp(g):
                    invK = self.solve._autograd(self, K, np.eye(len(K)))
                    return g[..., None, None] * invK
                return vjp
            autograd.extend.defvjp(
                logdet_autograd,
                logdet_vjp,
                argnums=[1]
            )
            # TODO the vjp is bad because it makes a matrix inversion. I would
            # like to do forward propagation just for the logdet plus the
            # preceding step, but I don't think autograd supports this.
            def logdet_jvp(ans, self, K):
                assert ans.shape == ()
                assert K.shape[0] == K.shape[1]
                def jvp(g):
                    assert g.shape[:2] == K.shape
                    return np.trace(self.solve._autograd(self, K, g))
                return jvp
            autograd.extend.defjvp(
                logdet_autograd,
                logdet_jvp,
                argnums=[1]
            )
            def logdet(self):
                return logdet_autograd(self, self._K)
            logdet._autograd = True
            cls.logdet = logdet

if autograd is None:
    DecompMeta = abc.ABCMeta

class Decomposition(metaclass=DecompMeta):
    """
    
    Abstract base class for positive definite symmetric matrices decomposition.
    
    Methods
    -------
    solve
    usolve
    quad
    logdet
    
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
        Solve the linear system K @ x = b.
        """
        pass
    
    @abc.abstractmethod
    def usolve(self, ub):
        """
        Solve the linear system K @ x = b where b is possibly an array of
        gvars.
        """
        pass
    
    def quad(self, b):
        """
        Compute the quadratic form b.T @ inv(K) @ b.
        """
        return b.T @ self.solve(b)
    
    @abc.abstractmethod
    def logdet(self):
        """
        Compute log(det(K)).
        """
        pass

class Diag(Decomposition):
    """
    Diagonalization.
    """
    
    def __init__(self, K):
        self._w, self._V = linalg.eigh(K, check_finite=False)
    
    def solve(self, b):
        return (self._V / self._w) @ (self._V.T @ b)
    
    usolve = solve
    
    def quad(self, b):
        VTb = self._V.T @ b
        return (VTb.T / self._w) @ VTb
    
    def logdet(self):
        return np.sum(np.log(self._w))
    
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
    
    def __init__(self, K, eps=None, **kw):
        super().__init__(K, **kw)
        eps = self._eps(eps)
        self._w[self._w < eps] = eps
            
class EigCutLowRank(Diag):
    """
    Diagonalization. Eigenvalues below `eps` are removed, where `eps` is
    relative to the largest eigenvalue.
    """
    
    def __init__(self, K, eps=None, **kw):
        super().__init__(K, **kw)
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

def solve_triangular(a, b, lower=False):
    """
    Pure python implementation of scipy.linalg.solve_triangular for when
    a or b are object arrays.
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

def grad_chol(L):
    """
    Inverse of the Jacobian of the cholesky factor respect to the decomposed
    matrix, reshaped as a 2D matrix. It should actually work for any
    decomposition of the type A = L @ L.T, whatever is L. (Not tested.)
    """
    n = len(L)
    I = np.eye(n)
    s1 = I[:, None, :, None] * L[None, :, None, :]
    s2 = I[None, :, :, None] * L[:, None, None, :]
    return (s1 + s2).reshape(2 * (n**2,))
        
class Chol(Decomposition):
    """
    Cholesky decomposition.
    """
    
    def __init__(self, K):
        self._L = linalg.cholesky(K, lower=True, check_finite=False)
    
    def solve(self, b):
        invLb = linalg.solve_triangular(self._L, b, lower=True)
        return linalg.solve_triangular(self._L.T, invLb, lower=False)
    
    def usolve(self, b):
        invLb = solve_triangular(self._L, b, lower=True)
        return solve_triangular(self._L.T, invLb, lower=False)
    
    def quad(self, b):
        invLb = linalg.solve_triangular(self._L, b, lower=True)
        return invLb.T @ invLb
    
    def logdet(self):
        return 2 * np.sum(np.log(np.diag(self._L)))
    
    def _eps(self, eps, K, maxeigv):
        if eps is None:
            eps = len(K) * np.finfo(asinexact(K.dtype)).eps
        assert np.isscalar(eps) and 0 <= eps < 1
        return eps * maxeigv

class CholMaxEig(Chol):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff
    by adding to the diagonal a small number relative to the maximum eigenvalue.
    `eps` multiplies this number.
    """
    
    def __init__(self, K, eps=None, **kw):
        # TODO Should I precondition K *before* computing and adding the
        # epsilon on the diagonal? In case, there's a function in scipy for
        # generic rescaling preconditioning.
        w = sparse.linalg.eigsh(K, k=1, which='LM', return_eigenvectors=False)
        eps = self._eps(eps, K, w[0])
        super().__init__(K + np.diag(np.full(len(K), eps)), **kw)


class CholGersh(Chol):
    """
    Cholesky decomposition. The matrix is corrected for numerical roundoff
    by adding to the diagonal a small number relative to the maximum eigenvalue.
    `eps` multiplies this number. The maximum eigenvalue is estimated
    with the Gershgorin theorem.
    """
    
    def __init__(self, K, eps=None, **kw):
        maxeigv = _gershgorin_eigval_bound(K)
        eps = self._eps(eps, K, maxeigv)
        super().__init__(K + np.diag(np.full(len(K), eps)), **kw)

def _gershgorin_eigval_bound(K):
    """
    Upper bound on the largest magnitude eigenvalue of the matrix.
    """
    return np.max(np.sum(np.abs(K), axis=1))

class BlockDecomp:
    """
    Decomposition of a 2x2 symmetric block matrix using decompositions of the
    diagonal blocks.
    
    Reference: Gaussian Processes for Machine Learning, A.3, p. 201.
    """
    
    # This is not a subclass of Decomposition because the __init__
    # signature is different.
    
    # This class can be used only starting from a seed block and adding
    # other blocks one at a time. Would a divide et impera approach be useful
    # for my case?
    
    def __init__(self, P_decomp, S, Q, S_decomp_class):
        """
        The matrix to be decomposed is
        
        A = [[P,   Q]
             [Q.T, S]]
        
        Parameters
        ----------
        P_decomp : Decomposition
            An instantiated decomposition of P.
        S, Q : matrices
            The other blocks.
        S_decomp_class : DecompMeta
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
        y = tildeS.solve(g - Q.T @ invP.solve(f)) # TODO invP.quad(Q, f)
        x = invP.solve(f - Q @ y)
        return np.concatenate([x, y])
    
    def usolve(self, b):
        invP = self._invP
        tildeS = self._tildeS
        Q = self._Q
        f = b[:len(Q)]
        g = b[len(Q):]
        y = tildeS.usolve(g - Q.T @ invP.usolve(f))
        x = invP.usolve(f - Q @ y)
        return np.concatenate([x, y])
    
    def quad(self, b):
        invP = self._invP
        tildeS = self._tildeS
        Q = self._Q
        f = b[:len(Q)]
        g = b[len(Q):]
        QTinvPf = Q.T @ invP.solve(f) # TODO invP.quad(Q, f)
        fTinvPQtildeSg = QTinvPf.T @ tildeS.solve(g) # tildeS.quad(QTinvPf, g)
        return invP.quad(f) + tildeS.quad(QTinvPf) - fTinvPQtildeSg - fTinvPQtildeSg.T + tildeS.quad(g)

    def logdet(self):
        return self._invP.logdet() + self._tildeS.logdet()
