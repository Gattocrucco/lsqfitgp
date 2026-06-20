# lsqfitgp/_linalg/_decomp.py
#
# Copyright (c) 2023, 2024, Giacomo Petrillo
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

Copy-pasted from the notes:

2023-02-14
==========

My current decomposition system is a mess. I can't take reverse gradients.
I can't straightforwardly implement optimized algorithms that compute together
likelihood, gradient, and fisher. Jax patterns break down unpredictably. I
have to redesign it from scratch.

Guidelines and requirements:

  - Sufficient modularity to implement composite decompositions (Woodbury,
    Block)

  - Does not mess up jax in any way

  - Caches decompositions

  - Favors optimizing together the likelihood and its derivatives

Operations (in the following I indicate with lowercase inputs which are
typically vectors or tall matrices, and uppercase inputs which are typically
large matrices, since optimization requires taking it into account):

    pinv_bilinear(A, r) => A'K⁺r (for the posterior mean)
    pinv_bilinear_robj(A, r) same but r can be gvars
    ginv_quad(A) => A'K⁻A (for the posterior covariance)
        I want the pseudoinverse for the mean because the data may not be
        in the span and I want to project it orthogonally, while for the
        covariance I expect A and K to come from a pos def matrix so they are
        coherent
    ginv_diagquad(A) => diag(A'K⁻A) (for the posterior variance)
    minus_log_normal_density(
        r: 1d array,      # the residuals (data - prior mean)
        dr_vjp: callable, # x -> x_i ∂r_i/∂p_j,   gradrev and fishvec
        dK_vjp: callable, # x -> x_ij ∂K_ij/∂p_k, gradrev and fishvec
        dr_jvp: callable, # x -> ∂r_i/∂r_j x_j,  fishvec
        dK_jvp: callable, # x -> ∂K_ij/∂p_k x_k, fishvec
        dr: 2d array,     # ∂r_i/∂p_j,  gradfwd and fisher
        dK: 3d array,     # ∂K_ij/∂p_k, gradfwd and fisher
        vec: 1d array,    # input vector of fishvec, same size as params
        value: bool,
        gradrev: bool,
        gradfwd: bool,
        fisher: bool,
        fishvec: bool,
    )
        This computes on request
            value: 1/2 tr(KK⁺) log 2π
                 + 1/2 tr(I-KK⁺) log 2π
                 + 1/2 log pdet K
                 + 1/2 tr(I-KK⁺) log ε
                 + 1/2 r'(K⁺+(I-KK⁺)/ε)r
            gradrev,
            gradfwd: 1/2 tr(K⁺dK)
                    + r'(K⁺+(I-KK⁺)/ε) dr
                    - 1/2 r'(K⁺+2(I-KK⁺)/ε)dKK⁺r
            fisher: 1/2 tr(K⁺dK(K⁺+2(I-KK⁺)/ε)d'K)
                  - 2 tr(K⁺dK(I-KK⁺)d'KK⁺)
                  + dr'(K⁺+(I-KK⁺)/ε)d'r
            fishvec: fisher matrix times vec
        There should be options for omitting the pieces with ε. I also need a
        way to make densities with different values of ε comparable with each
        other (may not be possible, if it is, it probably requires a history of
        ranks and ε). gradfwd/rev form K⁺ explicitly to compute tr(K⁺dK) for
        efficiency.
    correlate(x)
        Zx where K = ZZ'.
    back_correlate(X):
        Z'X, this is used by Sandwich and Woodbury.

Since I also want to compute the Student density, I could split
minus_log_normal_density's return value into logdet and quad. The gradient
splits nicely between the two terms, but I have to redo the calculation of the
Fisher matrix for the Student distribution. Alternatively, I could use the the
Normal Fisher. => See Lange et al. (1989, app. B). => I think I can split the
gradient and Fisher matrix too.

2023-03-07
==========

To compute a Fisher-vector product when there are many parameters, do

    tr(K+ dK K+ dK) v =
    = K_vjp(K+ K_jvp(v) K+)

"""




import abc
import functools

import numpy
import jax
from jax import numpy as jnp
from jax.scipy import linalg as jlinalg
from jax import lax

from .. import _jaxext
from . import _pytree

class Decomposition(_pytree.AutoPyTree, abc.ABC):
    """
    Abstract base class for decompositions of positive semidefinite matrices.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kw):
        """ Decompose the input matrix """
        pass

    @abc.abstractmethod
    def matrix(self):
        """ The input matrix """
        pass

    @abc.abstractmethod
    def ginv_linear(self, X):
        """ Compute K⁻X """
        pass

    @abc.abstractmethod
    def pinv_bilinear(self, A, r):
        """Compute A'K⁺r."""
        pass

    @abc.abstractmethod
    def pinv_bilinear_robj(self, A, r):
        """Compute A'K⁺r, where r can be an array of objects."""
        pass

    @abc.abstractmethod
    def ginv_quad(self, A):
        """Compute A'K⁻A."""
        pass

    @abc.abstractmethod
    def ginv_diagquad(self, A):
        """Compute diag(A'K⁻A)."""
        pass

    @abc.abstractmethod
    def correlate(self, x):
        """ Compute Zx where K = ZZ' """
        pass

    @abc.abstractmethod
    def back_correlate(self, X):
        """ Compute Z'X """
        pass

    @abc.abstractmethod
    def pinv_correlate(self, x):
        """ Compute Z⁺x """
        pass

    @abc.abstractmethod
    def minus_log_normal_density(self,
        r,
        *,
        dr_vjp=None,
        dK_vjp=None,
        dr_jvp_vec=None,
        dK_jvp_vec=None,
        dr=None,
        dK=None,
        value=False,
        gradrev=False,
        gradfwd=False,
        fisher=False,
        fishvec=False,
        ):
        """
        Compute minus log a Normal density and its derivatives, with covariance
        matrix K.

        If an input derivative is not specified, it is assumed to be zero.

        Parameters
        ----------
        r: 1d array
            The residuals (value - mean)
        dr_vjp: callable
            x -> x_i ∂r_i/∂p_j, for gradrev and fishvec
        dK_vjp: callable
            x -> x_ij ∂K_ij/∂p_k, for gradrev and fishvec
        dr_jvp_vec: 1d array
            ∂r_i/∂r_j vec_j, for fishvec
        dK_jvp_vec: 2d array
            ∂K_ij/∂p_k vec_k, for fishvec
        dr: 2d array
            ∂r_i/∂p_j for gradfwd and fisher
        dK: 3d array
            ∂K_ij/∂p_k, for gradfwd and fisher
        value: bool
        gradrev: bool
        gradfwd: bool
        fisher: bool
        fishvec: bool
            These parameters indicate which of the return values to compute.
            Default all False.

        Returns
        -------
        value: 1/2 tr(KK⁺) log 2π
             + 1/2 tr(I-KK⁺) log 2π
             + 1/2 log pdet K
             + 1/2 tr(I-KK⁺) log ε
             + 1/2 r'(K⁺+(I-KK⁺)/ε)r
        gradrev,
        gradfwd: 1/2 tr(K⁺dK)
                + r'(K⁺+(I-KK⁺)/ε) dr
                - 1/2 r'(K⁺+2(I-KK⁺)/ε)dKK⁺r
        fisher: 1/2 tr(K⁺dK(K⁺+2(I-KK⁺)/ε)d'K)
              - 2 tr(K⁺dK(I-KK⁺)d'KK⁺)
              + dr'(K⁺+(I-KK⁺)/ε)d'r
        fishvec: fisher matrix @ vec
        """
        pass

    def _parseeps(self, K, epsrel, epsabs, maxeigv=None):
        """ Determine eps from input arguments """
        machine_eps = jnp.finfo(_jaxext.float_type(K)).eps
        if epsrel == 'auto':
            epsrel = len(K) * machine_eps
        if epsabs == 'auto':
            epsabs = machine_eps
        if maxeigv is None:
            maxeigv = eigval_bound(K)
        self._eps = epsrel * maxeigv + epsabs
        return self._eps

    @property
    def eps(self):
        """
        The threshold below which eigenvalues are too small to be determined.
        """
        return self._eps

    @property
    @abc.abstractmethod
    def n(self):
        """ Number of rows/columns of the matrix """
        pass

    @property
    @abc.abstractmethod
    def m(self):
        """ Number of columns of Z """
        pass

    def ginv(self):
        """ Compute K⁻ """
        return self.ginv_quad(jnp.eye(self.n))

def solve_triangular_python(a, b, *, lower=False):
    """
    Pure python implementation of scipy.linalg.solve_triangular for when
    a or b are object arrays.
    """
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

def solve_triangular_batched(a, b, *, lower=False):
    """ Version of jax.scipy.linalg.solve_triangular that batches matmul-like """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    vec = b.ndim < 2
    if vec:
        b = b[:, None]

    batch_shape = jnp.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a_shape = batch_shape + a.shape[-2:]
    b_shape = batch_shape + b.shape[-2:]
    result = lax.linalg.triangular_solve(
        jnp.broadcast_to(a, a_shape), jnp.broadcast_to(b, b_shape),
        left_side=True, lower=lower,
    )
    assert result.shape == b_shape

    if vec:
        result = result.squeeze(-1)
    return result

def solve_batched(a, b, **kw):
    """ Version of jax.scipy.linalg.solve that batches matmul-like """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    vec = b.ndim < 2
    if vec:
        b = b[:, None]

    @functools.partial(jnp.vectorize, signature='(i,j),(j,k)->(i,k)')
    def solve_batched(a, b):
        return jlinalg.solve(a, b, **kw)
    result = solve_batched(a, b)

    if vec:
        result = result.squeeze(-1)
    return result

def eigval_bound(K):
    """
    Upper bound on the largest magnitude eigenvalue of the matrix, from
    Gershgorin's theorem.
    """
    return jnp.max(jnp.sum(jnp.abs(K), axis=1))

def diag_scale_pow2(K):
    """
    Compute a vector s of powers of 2 such that diag(K / outer(s, s)) ~ 1.
    """
    d = jnp.diag(K)
    return jnp.where(d, jnp.exp2(jnp.rint(0.5 * jnp.log2(d))), 1)

    # Golub and Van Loan (2013) say this is not a totally general heuristic

def transpose(x):
    """ swap the last two axes of array x, corresponds to matrix tranposition
    with the broadcasting convention of matmul """
    if x.ndim < 2:
        return x
    elif isinstance(x, jnp.ndarray):
        return jnp.swapaxes(x, -2, -1)
    else:
        # need to support numpy because this function is used with gvars
        return numpy.swapaxes(x, -2, -1)

class Chol(Decomposition):
    """Cholesky decomposition. The matrix is regularized adding a small multiple
    of the identity."""

    def __init__(self, K, *, epsrel='auto', epsabs=0):
        # K <- K + Iε
        # K = LL'
        self._K = K
        s = diag_scale_pow2(K)
        K = K / s / s[:, None]
        eps = self._parseeps(K, epsrel, epsabs)
        K = K.at[jnp.diag_indices_from(K)].add(eps)
        L = jlinalg.cholesky(K, lower=True)
        with _jaxext.skipifabstract():
            if not jnp.all(jnp.isfinite(L)):
                raise numpy.linalg.LinAlgError('cholesky decomposition not finite, probably matrix not pos def numerically')
        self._L = L * s[:, None]
        self._eps = eps * jnp.min(s * s)

    def matrix(self):
        return self._K

    def ginv_linear(self, X):
        # = K⁻¹X
        # K⁻¹ = L'⁻¹L⁻¹
        # K⁻¹X = L'⁻¹(L⁻¹X)
        invLX = jlinalg.solve_triangular(self._L, X, lower=True)
        return jlinalg.solve_triangular(self._L.T, invLX, lower=False)

    def pinv_bilinear(self, A, r):
        # = A'K⁻¹r = A'L'⁻¹L⁻¹r = (L⁻¹A)'(L⁻¹r)
        invLr = jlinalg.solve_triangular(self._L, r, lower=True)
        invLA = jlinalg.solve_triangular(self._L, A, lower=True)
        return invLA.T @ invLr

    def pinv_bilinear_robj(self, A, r):
        # = A'K⁻¹r
        invLr = solve_triangular_python(self._L, r, lower=True)
        invLA = jlinalg.solve_triangular(self._L, A, lower=True)
        return numpy.asarray(invLA).T @ invLr

    def ginv_quad(self, A):
        # = A'K⁻¹A = A'K⁻¹A = A'L'⁻¹L⁻¹A = (L⁻¹A)'(L⁻¹A)
        invLA = jlinalg.solve_triangular(self._L, A, lower=True)
        return invLA.T @ invLA

    def ginv_diagquad(self, A):
        # = diag(A'K⁻¹A)
        # X = L⁻¹A
        # diag(A'K⁻¹A)_i = diag(X'X)_i = ∑_j X'_ij X_ji = ∑_j X_ji X_ji
        invLA = jlinalg.solve_triangular(self._L, A, lower=True)
        return jnp.einsum('ji,ji->i', invLA, invLA)

    def correlate(self, x):
        # = Lx
        return self._L @ x

    def back_correlate(self, X):
        # = L'X
        return self._L.T @ X

    def pinv_correlate(self, x):
        # = L⁻¹x
        return jlinalg.solve_triangular(self._L, x, lower=True)

    def minus_log_normal_density(self,
        r,               # 1d array, the residuals (data - prior mean)
        *,
        dr_vjp=None,     # callable, x -> x_i ∂r_i/∂p_j,   gradrev and fishvec
        dK_vjp=None,     # callable, x -> x_ij ∂K_ij/∂p_k, gradrev and fishvec
        dr_jvp_vec=None, # 1d array, ∂r_i/∂r_j v_j,  fishvec
        dK_jvp_vec=None, # 2d array, ∂K_ij/∂p_k v_k, fishvec
        dr=None,         # 2d array, ∂r_i/∂p_j,  gradfwd and fisher
        dK=None,         # 3d array, ∂K_ij/∂p_k, gradfwd and fisher
        value=False,
        gradrev=False,
        gradfwd=False,
        fisher=False,
        fishvec=False,
    ):

        L = self._L

        out = {}

        # compute shared factors
        grad = (
            (gradrev and (dK_vjp is not None or dr_vjp is not None))
            or (gradfwd and (dK is not None or dr is not None))
        )
        if value or grad:
            invLr = jlinalg.solve_triangular(L, r, lower=True)
        if grad:
            invKr = jlinalg.solve_triangular(L.T, invLr, lower=False)
        if (gradrev and dK_vjp is not None) or (gradfwd and dK is not None):
            invL = jlinalg.solve_triangular(L, jnp.eye(len(L)), lower=True)
            invK = invL.T @ invL

        if value:
            # = 1/2 n log 2π
            #   + 1/2 log det K
            #   + 1/2 r'K⁻¹r
            # K = LL'
            # K⁻¹ = L'⁻¹L⁻¹
            # det K = (det L)² =
            #       = (∏_i L_ii)²
            # r'K⁻¹r = r'L'⁻¹L⁻¹r =
            #        = (L⁻¹r)'(L⁻¹r)
            out['value'] = 1/2 * (
                len(L) * jnp.log(2 * jnp.pi) +
                2 * jnp.sum(jnp.log(jnp.diag(L))) +
                invLr @ invLr
            )
        else:
            out['value'] = None

        if gradrev:
            # = 1/2 tr(K⁻¹dK)
            #   + r'K⁻¹dr
            #   - 1/2 r'K⁻¹dKK⁻¹r
            # tr(K⁻¹dK) = K⁻¹_ij dK_ji =
            #           = K⁻¹_ij dK_ij =
            #           = dK_vjp(K⁻¹)
            # r'K⁻¹dr = r_i K⁻¹_ij dr_j =
            #         = (K⁻¹r)_j dr_j =
            #         = dr_vjp(K⁻¹r)
            # r'K⁻¹dKK⁻¹r = r_i K⁻¹_ij dK_jl K⁻¹_lm r_m =
            #             = (K⁻¹r)_j dK_jl (K⁻¹r)_l =
            #             = dK_vjp((K⁻¹r) ⊗ (K⁻¹r))
            out['gradrev'] = 0
            if dK_vjp is not None:
                tr_invK_dK = dK_vjp(invK)
                r_invK_dK_invK_r = dK_vjp(jnp.outer(invKr, invKr))
                out['gradrev'] += 1/2 * (tr_invK_dK - r_invK_dK_invK_r)
            if dr_vjp is not None:
                r_invK_dr = dr_vjp(invKr)
                out['gradrev'] += r_invK_dr
        else:
            out['gradrev'] = None

        if gradfwd:
            # = 1/2 tr(K⁻¹dK)
            #   + r'K⁻¹dr
            #   - 1/2 r'K⁻¹dKK⁻¹r
            # tr(K⁻¹dK)_k = K⁻¹_ij dK_ijk
            # r'K⁻¹dr = (K⁻¹r)'dr
            # (r'K⁻¹dKK⁻¹r)_k = r_i K⁻¹_ij dK_jlk K⁻¹_lm r_m =
            #                 = (K⁻¹r)_j dK_jlk (K⁻¹r)_l
            out['gradfwd'] = 0
            if dK is not None:
                tr_invK_dK = jnp.einsum('ij,ijk->k', invK, dK)
                r_invK_dK_invK_r = jnp.einsum('i,ijk,j->k', invKr, dK, invKr)
                out['gradfwd'] += 1/2 * (tr_invK_dK - r_invK_dK_invK_r)
            if dr is not None:
                r_invK_dr = invKr @ dr
                out['gradfwd'] += r_invK_dr
        else:
            out['gradfwd'] = None

        if fisher:
            # = 1/2 tr(K⁻¹dKK⁻¹d'K)
            #   + dr'K⁻¹d'r
            # tr(K⁻¹dKK⁻¹d'K)_ij = tr(L'⁻¹L⁻¹dKL'⁻¹L⁻¹d'K)_ij =
            #                    = tr(L⁻¹dKL'⁻¹L⁻¹d'KL'⁻¹)_ij =
            #                    = (L⁻¹dKL'⁻¹)_kli (L⁻¹dKL'⁻¹)_klj
            # (L⁻¹dKL'⁻¹)_ijk = L⁻¹_il dK_lmk L'⁻¹_mj =
            #                 = L⁻¹_il L⁻¹_jm dK_lmk
            # (dr'K⁻¹d'r)_kq = dr'_k L'⁻¹L⁻¹dr_q =
            #                = (L⁻¹dr_k)_i (L⁻¹dr_q)_i
            out['fisher'] = 0
            if dK is not None:
                invL_dK = solve_triangular_batched(L,
                    jnp.moveaxis(dK, 2, 0),
                    lower=True) # kim: L⁻¹_il dK_lmk
                invL_dK_invL = solve_triangular_batched(L,
                    jnp.swapaxes(invL_dK, 1, 2),
                    lower=True) # kji: L⁻¹_jm (L⁻¹_il dK_lmk)
                tr_invK_dK_invK_dK = jnp.einsum('kij,qij->kq', invL_dK_invL, invL_dK_invL)
                out['fisher'] += 1/2 * tr_invK_dK_invK_dK
            if dr is not None:
                invLdr = jlinalg.solve_triangular(L, dr, lower=True)
                dr_invK_dr = invLdr.T @ invLdr
                out['fisher'] += dr_invK_dr
        else:
            out['fisher'] = None

        if fishvec:
            # = 1/2 tr(K⁻¹dKK⁻¹d'K) v
            #   + dr'K⁻¹d'r v
            # tr(K⁻¹dKK⁻¹d'K) v = K_vjp(K⁻¹K_jvp(v)K⁻¹) =
            #                   = K_vjp(L'⁻¹L⁻¹ K_jvp(v) L'⁻¹L⁻¹)
            # dr'K⁻¹d'r v = dr'K⁻¹dr_jvp(v) =
            #             = dr_vjp(K⁻¹dr_jvp(v)) =
            #             = dr_vjp(L'⁻¹L⁻¹ dr_jvp(v))
            out['fishvec'] = 0
            if not (dK_jvp_vec is None and dK_vjp is None):
                invL_dKv = jlinalg.solve_triangular(L, dK_jvp_vec, lower=True)
                invK_dKv = jlinalg.solve_triangular(L.T, invL_dKv, lower=False)
                invL_dKv_invK = jlinalg.solve_triangular(L, invK_dKv.T, lower=True)
                invK_dKv_invK = jlinalg.solve_triangular(L.T, invL_dKv_invK, lower=False)
                tr_invK_dK_invK_dK_v = dK_vjp(invK_dKv_invK)
                out['fishvec'] += 1/2 * tr_invK_dK_invK_dK_v
            if not (dr_jvp_vec is None and dr_vjp is None):
                invL_drv = jlinalg.solve_triangular(L, dr_jvp_vec, lower=True)
                invK_drv = jlinalg.solve_triangular(L.T, invL_drv, lower=False)
                dr_invK_drv_v = dr_vjp(invK_drv)
                out['fishvec'] += dr_invK_drv_v
        else:
            out['fishvec'] = None

        return tuple(out.values())

    @classmethod
    def make_derivs(cls,
        K_fun, r_fun, primal,
        *,
        args=(),
        kw={},
        vec=None,
        value=False,
        gradrev=False,
        gradfwd=False,
        fisher=False,
        fishvec=False,
    ):
        """
        Prepares arguments for `minus_log_normal_density`.

        Parameters
        ----------
        K_fun, r_fun : callable
            Functions with signature ``f(primal, *args, **kw)`` that produce the
            `K` init argument and the `r` `minus_log_normal_density` argument.
        primal : 1d array
            The first argument to `K_fun` and `r_fun`.
        args : tuple
            Additional positional arguments to `K_fun` and `r_fun`.
        kw : dict
            Keyword arguments to `K_fun` and `r_fun`.
        vec : 1d array
            A tangent vector to compute the jacobian-vector products.
        value, gradrev, gradfwd, fisher, fishvec : bool
            Arguments to `minus_log_normal_density`, used to determine which
            derivatives are needed.

        Returns
        -------
        K : 2d array
            Output of `K_fun`.
        r : 1d array
            Output of `r_fun`.
        out : dict
            Dictionary with derivative arguments to `minus_log_normal_density`.
        """

        partial = lambda f: lambda x: f(x, *args, **kw)
        K_fun = partial(K_fun)
        r_fun = partial(r_fun)

        out = {}

        if gradrev or fishvec:
            K, dK_vjp = jax.vjp(K_fun, primal)
            r, dr_vjp = jax.vjp(r_fun, primal)
            out['dK_vjp'] = lambda x: dK_vjp(x)[0]
            out['dr_vjp'] = lambda x: dr_vjp(x)[0]
        else:
            K = K_fun(primal)
            r = r_fun(primal)
        if fishvec:
            _, out['dK_jvp_vec'] = jax.jvp(K_fun, (primal,), (vec,))
            _, out['dr_jvp_vec'] = jax.jvp(r_fun, (primal,), (vec,))
        if gradfwd or fisher:
            out['dK'] = jax.jacfwd(K_fun)(primal)
            out['dr'] = jax.jacfwd(r_fun)(primal)

        return K, r, out

    @property
    def n(self):
        return len(self._L)

    m = n
