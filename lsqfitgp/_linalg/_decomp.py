# lsqfitgp/_linalg/_decomp.py
#
# Copyright (c) 2023 Giacomo Petrillo
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

import numpy
from jax import numpy as jnp
from jax.scipy import linalg as jlinalg

from .. import _patch_jax

class Decomposition(abc.ABC):
    """
    Abstract base class for decompositions of positive semidefinite matrices.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kw):
        """Decompose the input matrix."""
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
        """Compute Zx where K = ZZ'."""
        pass

    @abc.abstractmethod
    def back_correlate(self, X):
        """Compute Z'X."""
        pass

    @abc.abstractmethod
    def minus_log_normal_density(self, *,
        r,             # 1d array, the residuals (data - prior mean)
        dr_vjp=None,   # callable, x -> x_i ∂r_i/∂p_j,   gradrev and fishvec
        dK_vjp=None,   # callable, x -> x_ij ∂K_ij/∂p_k, gradrev and fishvec
        dr_jvp=None,   # callable, x -> ∂r_i/∂r_j x_j,  fishvec
        dK_jvp=None,   # callable, x -> ∂K_ij/∂p_k x_k, fishvec
        dr=None,       # 2d array, ∂r_i/∂p_j,  gradfwd and fisher
        dK=None,       # 3d array, ∂K_ij/∂p_k, gradfwd and fisher
        vec=None,      # 1d array, input vector of fishvec, same size as params
        value=False,   # bool
        gradrev=False, # bool
        gradfwd=False, # bool
        fisher=False,  # bool
        fishvec=False, # bool
        ):
        """
        Compute the minus log of the normal density of residuals.

        If a derivative is not specified, it is assumed to be zero.

        Parameters
        ----------
        r: 1d array      # the residuals (data - prior mean)
        dr_vjp: callable # x -> x_i ∂r_i/∂p_j,   gradrev and fishvec
        dK_vjp: callable # x -> x_ij ∂K_ij/∂p_k, gradrev and fishvec
        dr_jvp: callable # x -> ∂r_i/∂r_j x_j,  fishvec
        dK_jvp: callable # x -> ∂K_ij/∂p_k x_k, fishvec
        dr: 2d array     # ∂r_i/∂p_j,  gradfwd and fisher
        dK: 3d array     # ∂K_ij/∂p_k, gradfwd and fisher
        vec: 1d array    # input vector of fishvec, same size as params
        value: bool
        gradrev: bool
        gradfwd: bool
        fisher: bool
        fishvec: bool

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
        fishvec: fisher matrix times vec
        """
        pass

    def _parseeps(self, K, epsrel, epsabs, maxeigv=None):
        """ Determine eps from input arguments """
        machine_eps = jnp.finfo(_patch_jax.float_type(K)).eps
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
        s = diag_scale_pow2(K)
        K = K / s / s[:, None]
        eps = self._parseeps(K, epsrel, epsabs)
        K = K.at[jnp.diag_indices_from(K)].add(eps)
        L = jlinalg.cholesky(K, lower=True)
        with _patch_jax.skipifabstract():
            if not jnp.all(jnp.isfinite(L)):
                # TODO check that jax fills with nan after failed row, detect
                # and report minor index like scipy
                raise numpy.linalg.LinAlgError('cholesky decomposition not finite, probably matrix not pos def numerically')
        self._L = L * s[:, None]

    def pinv_bilinear(self, A, r):
        # = A'K⁻¹r
        # K⁻¹ = L'⁻¹L⁻¹
        # A'K⁻¹r = A'L'⁻¹L⁻¹r = (L⁻¹A)'(L⁻¹r)
        invLr = jlinalg.solve_triangular(self._L, r, lower=True)
        invLA = jlinalg.solve_triangular(self._L, A, lower=True)
        return invLA.T @ invLr

    def pinv_bilinear_robj(self, A, r):
        # = A'K⁻¹r = A'L'⁻¹L⁻¹r = (L⁻¹A)'(L⁻¹r)
        invLr = solve_triangular(self._L, r, lower=True)
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

    def minus_log_normal_density(self, *,
        r,             # 1d array, the residuals (data - prior mean)
        dr_vjp=None,   # callable, x -> x_i ∂r_i/∂p_j,   gradrev and fishvec
        dK_vjp=None,   # callable, x -> x_ij ∂K_ij/∂p_k, gradrev and fishvec
        dr_jvp=None,   # callable, x -> ∂r_i/∂r_j x_j,  fishvec
        dK_jvp=None,   # callable, x -> ∂K_ij/∂p_k x_k, fishvec
        dr=None,       # 2d array, ∂r_i/∂p_j,  gradfwd and fisher
        dK=None,       # 3d array, ∂K_ij/∂p_k, gradfwd and fisher
        vec=None,      # 1d array, input vector of fishvec, same size as params
        value=False,   # bool
        gradrev=False, # bool
        gradfwd=False, # bool
        fisher=False,  # bool
        fishvec=False, # bool
        ):

        L = self._L
        
        out = {}

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
            invLr = jlinalg.solve_triangular(L, r, lower=True)
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
            if dK_vjp is not None or dr_vjp is not None:
                invLr = jlinalg.solve_triangular(L, r, lower=True)
                invKr = jlinalg.solve_triangular(L.T, invLr, lower=False)
            if dK_vjp is not None:
                invL = jlinalg.solve_triangular(L, jnp.eye(len(L)), lower=True)
                invK = invL.T @ invL
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
            if dK is not None or dr is not None:
                invLr = jlinalg.solve_triangular(L, r, lower=True)
                invKr = jlinalg.solve_triangular(L.T, invLr, lower=False)
            if dK is not None:
                invL = jlinalg.solve_triangular(L, jnp.eye(len(L)), lower=True)
                invK = invL.T @ invL
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
                invL_dK = jlinalg.solve_triangular(L,
                    jnp.moveaxis(dK, 2, 0),
                    lower=True) # kim: L⁻¹_il dK_lmk
                invL_dK_invL = jlinalg.solve_triangular(L,
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
            if not (dK_jvp is None and dK_vjp is None):
                dKv = K_jvp(v)
                invL_dKv = jlinalg.solve_triangular(L, dKv, lower=True)
                invK_dKv = jlinalg.solve_triangular(L.T, invL_dKv, lower=False)
                invL_dKv_invK = jlinalg.solve_triangular(L, invK_dKv.T, lower=True)
                invK_dKv_invK = jlinalg.solve_triangular(L.T, invL_dKv_invK, lower=False)
                tr_invK_dK_invK_dK_v = K_vjp(invK_dKv_invK)
                out['fishvec'] += 1/2 * tr_invK_dK_invK_dK_v
            if not (dr_vjp is None and dr_vjp is None):
                drv = dr_jvp(v)
                invL_drv = jlinalg.solve_triangular(L, drv, lower=True)
                invK_drv = jlinalg.solve_triangular(L.T, invL_drv, lower=False)
                dr_invK_drv_v = dr_vjp(invK_drv)
                out['fishvec'] += dr_invK_drv_v
        else:
            out['fishvec'] = None

        return tuple(out.values())
