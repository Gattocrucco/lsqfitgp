# lsqfitgp/tests/linalg/test_decomp.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

import pytest
from scipy import linalg, stats
import numpy as np
import jax
from jax import numpy as jnp
from jax.scipy import linalg as jlinalg
import gvar

from lsqfitgp import _linalg, _patch_jax
from .. import util

class TestChol:

    def randortho(self, n, *, rng=None):
        """ generate a random nxn orthogonal matrix """
        rng = np.random.default_rng(rng)
        if n > 1:
            return stats.ortho_group.rvs(n, random_state=rng)
        else:
            # stats.ortho_group does not support n < 2
            return np.atleast_2d(2 * rng.integers(2) - 1)

    def mat(self, n, s, *, rank=None, eps=1e-3, rng=None):
        """
        Generate a p.s.d. matrix that depends smoothly on parameters.

        Parameters
        ----------
        n : int
            The size of the matrix.
        s : 1d array
            The parameters the matrix depends on. Length 1.
        rank : int, optional
            The rank of the matrix. If not specified, it is equal to `n`. If
            singular, the null space of the matrix depends on `s`.
        eps : float, optional
            Lower limit on non-null eigenvalues of the matrix. Default 0.001.
        rng : seed or random number generator, optional
            Generator used to pick the eigenspaces. Anything accepted by
            `numpy.random.default_rng` goes.

        Returns
        -------
        out : (n, n) array
            A positive semidefinite matrix.
        """
        rng = np.random.default_rng(rng)
        if rank is None:
            rank = n
        s = s[0]
        eigvals = 1 + eps + jnp.cos(s + jnp.arange(n))
        eigvals = jnp.where(jnp.arange(n) < rank, eigvals, 0)
        transf = self.randortho(n, rng=rng)
        if n >= 2:
            rot = jnp.array([[jnp.cos(s), -jnp.sin(s)],
                             [jnp.sin(s),  jnp.cos(s)]])
            indices = rng.choice(n, size=2, replace=False)
            rot = jnp.eye(n).at[jnp.ix_(indices, indices)].set(rot)
            transf = rot @ transf
        with _patch_jax.skipifabstract():
            util.assert_allclose(transf @ transf.T, np.eye(n), atol=1e-14)
        return (transf * eigvals) @ transf.T

    @pytest.fixture(params=[1, 2, 10])
    def n(self, request):
        """ Size of the test matrix """
        return request.param

    @pytest.fixture
    def s(self, rng):
        """ A value for the namesake `mat` parameter """
        return rng.uniform(-np.pi, np.pi, 1)

    @pytest.fixture
    def K_factory(self, n, rng):
        """ A function of `s` producing the matrix to be decomposed """
        high = np.iinfo(np.uint64).max
        seed = rng.integers(high, dtype=np.uint64, endpoint=True)
        def K_factory(s):
            return self.mat(n, s, rng=seed)
        return K_factory

    @pytest.fixture
    def K(self, s, K_factory):
        """ The matrix to be decomposed """
        return K_factory(s)

    @pytest.fixture
    def r_factory(self, n, rng):
        """ A function of `s` producing an auxiliary vector """
        high = np.iinfo(np.uint64).max
        seed = rng.integers(high, dtype=np.uint64, endpoint=True)
        def r_factory(s):
            rng = np.random.default_rng(seed)
            return s[0] * rng.standard_normal(n)
        return r_factory

    @pytest.fixture
    def r(self, s, r_factory):
        """ Auxiliary vector with as many rows as K """
        return r_factory(s)

    @pytest.fixture
    def A(self, n, rng):
        """ Auxiliary matrix with as many rows as K """
        return rng.standard_normal((n, 2 * n))

    @pytest.fixture
    def likelihood(self, n, K_factory, r_factory):
        """ A likelihood w.r.t. the `s` parameter """
        def likelihood(s):
            K = K_factory(s)
            r = r_factory(s)
            return 1/2 * (
                n * jnp.log(2 * jnp.pi) +
                jnp.sum(jnp.log(jnp.linalg.eigvalsh(K))) +
                r @ jlinalg.solve(K, r, assume_a='pos')
            )
        return likelihood

    @pytest.fixture
    def decomp(self, K):
        """ Decomposition of K """
        return _linalg.Chol(K)

    def test_ginv_linear(self, n, K, A, decomp):
        result = decomp.ginv_linear(A)
        expected = jlinalg.solve(K, A, assume_a='pos')
        util.assert_close_matrices(result, expected, rtol=1e-11)

    def test_pinv_bilinear_direct(self, n, K, A, r, decomp):
        result = decomp.pinv_bilinear(A, r)
        K_reg = K + np.eye(n) * decomp.eps
        expected = A.T @ linalg.solve(K_reg, r, assume_a='pos')
        util.assert_close_matrices(result, expected, rtol=1e-14)

    def test_pinv_bilinear_proj(self, n, K, r, decomp):
        result = decomp.pinv_bilinear(K, r)
        expected = r
        util.assert_close_matrices(result, expected, rtol=1e-13)

    def test_pinv_bilinear_robj_direct(self, n, K, A, rng, decomp):
        r = gvar.gvar(rng.standard_normal(n), rng.gamma(2, 1 / 2, n))
        result = decomp.pinv_bilinear_robj(A, r)
        K_reg = K + np.eye(n) * decomp.eps
        expected = linalg.solve(K_reg, A, assume_a='pos').T @ r
        util.assert_same_gvars(result, expected, rtol=1e-11)

    def test_ginv_quad_direct(self, n, K, A, decomp):
        result = decomp.ginv_quad(A)
        K_reg = K + np.eye(n) * decomp.eps
        expected = A.T @ linalg.solve(K_reg, A, assume_a='pos')
        util.assert_close_matrices(result, expected, rtol=1e-12)

    def test_ginv_quad_mp1(self, K, decomp):
        result = decomp.ginv_quad(K)
        expected = K
        util.assert_close_matrices(result, expected, rtol=1e-14)

    def test_ginv_diagquad_direct(self, n, K, A, decomp):
        result = decomp.ginv_diagquad(A)
        K_reg = K + np.eye(n) * decomp.eps
        expected = A.T @ linalg.solve(K_reg, A, assume_a='pos')
        expected = np.diag(expected)
        util.assert_close_matrices(result, expected, rtol=1e-12)

    def test_ginv_diagquad_mp1(self, K, decomp):
        result = decomp.ginv_diagquad(K)
        expected = np.diag(K)
        util.assert_close_matrices(result, expected, rtol=1e-13)

    def test_correlate(self, n, K, decomp):
        Z = decomp.correlate(np.eye(n))
        result = Z @ Z.T
        expected = K
        util.assert_close_matrices(result, expected, rtol=1e-14)

    def test_back_correlate(self, n, K, decomp):
        Zt = decomp.back_correlate(np.eye(n))
        result = Zt.T @ Zt
        expected = K
        util.assert_close_matrices(result, expected, rtol=1e-14)

    def test_pinv_correlate(self, n, K, decomp):
        result = decomp.pinv_correlate(K) # = Z⁺K = Z⁺ZZ' = Z'
        expected = decomp.back_correlate(np.eye(n)) # = Z'I = Z'
        util.assert_close_matrices(result, expected, rtol=1e-13)

    def test_correlate_back_correlate(self, n, K, decomp, r):
        result = decomp.correlate(decomp.back_correlate(r))
        expected = K @ r
        util.assert_close_matrices(result, expected, rtol=1e-14)

    def test_normal_nothing(self, decomp, r):
        result = decomp.minus_log_normal_density(r)
        expected = 5 * (None,)
        assert result == expected

    def test_normal_value(self, n, K, decomp, r):
        result, _, _, _, _ = decomp.minus_log_normal_density(r, value=True)
        expected = 1/2 * (
            n * np.log(2 * np.pi) +
            np.sum(np.log(linalg.eigvalsh(K))) +
            r @ linalg.solve(K, r, assume_a='pos')
        )
        util.assert_allclose(result, expected, atol=1e-9)

    def test_normal_gradrev(self, likelihood, s, decomp, r, r_factory, K_factory):
        _, _, kw = decomp.make_derivs(K_factory, r_factory, s, gradrev=True)
        _, result, _, _, _ = decomp.minus_log_normal_density(r, gradrev=True, **kw)
        jac = jax.jacrev(likelihood)
        expected = jac(s)
        util.assert_close_matrices(result, expected, rtol=1e-11)

    def test_normal_gradfwd(self, likelihood, s, decomp, r, r_factory, K_factory):
        _, _, kw = decomp.make_derivs(K_factory, r_factory, s, gradfwd=True)
        _, _, result, _, _ = decomp.minus_log_normal_density(r, gradfwd=True, **kw)
        jac = jax.jacfwd(likelihood)
        expected = jac(s)
        util.assert_close_matrices(result, expected, rtol=1e-10)

    def test_normal_fisher(self, s, decomp, r, K, K_factory, r_factory):
        _, _, kw = decomp.make_derivs(K_factory, r_factory, s, fisher=True)
        _, _, _, result, _ = decomp.minus_log_normal_density(r, fisher=True, **kw)
        dK = jax.jacfwd(K_factory)(s)
        dr = jax.jacfwd(r_factory)(s)
        invK_dK = _linalg.solve_batched(K, np.moveaxis(dK, 2, 0), assume_a='pos')
        expected = 1/2 * np.einsum('kij,qji->kq', invK_dK, invK_dK)
        expected += dr.T @ linalg.solve(K, dr, assume_a='pos')
        util.assert_close_matrices(result, expected, rtol=1e-12)

    def test_normal_fishvec(self, s, decomp, r, K, K_factory, r_factory, rng):
        vec = rng.standard_normal(s.shape)
        _, _, kw = decomp.make_derivs(K_factory, r_factory, s, fishvec=True, vec=vec)
        _, _, _, _, result = decomp.minus_log_normal_density(r, fishvec=True, **kw)
        dK = jax.jacfwd(K_factory)(s)
        dr = jax.jacfwd(r_factory)(s)
        invK_dK = _linalg.solve_batched(K, np.moveaxis(dK, 2, 0), assume_a='pos')
        fisher = 1/2 * np.einsum('kij,qji->kq', invK_dK, invK_dK)
        fisher += dr.T @ linalg.solve(K, dr, assume_a='pos')
        expected = fisher @ vec
        util.assert_close_matrices(result, expected, rtol=1e-11)
