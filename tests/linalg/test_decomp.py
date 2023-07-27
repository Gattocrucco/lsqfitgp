# lsqfitgp/tests/test_linalg/test_decomp.py
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
from pytest import mark
from scipy import linalg, stats
import numpy as np
from jax import numpy as jnp

from lsqfitgp import _linalg
from .. import util

class TestChol:

    @staticmethod
    def randortho(n, *, rng=None):
        """ generate a random nxn orthogonal matrix """
        rng = np.random.default_rng(rng)
        if n > 1:
            return stats.ortho_group.rvs(n, random_state=rng)
        else:
            # stats.ortho_group does not support n < 2
            return np.atleast_2d(2 * rng.integers(2) - 1)

    @classmethod
    def mat(cls, n, s, *, rank=None, eps=1e-3, rng=None):
        """
        Generate a p.s.d. matrix that depends smoothly on a parameter.

        Parameters
        ----------
        n : int
            The size of the matrix.
        s : float
            The parameter the matrix depends on.
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
        eigvals = 1 + eps + jnp.cos(s + jnp.arange(n))
        eigvals = jnp.where(jnp.arange(n) < rank, eigvals, 0)
        transf = cls.randortho(n, rng=rng)
        if n >= 2:
            rot = jnp.array([[jnp.cos(s), -jnp.sin(s)],
                             [jnp.sin(s),  jnp.cos(s)]])
            indices = rng.choice(n, size=2, replace=False)
            rot = jnp.eye(n).at[jnp.ix_(indices, indices)].set(rot)
            transf = rot @ transf
        util.assert_allclose(transf @ transf.T, np.eye(n), atol=1e-15)
        return (transf * eigvals) @ transf.T

    @classmethod
    def randmat(cls, n, *, rng=None, **kw):
        """ Call `mat` with random `s` argument """
        rng = np.random.default_rng(rng)
        s = rng.uniform(-np.pi, np.pi)
        return cls.mat(n, s, rng=rng, **kw)

    @staticmethod
    def node_path(node):
        """ Take a pytest node, walk up its ancestors collecting all node
        names, return concatenated names """
        names = []
        while node is not None:
            names.append(node.name)
            node = node.parent
        return '::'.join(reversed(names))

    @pytest.fixture
    @classmethod
    def rng(cls, request):
        """ Initialize a random generator for a test, using the full unique test
        path, including parametrizations, as seed """
        path = cls.node_path(request.node)
        seed = np.array([path], np.bytes_).view(np.uint8)
        return np.random.default_rng(seed)

    @pytest.fixture(params=[1, 2, 10])
    @staticmethod
    def n(request):
        """ Size of the test matrix """
        return request.param

    @pytest.fixture
    @staticmethod
    def rank(n):
        """ Rank of the test matrix """
        return n

    def test_pinv_bilinear(self, n, rank, rng):
        K = self.randmat(n, rng=rng, rank=rank)
        A = rng.standard_normal((n, 2 * n))
        r = rng.standard_normal(n)
        decomp = _linalg.Chol(K)
        result = decomp.pinv_bilinear(A, r)
        K_reg = K + np.eye(n) * decomp.eps
        expected = A.T @ linalg.solve(K_reg, r, assume_a='pos')
        util.assert_allclose(result, expected, rtol=1e-14)

# fixtures are run once and cached at their scope (test function, by default)
