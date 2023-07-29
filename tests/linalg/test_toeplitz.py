# lsqfitgp/tests/linalg/test_toeplitz.py
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

import jax
import numpy as np
from scipy import linalg
import pytest

from .. import util

from lsqfitgp import _linalg
from lsqfitgp._linalg import _toeplitz

s1, s2 = np.random.SeedSequence(202307261619).spawn(2)
rng = np.random.default_rng(s1)
np.random.seed(s2.generate_state(1))

@util.tryagain
def test_toeplitz_gershgorin():
    t = rng.standard_normal(100)
    m = linalg.toeplitz(t)
    b1 = _linalg._decomp.eigval_bound(m)
    b2 = _toeplitz.eigv_bound(t)
    util.assert_close_matrices(b2, b1, rtol=1e-15)

def check_toeplitz():
    for n in [10, 2, 1]:
        x = np.linspace(0, 3, n)
        t = np.pi * np.exp(-1/2 * x ** 2)
        m = linalg.toeplitz(t)
    
        l1 = _toeplitz.chol(t)
        l2 = linalg.cholesky(m, lower=True)
        util.assert_close_matrices(l1, l2, rtol=1e-10)
    
        b = rng.standard_normal((len(t), 30))
        lb1 = _toeplitz.chol_matmul(t, b)
        lb2 = l2 @ b
        util.assert_close_matrices(lb1, lb2, rtol=1e-10)

        ld1 = _toeplitz.logdet(t)
        _, ld2 = np.linalg.slogdet(m)
        util.assert_close_matrices(ld1, ld2, rtol=1e-9)
    
        ilb1 = _toeplitz.chol_solve(t, b)
        ilb2 = linalg.solve_triangular(l2, b, lower=True)
        util.assert_close_matrices(ilb1, ilb2, rtol=1e-8)
    
        imb1 = _toeplitz.solve(t, b)
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
                ilb = _toeplitz.chol_solve_numpy(t, b, diageps=1e-12)
                lhs, rhs = np.broadcast_arrays(l @ ilb, b)
                lhs = lhs.reshape(-1 if lhs.size else 0, lhs.shape[-1])
                rhs = rhs.reshape(lhs.shape)
                util.assert_close_matrices(lhs, rhs, rtol=1e-7)
    with pytest.raises(np.linalg.LinAlgError):
        _toeplitz.chol_solve_numpy([-1], [1])
    with pytest.raises(np.linalg.LinAlgError):
        _toeplitz.chol_solve_numpy([1, 2], [1, 1])
    with pytest.raises(np.linalg.LinAlgError):
        _toeplitz.chol_solve_numpy([1, 0.5, 2], [1, 1, 1])
