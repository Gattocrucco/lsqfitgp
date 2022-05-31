# lsqfitgp/tests/test_raniter.py
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

import sys

import numpy as np
from scipy import stats
import pytest

sys.path = ['.'] + sys.path
import lsqfitgp as lgp

def make_mean_cov(n):
    mean = np.random.randn(n)
    a = np.random.randn(n, n)
    cov = a.T @ a
    return mean, cov

def test_raniter_randomness():
    n = 40
    mean, cov = make_mean_cov(n)
    samples = list(lgp.raniter(mean, cov, 10 * n))
    smean = np.mean(samples, axis=0)
    scov = np.cov(samples, rowvar=False, ddof=1)
    w, v = np.linalg.eigh(scov / len(samples))
    vsm = v.T @ (smean - mean)
    eps = n * 1e-12 * np.max(w)
    q = (vsm.T / np.maximum(w, eps)) @ vsm
    assert stats.chi2(n).sf(q) > 1e-5
    assert stats.chi2(n).cdf(q) > 1e-5

def test_raniter_packing():
    mean, cov = make_mean_cov(3 + 4)
    dmean = {'a': mean[:3], 'b': mean[3:]}
    dcov = {
        ('a', 'a'): cov[:3, :3],
        ('a', 'b'): cov[:3, 3:],
        ('b', 'a'): cov[3:, :3],
        ('b', 'b'): cov[3:, 3:]
    }
    np.random.seed(0)
    s1 = next(lgp.raniter(mean, cov))
    np.random.seed(0)
    s2 = next(lgp.raniter(dmean, dcov))
    assert np.array_equal(s1, s2.buf)

def test_raniter_warning():
    cov = np.array([1, 1.1, 1.1, 1]).reshape(2, 2)
    with pytest.warns(UserWarning, match='positive definite'):
        next(lgp.raniter(np.zeros(len(cov)), cov))
    with pytest.warns(None) as record:
        next(lgp.raniter(np.zeros(len(cov)), cov, eps=0.1))
    assert len(record) == 0
