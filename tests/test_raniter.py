# lsqfitgp/tests/test_raniter.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

import numpy as np
from scipy import stats
import pytest
import gvar

import lsqfitgp as lgp

def make_mean_cov(n):
    mean = np.random.randn(n)
    a = np.random.randn(n, n)
    cov = a.T @ a
    return mean, cov

def make_mean_cov_dict(*shapes):
    sizes = [np.prod(tuple(s), dtype=int) for s in shapes]
    totsize = sum(sizes)
    mean, cov = make_mean_cov(totsize)
    cumsize = np.cumsum(np.pad(sizes, (1, 0)))
    mean = {
        i: mean[cumsize[i]:cumsize[i + 1]].reshape(shapes[i])
        for i in range(len(shapes))
    }
    cov = {
        (i, j): cov[
            cumsize[i]:cumsize[i + 1],
            cumsize[j]:cumsize[j + 1],
        ].reshape(shapes[i] + shapes[j])
        for i in range(len(shapes))
        for j in range(len(shapes))
    }
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
    n = 3
    mean, cov = make_mean_cov(n + 4)
    dmean = {'a': mean[:n], 'b': mean[n:]}
    dcov = {
        ('a', 'a'): cov[:n, :n],
        ('a', 'b'): cov[:n, n:],
        ('b', 'a'): cov[n:, :n],
        ('b', 'b'): cov[n:, n:]
    }
    seed = 202306241059
    s1 = next(lgp.raniter(mean, cov, rng=seed))
    s2 = next(lgp.raniter(dmean, dcov, rng=seed))
    assert np.array_equal(s1, s2.buf)

def test_raniter_warning():
    cov = np.array([1, 1.1, 1.1, 1]).reshape(2, 2)
    with pytest.warns(UserWarning, match='positive definite'):
        next(lgp.raniter(np.zeros(len(cov)), cov))
    with pytest.warns(None) as record:
        next(lgp.raniter(np.zeros(len(cov)), cov, eps=0.1))
    assert len(record) == 0

def test_raniter_shape():
    shape = (2, 5)
    size = np.prod(shape)
    mean, cov = make_mean_cov(size)
    mean = mean.reshape(shape)
    cov = cov.reshape(2 * shape)
    sample = lgp.sample(mean, cov)
    assert sample.shape == shape

def assert_equal_dict_shapes(a, b):
    for k in a:
        assert a[k].shape == b[k].shape
    for k in b:
        assert k in a

def test_raniter_shape_dict():
    mean, cov = make_mean_cov_dict((), (2, 5), (13,))
    sample = lgp.sample(mean, cov)
    assert_equal_dict_shapes(sample, mean)

def test_raniter_bd():
    mean, cov = make_mean_cov_dict((1,))
    mean = gvar.BufferDict(mean)
    sample = lgp.sample(mean, cov)
