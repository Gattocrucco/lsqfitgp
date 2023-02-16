# lsqfitgp/tests/test_bart.py
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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
import pytest
from pytest import mark

sys.path.insert(0, '.')
import lsqfitgp as lgp

import util

gen = np.random.default_rng(202207191826)

plist = [1, 2, 5]
smark = mark.parametrize('sb,sbw,sa,w', sum([[
    (*gen.integers(0, 4, (3, p)), gen.integers(1, 10, p)),
    (*np.zeros((3, p), int), gen.integers(1, 10, p)),
    (gen.integers(0, 10, p), (np.arange(p) == gen.integers(p)).astype(int), gen.integers(0, 10, p), gen.integers(1, 10, p)),
] for p in plist], []))
amark = mark.parametrize('a', [
    pytest.param(0, id='a0'),
    pytest.param(1, id='a1'),
    pytest.param(np.linspace(0.01, 0.99, 7)[:, None], id='aother'),
])
bmark = mark.parametrize('b', [
    pytest.param(0, id='b0'),
    pytest.param(np.inf, id='binf'),
    pytest.param(np.linspace(1, 10, 10), id='bother'),
])
umark = mark.parametrize('u', [np.arange(0, 2)[:, None, None]])
mdmark = mark.parametrize('md', range(5))

@mdmark
@bmark
@amark
@smark
def test_lower_lt_upper(sb, sbw, sa, w, a, b, md):
    """ 0 <= lower <= interp/stricter upper <= upper <= 1 """
    lw = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=0, maxd=md, weights=w)
    au = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma='auto' if 1 <= md <= 3 else 0, maxd=md, weights=w)
    vg = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=1, maxd=md * 2, reset=[md], weights=w)
    up = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=1, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(np.zeros_like(lw), np.minimum(0, lw))
    np.testing.assert_array_max_ulp(lw, np.minimum(lw, au))
    np.testing.assert_array_max_ulp(lw, np.minimum(lw, vg))
    np.testing.assert_array_max_ulp(au, np.minimum(au, up))
    np.testing.assert_array_max_ulp(vg, np.minimum(vg, up))
    np.testing.assert_array_max_ulp(up, np.minimum(up, 1))

@bmark
@amark
@smark
def test_lower_upper_incr_maxd(sb, sbw, sa, w, a, b):
    """
    lower/upper increases/decreases as maxd is increased
    """
    for md in range(4):
        lw = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=0, maxd=md, weights=w)
        up = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=1, maxd=md, weights=w)
        if md:
            np.testing.assert_array_max_ulp(plw, np.minimum(lw, plw), 2)
            np.testing.assert_array_max_ulp(pup, np.maximum(up, pup), 2)
        plw = lw
        pup = up

@mdmark
@umark
@bmark
@amark
@smark
def test_incr_beta(sb, sbw, sa, w, a, b, u, md):
    """increases as beta is increased"""
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    ci = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b + 1e-3, gamma=u, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(ci, np.maximum(c, ci))

@mdmark
@umark
@bmark
@amark
@smark
def test_incr_alpha(sb, sbw, sa, w, a, b, u, md):
    """decreases as alpha is increased"""
    da = 1e-3
    a = np.minimum(1 - da, a)
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    ci = lgp.BART.correlation(sb, sbw, sa, alpha=a + da, beta=b, gamma=u, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(ci, np.minimum(c, ci))

@mdmark
@umark
@bmark
@amark
@mark.parametrize('sb,sbw,sa,w', sum([[
    # n^0 = 0
    (gen.integers(0, 10, p), np.zeros(p, int), gen.integers(0, 10, p), gen.integers(1, 10, p)),
    (*np.zeros((3, p), int), gen.integers(1, 10, p)),
] for p in plist], []) + sum([[ 
    # w = 0
    (*gen.integers(0, 4, (3, p)), np.zeros(p)),
] for p in plist], []) + sum([[
    # wi = 0 or ni = 0
    np.concatenate([gen.integers(0, 4, (3, p)), gen.integers(1, 10, (1, p))]) *
    (gen.integers(0, 2, p).astype(bool) ^ np.array([0, 0, 0, 1], bool)[:, None]),
] for p in plist], []) + [
    # p = 0
    (*np.empty((3, 0), int), np.empty(0)),
])
def test_corr_1(sb, sbw, sa, w, a, b, u, md):
    """
    correlation = 1 if:
        - n^0 = 0
        - w = 0
        - wi = 0 or ni = 0
        - p = 0
    """
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(c, np.ones_like(c))

@mdmark
@umark
@bmark
@amark
@smark
def test_swap_ab(sb, sbw, sa, w, a, b, u, md):
    """invariant under swapping of nplus and nminus"""
    swap = gen.integers(0, 2, sb.size)
    s1 = np.where(swap, sa, sb)
    s2 = np.where(swap, sb, sa)
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    cs = lgp.BART.correlation(s1, sbw, s2, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(c, cs, 16)

@mdmark
@umark
@bmark
@amark
@smark
def test_perm_dims(sb, sbw, sa, w, a, b, u, md):
    """invariant under reordering of the dimensions"""
    perm = gen.permutation(sb.size)
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    cp = lgp.BART.correlation(sb[perm], sbw[perm], sa[perm], alpha=a, beta=b, gamma=u, maxd=md, weights=w[perm])
    np.testing.assert_array_max_ulp(c, cp, 31)

@mdmark
@umark
@bmark
@amark
@mark.parametrize('sb,sbw,sa,w', sum([[
    (*gen.integers(0, 10, (3, p)), gen.integers(1, 10, p)),
] for p in plist], []))
def test_incr_n0(sb, sbw, sa, w, a, b, u, md):
    """correlation decreases as n0 increases at fixed ntot"""
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)

    ntot = sb + sbw + sa
    which = gen.permuted(np.eye(sb.size)[0]).astype(bool)
    dn = np.where(which & (sb + sa), 1, 0)
    lr = gen.integers(0, 2, sb.size).astype(bool)
    lr ^= lr & ~sb.astype(bool) | ~lr & ~sa.astype(bool)
    # lr sb sa
    # 0  *  0   1
    # 0  *  1   0
    # 1  0  *   0
    # 1  1  *   1
    sb -= np.where(lr, dn, 0)
    sbw += dn
    sa -= np.where(~lr, dn, 0)
    assert np.all(ntot == sb + sbw + sa)
    assert np.all(sb >= 0) and np.all(sa >= 0)

    ci = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(ci, np.minimum(c, ci))

def values(mark):
    vals = mark.args[1]
    return sum((getattr(v, 'values', (v,)) for v in vals), ())

@mark.parametrize('sb,sbw,sa,w,a,b,md', [
    # n^0 = 0
    (gen.integers(0, 10, p), np.zeros(p, int), gen.integers(0, 10, p), gen.integers(1, 10, p), a, b, d)
    for p in plist
    for a in values(amark)
    for b in values(bmark)
    for d in range(0, mdmark.args[1].stop)
] + [
    # alpha = 0
    (gen.integers(0, 10, p), gen.integers(1, 10, p), gen.integers(0, 10, p), gen.integers(1, 10, p), 0, b, d)
    for p in plist
    for b in values(bmark)
    for d in range(0, mdmark.args[1].stop)
] + [
    # beta = inf
    (gen.integers(0, 10, p), gen.integers(1, 10, p), gen.integers(0, 10, p), gen.integers(1, 10, p), a, np.inf, d)
    for p in plist
    for a in values(amark)
    for d in range(1, mdmark.args[1].stop)
])
def test_lower_eq_upper(sb, sbw, sa, w, a, b, md):
    """
    upper = lower in cases where the solution is exact
     - beta = inf if maxd > 0
     - alpha = 0
     - n^0 = 0
    """
    lw = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=0, maxd=md, weights=w)
    up = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=1, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(lw, up)

@mark.parametrize('md', range(4))
@umark
@bmark
@amark
@smark
def test_no_shortcuts(sb, sbw, sa, w, a, b, u, md):
    """check the result is the same if no recursions are avoided"""
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    cd = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, debug=True, weights=w)
    np.testing.assert_array_max_ulp(c, cd, 32)

@mdmark
@umark
@bmark
@amark
@smark
def test_nzero(sb, sbw, sa, w, a, b, u, md):
    """adding zeros in sb, sbw, sa does not change the result"""
    kw = dict(alpha=a, beta=b, gamma=u, maxd=md)
    c = lgp.BART.correlation(sb, sbw, sa, weights=w, **kw)
    z = lambda x, f=0, p=4: np.concatenate([x, np.full(p, f)])
    c0 = lgp.BART.correlation(z(sb), z(sbw), z(sa), weights=z(w, 1), **kw)
    np.testing.assert_array_max_ulp(c, c0, 2)

@mdmark
@umark
@bmark
@amark
@smark
def test_wzero(sb, sbw, sa, w, a, b, u, md):
    """adding zeros to weights with random n does not change the result"""
    kw = dict(alpha=a, beta=b, gamma=u, maxd=md)
    c = lgp.BART.correlation(sb, sbw, sa, weights=w, **kw)
    z = lambda x, f=5, p=4: np.concatenate([x, gen.integers(f + 1, size=p)])
    c0 = lgp.BART.correlation(z(sb), z(sbw), z(sa), weights=z(w, 0), **kw)
    np.testing.assert_array_max_ulp(c, c0, 2)

def test_structured():
    X = np.arange(10 * 2.).reshape(1, -1, 2).view('d,d')
    splits = lgp.BART.splits_from_coord(X)
    cov = lgp.BART(splits=splits)(X, X.T)

def test_duplicates():
    x = np.repeat(np.arange(10 * 2.).reshape(-1, 2), 2, axis=0).view('d,d').squeeze(axis=-1)
    length, splits = lgp.BART.splits_from_coord(x)
    assert np.all(length == 9)

def test_integer():
    X = np.arange(10 * 2).reshape(1, -1, 2)
    X1 = X.astype('d').view('d,d').squeeze(-1)
    X2 = X.astype('l').view('l,l').squeeze(-1)
    kernel1 = lgp.BART(splits=lgp.BART.splits_from_coord(X1))
    kernel2 = lgp.BART(splits=lgp.BART.splits_from_coord(X2))
    v1 = kernel1(X1, X1.T)
    v2 = kernel2(X1, X1.T)
    util.assert_equal(v1, v2)

def test_pnt():
    sss = [1, 2], [3, 1], [5, 8]
    alpha = 0.9
    beta = 1.6
    c1 = lgp.BART.correlation(*sss, alpha=alpha, beta=beta, maxd=2)
    c2 = lgp.BART.correlation(*sss, alpha=None, beta=None, pnt=[alpha, alpha/2 ** beta, alpha/3 ** beta])
    util.assert_allclose(c1, c2)

def test_wrong_gamma():
    with pytest.raises(KeyError):
        lgp.BART.correlation([0], [0], [0], gamma='ciao')

def test_intercept():
    sss = [1, 2], [3, 1], [5, 8]
    alpha = 0.9
    c1 = lgp.BART.correlation(*sss, alpha=alpha)
    c2 = lgp.BART.correlation(*sss, alpha=alpha, intercept=False)
    c2 = c2 * alpha + (1 - alpha)
    util.assert_allclose(c1, c2, rtol=1e-15)

def test_splits_1d():
    l = [2]
    s = np.array([-1/2, 1/2])
    x = np.array([-1, 0, 1])[:, None]
    k1 = lgp.BART(splits=(l, s))
    k2 = lgp.BART(splits=(l, s[:, None]))
    v1 = k1(x, x.T)
    v2 = k1(x, x.T)
    util.assert_equal(v1, v2)

def test_f32():
    sss = [1, 2], [3, 1], [5, 8]
    c = lgp.BART.correlation(*sss, pnt=np.array([0.9, 0.4, 0.3], 'f'))
    assert c.dtype == 'f'

# TODO
# - increases at fixed n0 and ntot if the difference between nminus and nplus
#   decreases (not completely sure)
# - test gamma='auto' gives 0 for beta=0, alpha=1
# - test of equality with precomputed values sampled with qmc
# - duplicated entries in reset have no effect
