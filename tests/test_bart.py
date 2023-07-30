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

import numpy as np
import pytest
from pytest import mark

import lsqfitgp as lgp

from . import util

rng = np.random.default_rng(202307302223)

plist = [1, 5]
smark = mark.parametrize('sb,sbw,sa,w', sum([[
    (*rng.integers(0, 4, (3, p)), rng.integers(1, 10, p)),
    (*np.zeros((3, p), int), rng.integers(1, 10, p)),
    (np.zeros(p, int), np.pad([1], (0, p - 1)), np.zeros(p, int), rng.integers(1, 10, p)),
    (rng.integers(0, 10, p), (np.arange(p) == rng.integers(p)).astype(int), rng.integers(0, 10, p), rng.integers(1, 10, p)),
] for p in plist], []))
amark = mark.parametrize('a', [
    pytest.param(np.array([0, 1])[:, None], id='a01'),
    pytest.param(np.linspace(0.01, 0.99, 7)[:, None], id='aother'),
])
bmark = mark.parametrize('b', [
    pytest.param(np.array([0, np.inf]), id='b0inf'),
    pytest.param(np.linspace(1, 10, 10), id='bother'),
])
umark = mark.parametrize('u', [np.array([0, 1])[:, None, None]])
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
    (rng.integers(0, 10, p), np.zeros(p, int), rng.integers(0, 10, p), rng.integers(1, 10, p)),
    (*np.zeros((3, p), int), rng.integers(1, 10, p)),
] for p in plist], []) + sum([[ 
    # w = 0
    (*rng.integers(0, 4, (3, p)), np.zeros(p)),
] for p in plist], []) + sum([[
    # wi = 0 or ni = 0
    np.concatenate([rng.integers(0, 4, (3, p)), rng.integers(1, 10, (1, p))]) *
    (rng.integers(0, 2, p).astype(bool) ^ np.array([0, 0, 0, 1], bool)[:, None]),
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
def test_swap_ab(sb, sbw, sa, w, a, b, u, md, rng):
    """invariant under swapping of nplus and nminus"""
    swap = rng.integers(0, 2, sb.size)
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
def test_perm_dims(sb, sbw, sa, w, a, b, u, md, rng):
    """invariant under reordering of the dimensions"""
    perm = rng.permutation(sb.size)
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    cp = lgp.BART.correlation(sb[perm], sbw[perm], sa[perm], alpha=a, beta=b, gamma=u, maxd=md, weights=w[perm])
    np.testing.assert_array_max_ulp(c, cp, 64)

@mdmark
@umark
@bmark
@amark
@mark.parametrize('sb,sbw,sa,w', sum([[
    (*rng.integers(0, 10, (3, p)), rng.integers(1, 10, p)),
] for p in plist], []))
def test_incr_n0(sb, sbw, sa, w, a, b, u, md, rng):
    """correlation decreases as n0 increases at fixed ntot"""
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)

    ntot = sb + sbw + sa
    which = rng.permuted(np.eye(sb.size)[0]).astype(bool)
    dn = np.where(which & (sb + sa), 1, 0)
    lr = rng.integers(0, 2, sb.size).astype(bool)
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
    (rng.integers(0, 10, p), np.zeros(p, int), rng.integers(0, 10, p), rng.integers(1, 10, p), a, b, d)
    for p in plist
    for a in values(amark)
    for b in values(bmark)
    for d in range(0, mdmark.args[1].stop)
] + [
    # alpha = 0
    (rng.integers(0, 10, p), rng.integers(1, 10, p), rng.integers(0, 10, p), rng.integers(1, 10, p), 0, b, d)
    for p in plist
    for b in values(bmark)
    for d in range(0, mdmark.args[1].stop)
] + [
    # beta = inf
    (rng.integers(0, 10, p), rng.integers(1, 10, p), rng.integers(0, 10, p), rng.integers(1, 10, p), a, np.inf, d)
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
    np.testing.assert_array_max_ulp(c, c0, 4)

@mdmark
@umark
@bmark
@amark
@smark
def test_wzero(sb, sbw, sa, w, a, b, u, md, rng):
    """adding zeros to weights with random n does not change the result"""
    kw = dict(alpha=a, beta=b, gamma=u, maxd=md)
    c = lgp.BART.correlation(sb, sbw, sa, weights=w, **kw)
    z = lambda x, f=5, p=4: np.concatenate([x, rng.integers(f + 1, size=p)])
    c0 = lgp.BART.correlation(z(sb), z(sbw), z(sa), weights=z(w, 0), **kw)
    np.testing.assert_array_max_ulp(c, c0, 4)

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

def test_i32():
    """ test that passing 32 bit integers does not result in 32 bit floating
    point calculations, which would happen because jax casts int32 to float32
    """
    sss = [1, 2], [3, 1], [5, 8]
    sss = [np.array(s, np.int32) for s in sss]
    pnt = np.array([0.9, 0.4, 0.3], np.float64)
    c1 = lgp.BART.correlation(*sss, pnt=pnt)
    assert c1.dtype == np.float64
    sss = [np.array(s, np.int64) for s in sss]
    c2 = lgp.BART.correlation(*sss, pnt=pnt)
    assert c2.dtype == np.float64
    np.testing.assert_array_max_ulp(c1, c2, 0)

@mark.parametrize('md,reset', [
    (0, None),
    (1, None),
    (2, None), (2, [1]),
    (3, None), (3, [1, 2]),
    (4, None), (4, [2]), (4, [1, 2, 3]),
])
@umark
@bmark
@amark
@smark
def test_altinput(sb, sbw, sa, w, a, b, u, md, reset, rng):
    """ two alternative implementations give the same result """
    c1 = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, reset=reset, weights=w)
    n = sb + sbw + sa
    ix = sb
    iy = sb + sbw
    ix, iy = np.broadcast_arrays(ix, iy)
    swap = rng.integers(0, 2, size=ix.shape, dtype=bool)
    ix, iy = np.where(swap, iy, ix), np.where(swap, ix, iy)
    c2 = lgp.BART.correlation(n, ix, iy, alpha=a, beta=b, gamma=u, maxd=md, reset=reset, weights=w, altinput=True)
    util.assert_allclose(c1, c2, rtol=1e-16, atol=2e-15)

def test_index_input(rng):
    """ passing coordinates or directly indices gives the same result """
    n, p = 100, 10
    asstruct = lambda x: np.asarray(x).view([('f0', x.dtype, p)]).squeeze(-1)
    X = asstruct(rng.standard_normal((n, p)))
    splits = lgp.BART.splits_from_coord(X)
    x, y = asstruct(rng.standard_normal((2, n, p)))
    x, y = np.ix_(x, y)
    ix = asstruct(lgp.BART.indices_from_coord(x, splits))
    iy = asstruct(lgp.BART.indices_from_coord(y, splits))
    c1 = lgp.BART(splits=splits)(x, y)
    c2 = lgp.BART(splits=splits, indices=True)(ix, iy)
    np.testing.assert_array_max_ulp(c1, c2, 0)

# TODO
# - increases at fixed n0 and ntot if the difference between nminus and nplus
#   decreases (not completely sure)
# - test gamma='auto' gives 0 for beta=0, alpha=1
# - test of equality with precomputed values sampled with qmc
# - duplicated entries in reset have no effect
# - continuity w.r.t. weight in zero, unless all differring covariates disappear
#   and the points collapse (which should still be continuous if gamma=1)
# - reset
# - splitting points are checked only if indices=False
