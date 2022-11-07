# lsqfitgp/tests/test_bart.py
#
# Copyright (c) 2022, Giacomo Petrillo
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

gen = np.random.default_rng(202207191826)

plist = [1, 2, 10]
smark = mark.parametrize('sb,sbw,sa,w', sum([[
    (*gen.integers(0, 10, (3, p)), gen.integers(1, 10, p)),
    (*np.zeros((3, p), int), gen.integers(1, 10, p)),
    (gen.integers(0, 10, p), np.arange(p) == gen.integers(p), gen.integers(0, 10, p), gen.integers(1, 10, p)),
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
umark = mark.parametrize('u', [False, True])
mdmark = mark.parametrize('md', range(5))

@mdmark
@bmark
@amark
@smark
def test_lower_lt_upper(sb, sbw, sa, w, a, b, md):
    """lower <= upper"""
    lw = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=0, maxd=md, weights=w)
    up = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=1, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(lw, np.minimum(lw, up))

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
    (gen.integers(0, 10, p), np.zeros(p, int), gen.integers(0, 10, p), gen.integers(1, 10, p)),
    (*np.zeros((3, p), int), gen.integers(1, 10, p)),
] for p in plist], []))
def test_corr_1(sb, sbw, sa, w, a, b, u, md):
    """correlation = 1 if n^0 = 0"""
    if u or md:
        c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
        np.testing.assert_array_max_ulp(c, np.broadcast_to(1, c.shape))

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
    (gen.integers(0, 10, p), np.zeros(p), gen.integers(0, 10, p), gen.integers(1, 10, p), a, b, d)
    for p in plist
    for a in values(amark)
    for b in values(bmark)
    for d in range(1, mdmark.args[1].stop)
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
     - beta = inf
     - alpha = 0
     - n^0 = 0
    unless maxd = 0, in which case n^0 and beta are ignored
    """
    lw = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=0, maxd=md, weights=w)
    up = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=1, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(lw, up)

@mdmark
@umark
@bmark
@amark
@smark
def test_0_1(sb, sbw, sa, w, a, b, u, md):
    """0 <= correlation <= 1"""
    c = lgp.BART.correlation(sb, sbw, sa, alpha=a, beta=b, gamma=u, maxd=md, weights=w)
    np.testing.assert_array_max_ulp(np.broadcast_to(0, c.shape), np.minimum(0, c))
    np.testing.assert_array_max_ulp(np.broadcast_to(1, c.shape), np.maximum(1, c))

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

# TODO
# - test maxd = 0, 1 with handwritten solution
# - increases at fixed n0 and ntot if the difference between nminus and nplus
#   decreases (not completely sure)
