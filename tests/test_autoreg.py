# lsqfitgp/tests/test_autoreg.py
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

import sys

import numpy as np
from numpy.polynomial import polynomial
from scipy import linalg
import pytest

import lsqfitgp as lgp
import util

gen = np.random.default_rng(202207202302)

def gen_ar_acf(p):
    if p:
        mod = gen.uniform(1.1, 10, p)
        phase = gen.uniform(0, 2 * np.pi, p)
        root = mod * np.exp(1j * phase)
        ampl = np.abs(gen.standard_normal(p))
        tau = np.arange(p + 1)
        return np.sum(ampl * root ** -tau[:, None], axis=1).real
    else:
        return np.abs(gen.standard_normal(1))

plist = [0, 1, 2, 10, 30, 100]

def test_gen_ar_acf():
    for p in plist:
        acf = gen_ar_acf(p)
        assert acf.ndim == 1 and acf.size == 1 + p
        mat = linalg.toeplitz(acf)
        w = linalg.eigvalsh(mat)
        assert np.min(w) >= -np.max(w) * len(mat) * np.finfo(float).eps

def test_yule_walker_inv():
    for p in plist:
        acf = gen_ar_acf(p)
        phi = lgp.AR.phi_from_gamma(acf)
        acf2 = lgp.AR.gamma_from_phi(phi)
        acf /= acf[0]
        acf2 /= acf2[0]
        np.testing.assert_allclose(acf2, acf, rtol=1e-12)

def test_yule_walker_inv_extend():
    for p in plist:
        acf = gen_ar_acf(p)
        phi = lgp.AR.phi_from_gamma(acf)
        acf2 = lgp.AR.gamma_from_phi(phi)
        phi3 = np.pad(phi, (0, 1 + p))
        acf3 = lgp.AR.gamma_from_phi(phi3)
        np.testing.assert_allclose(acf3[:len(acf2)], acf2, rtol=1e-14)

def test_yule_walker_inv_evolve():
    for p in plist:
        acf = gen_ar_acf(p)
        phi = lgp._kernels.AR.phi_from_gamma(acf)
        phi2 = np.pad(phi, (0, 1 + p))
        acf2 = lgp.AR.gamma_from_phi(phi2)
        acf3 = lgp.AR.extend_gamma(acf2[:1 + p], phi, 1 + p)
        np.testing.assert_allclose(acf3, acf2, atol=1e-300, rtol=1e-13)

def test_yule_walker_inv_0():
    acf = lgp._kernels.AR.gamma_from_phi(np.empty(0))
    np.testing.assert_allclose(acf, [1], rtol=1e-15)

def test_yule_walker_inv_1():
    bound = 1 - 1e-8
    phi = gen.uniform(-bound, bound)
    acf = lgp.AR.gamma_from_phi([phi])
    acf2 = 1 / ((1 - phi) * (1 + phi)) * phi ** np.arange(2)
    np.testing.assert_allclose(acf, acf2, rtol=1e-15)

def test_phase_degeneracy():
    phases = [
        [[1], [-1]],
        [[1], [1 + 2 * np.pi]],
        [[1], [1 - 2 * np.pi]],
        [[1, -1], [1, 1]],
        [[0.01], [0.01 + 4 * np.pi]],
        [[0.01], [0.01 - 4 * np.pi]],
    ]
    lag = np.arange(100)
    def lnc(ph):
        return 0.1 + 1j * np.array(ph)
    for ph1, ph2 in phases:
        c1 = lgp.AR(slnr=[], lnc=lnc(ph1))(0, lag)
        c2 = lgp.AR(slnr=[], lnc=lnc(ph2))(0, lag)
        np.testing.assert_allclose(c2, c1, atol=0, rtol=1e-13)

def test_real_complex():
    lag = np.arange(100)
    for r in np.logspace(-5, 0, 10):
        for n in range(3):
            for m in range(2):
                add = list(np.arange(1, m + 1) * 0.1 + 1j)
                c1 = lgp.AR(slnr=2 * n * [r], lnc=add)(0, lag)
                c2 = lgp.AR(slnr=[], lnc=add + n * [r])(0, lag)
                np.testing.assert_allclose(c2, c1, atol=0, rtol=1e-7)

def test_ar0():
    lag = np.arange(100)
    acf = np.where(lag, 0, 1)
    params = [
        dict(phi=[], maxlag=lag.size),
        dict(gamma=[1], maxlag=lag.size),
        dict(slnr=[], lnc=[]),
    ]
    for kw in params:
        for norm in range(1):
            c = lgp.AR(**kw, norm=norm)(0, lag)
            np.testing.assert_allclose(c, acf, atol=0, rtol=0)

def test_ar1():
    lag = np.arange(100)
    for phi in np.logspace(-5, -0.001, 10):
        acf = 1 / ((1 - phi) * (1 + phi)) * phi ** lag
        params = [
            dict(phi=[phi], maxlag=lag.size),
            dict(gamma=acf[:2], maxlag=lag.size),
            dict(slnr=[-np.log(phi)], lnc=[]),
        ]
        for kw in params:
            for norm in range(1):
                c = lgp.AR(**kw, norm=norm)(0, lag)
                den = acf[0] if norm else 1
                np.testing.assert_allclose(c, acf / den, atol=1e-300, rtol=1e-12)

# def test_ar2():
#     p = 2
#     vertices = np.stack([
#         lgp.AR.phi_from_roots(l * [-0.] + (p - l) * [0], [])
#         for l in range(p + 1)
#     ])
#     a = np.abs(gen.standard_normal(p + 1))
#     phi = np.sum(a[:, None] * vertices, 0) / np.sum(a)
    # TODO formula for the correlation?

def test_zero_slnr():
    for p in range(1, 10):
        for s in [1, -1]:
            p1 = lgp.AR.phi_from_roots(p * [s * 0.], [])
            p2 = -np.atleast_1d(np.poly(p * [s]))[1:]
            np.testing.assert_equal(p1, p2)
# TODO
# test reflection of extend_gamma
