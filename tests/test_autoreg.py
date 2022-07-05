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

sys.path = ['.'] + sys.path
import lsqfitgp as lgp
import util

def gen_ar_acf(p):
    if p:
        mod = np.random.uniform(1.1, 10, p)
        phase = np.random.uniform(0, 2 * np.pi, p)
        root = mod * np.exp(1j * phase)
        ampl = np.abs(np.random.randn(p))
        tau = np.arange(p + 1)
        return np.sum(ampl * root ** -tau[:, None], axis=1).real
    else:
        return np.abs(np.random.randn(1))

plist = [0, 1, 2, 10, 30, 100]

def test_gen_ar_acf():
    for p in plist:
        acf = gen_ar_acf(p)
        mat = linalg.toeplitz(acf)
        w = linalg.eigvalsh(mat)
        assert np.min(w) >= -np.max(w) * len(mat) * np.finfo(float).eps

def test_yule_walker_inv():
    for p in plist:
        acf = gen_ar_acf(p)
        phi = lgp._kernels.AR.phi_from_gamma(acf)
        acf2 = lgp._kernels.AR.gamma_from_phi(phi)
        acf /= acf[0]
        acf2 /= acf2[0]
        np.testing.assert_allclose(acf2, acf, rtol=1e-12)

def test_yule_walker_inv_extend():
    for p in plist:
        acf = gen_ar_acf(p)
        phi = lgp._kernels.AR.phi_from_gamma(acf)
        acf2 = lgp._kernels.AR.gamma_from_phi(phi)
        phi3 = np.pad(phi, (0, 1 + p))
        acf3 = lgp._kernels.AR.gamma_from_phi(phi3)
        np.testing.assert_allclose(acf3[:len(acf2)], acf2, rtol=1e-14)

def test_yule_walker_inv_evolve():
    for p in plist:
        acf = gen_ar_acf(p)
        phi = lgp._kernels.AR.phi_from_gamma(acf)
        phi2 = np.pad(phi, (0, 1 + p))
        acf2 = lgp._kernels.AR.gamma_from_phi(phi2)
        acf3 = lgp._kernels.AR.extend_gamma(acf2[:1 + p], phi, 1 + p)
        np.testing.assert_allclose(acf3, acf2, rtol=1e-13)

def test_yule_walker_inv_0():
    acf = lgp._kernels.AR.gamma_from_phi(np.empty(0))
    np.testing.assert_allclose(acf, [1], rtol=1e-15)

def test_yule_walker_inv_1():
    bound = 1 - 1e-8
    phi = np.random.uniform(-bound, bound)
    acf = lgp._kernels.AR.gamma_from_phi([phi])
    acf2 = 1 / ((1 - phi) * (1 + phi)) * phi ** np.arange(2)
    np.testing.assert_allclose(acf, acf2, rtol=1e-15)
