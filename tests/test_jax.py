# lsqfitgp/tests/test_jax.py
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

import jax
import numpy as np

from lsqfitgp import _patch_jax
import util

def test_sinc():
    x = np.linspace(-0.1, 0.1, 1000)
    s1 = np.sinc(x)
    s2 = _patch_jax.sinc(x)
    np.testing.assert_allclose(s2, s1, atol=1e-15, rtol=1e-15)
