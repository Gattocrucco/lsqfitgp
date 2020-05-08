# lsqfitgp/tests/test_array.py
#
# Copyright (c) Giacomo Petrillo 2020
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

sys.path.insert(0, '.')
import lsqfitgp as lgp

import numpy as np

def test_ellipsis():
    x = np.empty((2, 3), dtype=[('a', float, 4)])
    y = lgp.StructuredArray(x)
    z = y[..., 0]
    assert z.shape == y.shape[:1]
    assert z['a'].shape == y.shape[:1] + x.dtype.fields['a'][0].shape
