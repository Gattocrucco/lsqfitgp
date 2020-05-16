# lsqfitgp/tests/test_GP.py
#
# Copyright (c) 2020, Giacomo Petrillo
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

sys.path.insert(0, '.')
import lsqfitgp as lgp

def test_prior_raw_shape():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(np.arange(20).reshape(2, 10), 'x')
    cov = gp.prior(raw=True)
    assert cov['x', 'x'].shape == (2, 10, 2, 10)
    
    cov = gp.prior('x', raw=True)
    assert cov.shape == (2, 10, 2, 10)
