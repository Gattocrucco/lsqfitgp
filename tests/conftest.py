# lsqfitgp/tests/conftest.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

import pytest
import gvar
import numpy as np

@pytest.fixture(autouse=True)
def clean_gvar_env():
    """ Create a new hidden global covariance matrix for primary gvars, restore
    the previous one during teardown. Otherwise the global covariance matrix
    grows arbitrarily. """
    yield gvar.switch_gvar()
    gvar.restore_gvar()

def nodepath(node):
    """ Take a pytest node, walk up its ancestors collecting all node
    names, return concatenated names """
    names = []
    while node is not None:
        names.append(node.name)
        node = node.parent
    return '::'.join(reversed(names))

@pytest.fixture
def rng(request):
    """ A random generator with a deterministic per-test seed """
    path = nodepath(request.node)
    seed = np.array([path], np.bytes_).view(np.uint8)
    return np.random.default_rng(seed)
