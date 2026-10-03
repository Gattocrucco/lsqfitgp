# lsqfitgp/tests/copula/test_makedict.py
#
# Copyright (c) 2023, 2026, Giacomo Petrillo
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

"""Test `copula.makedict`."""

import pytest

from lsqfitgp import copula


def test_dependencies():
    """Check that `makedict` forbids interdependencies between the keys."""
    x = copula.beta(1, 1)
    y = copula.beta(1, x)
    with pytest.raises(ValueError, match='cross-key occurrences'):
        copula.makedict({'x': x, 'y': y})
    xy = copula.Copula({'x': x, 'y': y})
    copula.makedict({'xy': xy})
    with pytest.raises(ValueError, match='cross-key occurrences'):
        copula.makedict({'xy': xy, 'x': x})
