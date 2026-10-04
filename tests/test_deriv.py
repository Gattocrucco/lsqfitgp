# lsqfitgp/tests/test_deriv.py
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

"""Test the `Deriv` class."""

import pytest

import lsqfitgp as lgp


def test_manyargs():
    """Check that `Deriv` raises `ValueError` with more than one argument."""
    with pytest.raises(ValueError, match=r'^2$'):
        lgp.Deriv(1, 2)


def test_alienargs():
    """Check that `Deriv` raises `TypeError` on a sequence with an invalid item."""
    with pytest.raises(TypeError):
        lgp.Deriv((None,))


def test_manyintegers():
    """Check that `Deriv` raises `ValueError` on consecutive integers."""
    with pytest.raises(ValueError, match='consecutive integers'):
        lgp.Deriv((1, 2))


def test_alienarg():
    """Check that `Deriv` raises `TypeError` on an argument of invalid type."""
    with pytest.raises(TypeError):
        lgp.Deriv(object)


def test_orphan():
    """Check that `Deriv` raises `ValueError` on an integer not followed by a name."""
    with pytest.raises(ValueError, match='dangling derivative order'):
        lgp.Deriv((1,))
    with pytest.raises(ValueError, match='dangling derivative order'):
        lgp.Deriv(('ciao', 1))


def test_length():
    """Check that the length of a `Deriv` is the number of distinct variables."""
    assert len(lgp.Deriv([1, 'ciao', 2, 'pippo'])) == 2


def test_compare():
    """Check that a `Deriv` does not compare equal to a string."""
    assert lgp.Deriv() != 'cippa'


def test_repr():
    """Check that the `repr` of an empty `Deriv` is `{}`."""
    assert repr(lgp.Deriv()) == '{}'
