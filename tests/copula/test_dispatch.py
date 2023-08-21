# lsqfitgp/tests/test_dispatch.py
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

""" Test numpy array protocols on Distr """

import operator

from pytest import mark

from lsqfitgp import copula

from .. import util

@mark.parametrize('op', [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.pow,
    operator.mod,    
])
@mark.parametrize('number', [False, True])
def test_binary(op, number, rng):
    x = copula.beta(2, 3)
    y = 1.3 if number else copula.gamma(1, 1)    
    z = op(x, y)
    
    def invfcn(n):
        xval = x.partial_invfcn(n[..., 0])
        if number:
            yval = y
        else:
            yval = y.partial_invfcn(n[..., 1])
        return op(xval, yval)

    n = rng.uniform(size=z.in_shape)
    out1 = z.partial_invfcn(n)
    out2 = invfcn(n if z.in_shape else n[..., None])
    util.assert_equal(out1, out2)

@mark.parametrize('op', [
    operator.abs,
    operator.neg,
    operator.pos,
])
def test_unary(op, rng):
    x = copula.beta(2, 3)    
    z = op(x)
    
    def invfcn(n):
        xval = x.partial_invfcn(n)
        return op(xval)

    n = rng.uniform(size=z.in_shape)
    out1 = z.partial_invfcn(n)
    out2 = invfcn(n)
    util.assert_equal(out1, out2)

def test_repr():
    x = copula.beta(1, 1) + copula.gamma(1, 1)
    assert repr(x) == 'add(beta(1, 1), gamma(1, 1))'
