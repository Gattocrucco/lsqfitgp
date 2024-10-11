# lsqfitgp/copula/__init__.py
#
# Copyright (c) 2023, 2024, Giacomo Petrillo
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

""" Reparametrize probability distributions as Normal """

from ._base import DistrBase
from ._distr import Distr, distribution
from ._copula import Copula
from ._makedict import makedict
from ._copulas import (
    beta,
    dirichlet,
    gamma,
    loggamma,
    invgamma,
    halfcauchy,
    halfnorm,
    uniform,
    lognorm,
)

# TODO I could try to drop BufferDict altogether. It adds more complexity than
# necessary: won't keep track of dependencies between keys, structure fixed to a
# dictionary, and global unique transformation names.
#
# I need something analogous to carry around both the distribution definition
# and specific values. Copula.concretize(x) -> ConcreteCopula object. Method
# .values() returns pytree of transformed values (container-copied),
# .input_values() the flat input array (readonly view). Prints as:
# Copula({
#     'x': beta(1, 2),
#     'y': gamma(1, <x>),
# })
# ConcreteCopula({
#     'x': 0.1234,
#     'y': 0.1314,
# })
# But there is a description method to list the original values:
# ConcreteCopula({
#     'x': 0.1234 <- 0.1442,
#     'y': 0.1314 <- 0.4124, <x>
# })
# or to do comparisons:
# ConcreteCopula({
#     'x': 0.1234 | 0.12344 <- 0.1442 | 0.7777,
#     'y': 0.1314 | 0.4144 <- 0.4124 | 0.1341, <x>
# })
#
# Make empbayes_fit take Copula_like as input, and output ConcreteCopula for
# the posterior. Attribute .p would be ConcreteCopula.values(), new attribute
# post would contain everything.
#
# I could maintain the BufferDict functionality, listed under a bottom section
# in the reference.
