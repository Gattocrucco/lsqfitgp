# lsqfitgp/tests/copula/__init__.py
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

# TODO add a function copula that somehow allows to define dictionary without
# repeating twice each distribution name, and is also compatible with any normal
# usage => takes in a dict, scans it, when key is string, value is tuple, and
# first element of tuple is callable, then call to get value, rest is args,
# convert the result to BufferDict. Exaple:
#
# lgp.copula.copula({
#     'HC(a)': (lgp.copula.halfcauchy, 1.0),
# })
#
# or maybe I could make it even more automatic, the end user probably does not
# even want to deal with the transformation names or copula objects:
#
# lgp.copula.copula({
#     'a': ('halfcauchy', 1.0),
# }, prefix='ciao_')
#
# -> BufferDict({
#     'ciao_halfcauchy_1.0(a)': lgp.copula.halfcauchy('a', 1.0),
# })
#
# The default prefix would be something like '__copula_'.
