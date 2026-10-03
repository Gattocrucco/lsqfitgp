# lsqfitgp/examples/doubleint.py
#
# Copyright (c) 2022, 2023, 2024, 2026, Giacomo Petrillo
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

"""Test of double integral constraint."""

import gvar
import numpy as np
from matplotlib import pyplot as plt

import lsqfitgp as lgp

np.random.seed(20220417)

#### DEFINE MODEL ####
# h ~ GP
# f = h''
# int_0^1 dx f(x) = [h'(x)]_0^1 = h'(1) - h'(0)
# int_0^1 dx x f(x) = [xh'(x) - h(x)]_0^1 = h'(1) - h(1) + h(0)

gp = lgp.GP(lgp.ExpQuad())

x = np.linspace(0, 1, 10)
gp = (
    gp.addx(x, 'data', deriv=2)
    .addx([0, 1], 'xinteg', deriv=1)
    .addtransf({'xinteg': [-1, 1]}, 'integ')
    .addx([0, 1], 'xintegx0')
    .addx(1, 'xintegx1', deriv=1)
    .addtransf({'xintegx1': 1, 'xintegx0': [1, -1]}, 'integx')
)

#### GENERATE FAKE DATA ####

prior = gp.predfromdata({'integ': 1, 'integx': 1}, ['data', 'integ', 'integx'])
priorsample = gvar.sample(prior)

datamean = priorsample['data']
dataerr = np.full_like(datamean, 1)
datamean = datamean + dataerr * np.random.randn(*dataerr.shape)
data = gvar.gvar(datamean, dataerr)

# check the integral is one with trapezoid rule
print('prior:')
y = priorsample['data']
checksum = np.sum((y[1:] + y[:-1]) / 2 * np.diff(x))
print('sum_i int dx   f_i(x) =', checksum)
checksum = np.sum(((y * x)[1:] + (y * x)[:-1]) / 2 * np.diff(x))
print('sum_i int dx x f_i(x) =', checksum)

#### FIT ####

pred = gp.predfromdata(
    {'integ': 1, 'integx': 1, 'data': data}, ['data', 'integ', 'integx']
)

# check the integral is one with trapezoid rule
print('posterior:')
y = pred['data']
checksum = np.sum((y[1:] + y[:-1]) / 2 * np.diff(x))
print('sum_i int dx   f_i(x) =', checksum)
checksum = np.sum(((y * x)[1:] + (y * x)[:-1]) / 2 * np.diff(x))
print('sum_i int dx x f_i(x) =', checksum)

#### PLOT RESULTS ####

fig, ax = plt.subplots(num='doubleint', clear=True)

y = pred['data']
m = gvar.mean(y)
s = gvar.sdev(y)
ax.fill_between(x, m - s, m + s, alpha=0.6)

y = priorsample['data']
ax.plot(x, y)

y = pred['data'] * x
m = gvar.mean(y)
s = gvar.sdev(y)
ax.fill_between(x, m - s, m + s, alpha=0.6)

y = priorsample['data'] * x
ax.plot(x, y)

ax.errorbar(x, datamean, dataerr, color='black', linestyle='', capsize=2)

fig.show()
