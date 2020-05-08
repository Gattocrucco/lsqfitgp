# lsqfitgp/examples/e.py
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

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(-5, 5, 10)
xpred = np.linspace(-15, 25, 200)
y = np.sin(xdata)
y[1::2] = np.cos(xdata[1::2])

print('make GP...')
gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata[0::2], 'data', 0)
gp.addx(xdata[1::2], 'dataderiv', 1)
gp.addx(xpred, 'pred', 0)
gp.addx(xpred, 'predderiv', 1)

print('fit...')
u = gp.predfromdata({'data': y[0::2], 'dataderiv': y[1::2]}, ['pred', 'predderiv'])

print('figure...')
fig = plt.figure('e')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for label in u:
    m = gvar.mean(u[label])
    s = gvar.sdev(u[label])
    patch = ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)
    colors[label] = patch.get_facecolor()[0]
    
print('samples...')
for sample in gvar.raniter(u, 30):
    for label in u:
        ax.plot(xpred, sample[label], '-', color=colors[label])

for deriv, marker in (0, '+'), (1, 'x'):
    ax.plot(xdata[deriv::2], y[deriv::2], f'k{marker}', label=f'data deriv {deriv}')
ax.legend(loc='best')

fig.show()
