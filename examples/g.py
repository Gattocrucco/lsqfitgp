# lsqfitgp/examples/g.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 200)
y = np.sin(xdata)

print('make GP...')
gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata, 'data')
gp.addx(xpred, 'pred', 0)
gp.addx(xpred, 'predderiv', 1)

print('fit...')
umean, ucov = gp.predfromdata({'data': y}, ['pred', 'predderiv'], raw=True)
ualt = gp.predfromdata({'data': y}, ['pred', 'predderiv'])

print('figure...')
fig = plt.figure('g')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for label in umean:
    m = umean[label]
    s = np.sqrt(np.diag(ucov[label, label]))
    patch = ax.fill_between(xpred, m - s, m + s, label=label + ' (raw)', alpha=0.5)
    colors[label] = patch.get_facecolor()[0]
    
for label in ualt:
    m = gvar.mean(ualt[label])
    s = gvar.sdev(ualt[label])
    ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)

print('samples...')
for label in umean:
    m = umean[label]
    cov = ucov[label, label]
    samples = np.random.multivariate_normal(m, cov, size=10)
    ax.plot(xpred, samples.T, '-', color=colors[label])

ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
