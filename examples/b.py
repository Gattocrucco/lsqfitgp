# lsqfitgp/examples/b.py
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
gp.addx(xpred, 'pred')
gp.addx(xpred, 'deriv', 1)

print('fit...')
u = gp.pred({'data': y}, ['pred', 'deriv'], fromdata=True)

print('figure...')
fig = plt.figure('b')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for label in u:
    m = gvar.mean(u[label])
    s = gvar.sdev(u[label])
    patch = ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)
    colors[label] = patch.get_facecolor()[0]
    
print('samples...')
for sample in gvar.raniter(u, 1):
    for label in u:
        ax.plot(xpred, sample[label], '-', color=colors[label])
ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
