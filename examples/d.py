# lsqfitgp/examples/d.py
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

"""

                            EXAMPLE D.

    Where the primitive of our function takes some freedom to move
    up and down.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(-5, 5, 10)
xpred = np.linspace(-15, 25, 200)
y = np.cos(xdata)

print('make GP...')
gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata, 'data', 1)
gp.addx(xpred, 'integral')
gp.addx(xpred, 'pred', 1)

print('fit...')
u = gp.predfromdata({'data': y}, ['integral', 'pred'])

print('figure...')
fig, ax = plt.subplots(num='d', clear=True)

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
ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
