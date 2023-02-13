# lsqfitgp/examples/q.py
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

                            EXAMPLE Q.

    Where we extend a pattern of waves in space, but do not dare
    look too far from our data.

"""

import lsqfitgp as lgp
from lsqfitgp import _linalg
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import gvar

xdata1d = np.linspace(-4, 4, 10)
xpred1d = np.linspace(-10, 10, 50)

def makexy(x1d, y1d):
    xy = np.empty((len(x1d), len(y1d)), dtype=[
        ('x', float),
        ('y', float)
    ])
    x, y = np.meshgrid(x1d, y1d)
    xy['x'] = x
    xy['y'] = y
    return xy
    
xdata = makexy(xdata1d, xdata1d)
xpred = makexy(xpred1d, xpred1d)
y = np.cos(xdata['x']) * np.cos(xdata['y'])

gp = lgp.GP(lgp.ExpQuad(scale=3, dim='x') * lgp.ExpQuad(scale=3, dim='y'))
gp.addx(xdata.reshape(-1), 'pere')
gp.addx(xpred.reshape(-1), 'banane')

print('fit...')
m, cov = gp.predfromdata({'pere': y.reshape(-1)}, 'banane', raw=True)

print('samples...')
sample = m + gp.decompose(cov, solver='chol', epsrel=1e-4).correlate(np.random.randn(len(m)))
sample = sample.reshape(xpred.shape)

print('plot...')
fig, ax = plt.subplots(num='q', clear=True, subplot_kw=dict(projection='3d', computed_zorder=False))

ax.scatter(xdata['x'].reshape(-1), xdata['y'].reshape(-1), y.reshape(-1), color='black', zorder=10)
ax.plot_surface(xpred['x'], xpred['y'], sample, alpha=0.9, cmap='viridis')
ax.view_init(elev=70, azim=110)

fig.show()
