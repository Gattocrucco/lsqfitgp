# lsqfitgp/examples/p.py
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

                            EXAMPLE P.

    Where an overwhelming amount of datapoints is revelead to be
    nothing more than a pile of ten eigenfunctions.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 1000)
xpred = np.linspace(-15, 25, 300)
y = np.sin(xdata)

gp = lgp.GP(lgp.ExpQuad(scale=3), solver='lowrank', rank=10, checkpos=False)
gp.addx(xdata, 'pere')
gp.addx(xpred, 'banane')

print('fit...')
m, cov = gp.predfromdata({'pere': y}, 'banane', raw=True)
# u = gp.predfromdata({'pere': y}, 'banane')
# print('extract cov...')
# m = gvar.mean(u)
# cov = gvar.evalcov(u)

s = np.sqrt(np.diag(cov))

print('plot...')
fig, ax = plt.subplots(num='p', clear=True)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
