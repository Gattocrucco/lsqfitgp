# lsqfitgp/examples/y.py
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

                            EXAMPLE Y.

    Where a Zeta kernel forces some random samples to have
    zero mean.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 1, 10)
xpred = np.linspace(0, 1, 500)
y = np.ones_like(xdata)

gp = (lgp.
    GP(lgp.Zeta(nu=1.5))
    .addx(xdata, 'pere')
    .addx(xpred, 'banane')
)

u = gp.predfromdata({'pere': y}, 'banane')
m = gvar.mean(u)
s = gvar.sdev(u)
cov = gvar.evalcov(u)

fig, ax = plt.subplots(num='y', clear=True)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.plot(xdata, y, 'kx', label='data')
ax.legend(loc='best')

fig.show()
