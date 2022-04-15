# lsqfitgp/examples/f.py
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

                            EXAMPLE F.

    Where apparently in these times it is not anymore possible to
    know exactly where one gentleman's function will pass.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 300)
y = np.sin(xdata)
yerr = 0.1

gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata, 'pere')
gp.addx(xpred, 'banane')

uy = gvar.gvar(y + yerr * np.random.randn(len(y)), yerr * np.ones_like(y))
u = gp.predfromdata({'pere': uy}, 'banane')
m = gvar.mean(u)
s = gvar.sdev(u)
cov = gvar.evalcov(u)

fig, ax = plt.subplots(num='f', clear=True)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.errorbar(xdata, gvar.mean(uy), yerr=gvar.sdev(uy), fmt='k.', label='data')
ax.legend(loc='best')

fig.show()
