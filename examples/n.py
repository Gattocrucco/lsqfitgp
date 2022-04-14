# lsqfitgp/examples/n.py
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

x = np.linspace(-10, 10, 200)
derivs = [0, 1, 2]

gp = lgp.GP(lgp.ExpQuad(scale=2))
for d in derivs:
    gp.addx(x, d, d)

u = gp.prior()

fig = plt.figure('n')
fig.clf()
ax = fig.subplots(1, 1)

ax.axhline(0)

colors = dict()
for deriv in derivs:
    m = gvar.mean(u[deriv])
    s = gvar.sdev(u[deriv])
    patch = ax.fill_between(x, m - s, m + s, label=f'deriv {deriv}', alpha=0.5)
    colors[deriv] = patch.get_facecolor()[0]
    
for sample in gvar.raniter(u, 1):
    for deriv in derivs:
        ax.plot(x, sample[deriv], '-', color=colors[deriv])

ax.legend(loc='best')
ax.grid(linestyle=':')

fig.show()
