# lsqfitgp/examples/fourier.py
#
# Copyright (c) 2022, Giacomo Petrillo
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

"""Constrain the values of Fourier series coefficients"""

gp = (lgp
    .GP(lgp.Zeta(nu=2.5), checkpos=False) # TODO is this checkpos necessary
    .defkerneltransf('F', 'fourier', True, lgp.GP.DefaultProcess)
)
x = np.linspace(0, 1, 100)
gp = (gp
    .addx(x, 'x')
    .addx(1, 's1', proc='F')
    .addx(2, 'c1', proc='F')
)

comb = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]

fig, ax = plt.subplots(num='fourier', clear=True)

for s, c in comb:
    y = gp.predfromdata(dict(s1=s, c1=c), 'x')
    m = gvar.mean(y)
    u = gvar.sdev(y)
    pc = ax.fill_between(x, m - u, m + u, alpha=0.5, label=f's{s}c{c}')
    color = pc.get_facecolor()
    for sample in gvar.raniter(y, 3):
        ax.plot(x, sample, color=color)

ax.legend()

fig.show()
