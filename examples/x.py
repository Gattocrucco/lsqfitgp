# lsqfitgp/examples/x.py
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

fig = plt.figure('x')
fig.clf()
ax = fig.subplots(1, 1)

x = np.linspace(0, 10, 1000)

eps = 1e-8
for Q in eps, 0.5 - eps, 0.5 + eps, 1 - eps, 1, 1 + eps, 2:
    y = lgp.Harmonic(Q=Q).diff(1, 1)(0, x)
    ax.plot(x, y, label='Q={}'.format(Q))

ax.legend(loc='best')
fig.show()
