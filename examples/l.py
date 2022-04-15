# lsqfitgp/examples/l.py
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

                            EXAMPLE L.

    Where two formulas give the same results and so math triumphs
    once again.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots(num='l', clear=True)

x = np.linspace(0, 10, 1000)

for dv in 1, 3, 5:
    v = dv / 2
    y1 = lgp.Matern(nu=v)([0.], x)
    y2 = eval(f'lgp.Matern{dv}2')()([0.], x)
    line, = ax.plot(x, y1, label=f'{dv}/2', alpha=0.5)
    ax.plot(x, y2, '--', color=line.get_color(), alpha=0.5)

ax.legend(loc='best')
ax.set_yscale('log')
fig.show()
