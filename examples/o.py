# lsqfitgp/examples/o.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

                            EXAMPLE O.

    Where mister N's enquiry is repeated by different means.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

x = np.linspace(-10, 10, 200)
derivs = [0, 1, 2]

gp = lgp.GP(lgp.ExpQuad(scale=2))
for d in derivs:
    gp.addx(x, d, deriv=d)

cov = gp.prior(raw=True)

fig, ax = plt.subplots(num='o', clear=True)

colors = dict()
m = np.zeros(len(x))
for deriv in derivs:
    s = np.sqrt(np.diag(cov[deriv, deriv]))
    patch = ax.fill_between(x, m - s, m + s, label=f'deriv {deriv}', alpha=0.5)
    colors[deriv] = patch.get_facecolor()[0]
    
for deriv in derivs:
    samples = np.random.multivariate_normal(m, cov[deriv, deriv], size=5)
    ax.plot(x, samples.T, '-', color=colors[deriv])

ax.legend(loc='best')
ax.grid(linestyle=':')

fig.show()
