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

"""

                            EXAMPLE X.

    Where the derivatives of an interesting correlation function
    are put to harsh a trial.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots(num='x', clear=True)

x = np.linspace(0, 10, 1000)

eps = 1e-8
args = [
    # Q, kw
    (eps, {}),
    (0.5 - eps, {}),
    (0.5 + eps, dict(linestyle='--')),
    (1 - eps, {}),
    (1 + eps, dict(linestyle='--')),
    (2, {})
]
for Q, kw in args:
    y = lgp.Harmonic(Q=Q).diff(1, 1)(0, x)
    ax.plot(x, y, label=f'Q={Q}', **kw)

ax.legend(loc='best')
fig.show()
