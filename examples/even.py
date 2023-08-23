# lsqfitgp/examples/even.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

Split a function into even and odd parts.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

gp = lgp.GP(lgp.ExpQuad())
gp.defproclintransf('even', lambda f: lambda x: (f(x) + f(-x)) / 2, [gp.DefaultProcess])
gp.defproclintransf('odd', lambda f: lambda x: (f(x) - f(-x)) / 2, [gp.DefaultProcess])

x1, y1 = 1, 1
gp.addx(x1, 'even', proc='even')

x2, y2 = 1, -1
gp.addx(x2, 'odd', proc='odd')

xplot = np.linspace(-5, 5, 300)
gp.addx(xplot, 'function')
gp.addx(xplot, 'even part', proc='odd')
gp.addx(xplot, 'odd part', proc='even')

y = gp.predfromdata({'even': y1, 'odd': y2})

fig, ax = plt.subplots(num='even', clear=True)

labels = [label for label in y if len(label) > 4]
for j, sample in enumerate(gvar.raniter(y, 2, eps=1e-16)):
    for i, label in enumerate(labels):
        ax.plot(xplot, sample[label], color=f'C{i}', alpha=0.8, label=label if j == 0 else None)
ax.plot([x1, x2], [y1, y2], '.k')
ax.legend()
    
fig.show()
