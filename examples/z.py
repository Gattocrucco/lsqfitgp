# lsqfitgp/examples/z.py
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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

                            EXAMPLE Z.

    Where we sail in an infinite dimensional space to sum two
    numbers.

"""

import numpy as np
from matplotlib import pyplot as plt
import gvar

import lsqfitgp as lgp

gp = (lgp.GP()
    .defproc('short', lgp.ExpQuad(scale= 1))
    .defproc('long', lgp.ExpQuad(scale=10))
    .deftransf('sum', {'short': 0.3, 'long': 1})
)

time = np.arange(30)
time_pred = np.linspace(-30, 60, 200)

def addcomps(gp, key, time):
    return (gp
        .addx(time, key + 'short', proc='short')
        .addx(time, key + 'long' , proc='long' )
        .addx(time, key          , proc='sum'  )
    )

gp = addcomps(gp, 'data', time)
gp = addcomps(gp, 'pred', time_pred)

print('generate data...')
prior = gp.prior(['data', 'datashort', 'datalong'])
data = gvar.sample(prior)

print('prediction...')
pred = gp.predfromdata({'data': data['data']}, ['pred', 'predshort', 'predlong'])

print('sample posterior...')
mean = gvar.mean(pred)
sdev = gvar.sdev(pred)
samples = list(gvar.raniter(pred, 1))

print('figure...')
fig, axs = plt.subplots(3, 1, num='z', clear=True, figsize=[6, 7], layout='constrained')

for ax, comp in zip(axs, ['', 'short', 'long']):
    key = 'pred' + comp
    
    m = mean[key]
    s = sdev[key]
    ax.fill_between(time_pred, m - s, m + s, alpha=0.3, color='b')
    
    for sample in samples:
        ax.plot(time_pred, sample[key], alpha=0.2, color='b')
    
    ax.plot(time, data['data' + comp], '.k')

axs[0].set_ylabel('A + B')
axs[1].set_ylabel('A')
axs[2].set_ylabel('B')

fig.show()
