# lsqfitgp/examples/w.py
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

                            EXAMPLE W.

    Where, with limited success, we recover the identity of two
    mixed functions knowing their speed of variation.

"""

import numpy as np
from matplotlib import pyplot as plt
import gvar

import lsqfitgp as lgp

time = np.arange(30)
time_pred = np.linspace(-30, 60, 200)

def makex(time, comp):
    x = np.empty(len(time), dtype=[
        ('time', float),
        ('comp', 'U8')
    ])
    x['time'] = time
    x['comp'] = comp
    return x

kshort = lgp.ExpQuad(scale=1, dim='time')
klong = lgp.ExpQuad(scale=10, dim='time')
kernel = lgp.where(lambda comp: comp['dim'] == 'short', kshort, klong)
gp = lgp.GP(kernel)

def addcomps(gp, key, time):
    return (gp
        .addx(makex(time, 'short'), key + 'short')
        .addx(makex(time, 'long'), key + 'long')
        .addtransf({key + 'short': 0.3, key + 'long': 1}, key)
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
fig, axs = plt.subplots(3, 1, num='w', clear=True, figsize=[6, 7], layout='constrained')

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
