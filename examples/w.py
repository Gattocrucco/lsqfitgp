# lsqfitgp/examples/w.py
#
# Copyright (c) 2020, Giacomo Petrillo
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
kernel = lgp.where(lambda comp: comp == 'short', kshort, klong, dim='comp')
gp = lgp.GP(kernel)

def addcomps(key, time):
    gp.addx(makex(time, 'short'), key + 'short')
    gp.addx(makex(time, 'long'), key + 'long')
    gp.addtransf({key + 'short': 0.3, key + 'long': 1}, key)

addcomps('data', time)
addcomps('pred', time_pred)

print('generate data...')
prior = gp.prior(['data', 'datashort', 'datalong'])
data = next(gvar.raniter(prior))

print('prediction...')
pred = gp.predfromdata({'data': data['data']}, ['pred', 'predshort', 'predlong'])

print('sample posterior...')
mean = gvar.mean(pred)
sdev = gvar.sdev(pred)
samples = list(gvar.raniter(pred, 1))

print('figure...')
fig = plt.figure('w')
fig.clf()
fig.set_size_inches(6, 7)
axs = fig.subplots(3, 1)

for ax, comp in zip(axs, ['', 'short', 'long']):
    key = 'pred' + comp
    
    m = mean[key]
    s = sdev[key]
    ax.fill_between(time_pred, m - s, m + s, alpha=0.3, color='b')
    
    for sample in samples:
        ax.plot(time_pred, sample[key], alpha=0.2, color='b')
    
    ax.plot(time, data['data' + comp], '.k')

fig.tight_layout()
fig.show()
