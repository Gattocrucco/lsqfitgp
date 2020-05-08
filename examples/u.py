# lsqfitgp/examples/u.py
#
# Copyright (c) Giacomo Petrillo 2020
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

data_deriv = 1

time = np.linspace(-5, 5, 10)
x = np.empty(len(time), dtype=[
    ('time', float),
    ('label', int)
])
x['time'] = time
x['label'] = data_deriv

data_error = 0.05
data_mean = np.cos(time)
data_mean += data_error * np.random.randn(*data_mean.shape)
data = gvar.gvar(data_mean, np.full_like(data_mean, data_error))

label_scale = 5
corr = lgp.ExpQuad(scale=label_scale)(0, 1)
print(f'corr = {corr:.3g}')

def makegp(params):
    kernel_time = lgp.ExpQuad(scale=params['time_scale'], dim='time')
    kernel_label = lgp.ExpQuad(scale=label_scale, dim='label')
    gp = lgp.GP(kernel_time * kernel_label)
    gp.addx(x, 'data', deriv=(data_deriv, 'time'))
    gp.addx(np.array([(0, 0)], dtype=x.dtype), 'fixed_point')
    return gp

prior = {
    'log(time_scale)': gvar.log(gvar.gvar(3, 2))
}
datadict = {'data': data, 'fixed_point': [gvar.gvar(0, 1e2)]}
params = lgp.empbayes_fit(prior, makegp, datadict, raises=False)
print('time_scale:', params['time_scale'])
gp = makegp(gvar.mean(params))

time_pred = np.linspace(-10, 10, 100)
xpred = np.empty((2, len(time_pred)), dtype=x.dtype)
xpred['time'] = time_pred
xpred['label'][0] = 0
xpred['label'][1] = 1
gp.addx(xpred[0], 0)
gp.addx(xpred[1], 1, deriv=(1, 'time'))

pred = gp.predfromdata(datadict, [0, 1])

fig = plt.figure('u')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for deriv in pred:
    m = gvar.mean(pred[deriv])
    s = gvar.sdev(pred[deriv])
    polys = ax.fill_between(time_pred, m - s, m + s, alpha=0.5, label=f'deriv {deriv}')
    colors[deriv] = polys.get_facecolor()[0]

for sample in gvar.raniter(pred, 3):
    for deriv in pred:
        ax.plot(time_pred, sample[deriv], color=colors[deriv])

ax.errorbar(time, gvar.mean(data), yerr=gvar.sdev(data), fmt='.', color=colors[data_deriv], alpha=1, label='data')

ax.legend(loc='best')
ax.set_xlabel('time')

fig.show()
