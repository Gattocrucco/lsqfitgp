# lsqfitgp/examples/t.py
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

                            EXAMPLE T.

    Where we pretend to discover that two series of events
    were in fact one the delayed and imperfect copy of the
    other.

"""

import time as systime

import lsqfitgp as lgp
from matplotlib import pyplot as plt
from jax import numpy as jnp
import numpy as np
import gvar

time = np.arange(21)
x = np.empty((2, len(time)), dtype=[
    ('time', float),
    ('label', int)
])
x['time'][0] = time
delay = 20
x['time'][1] = time - delay
x['label'][0] = 0
x['label'][1] = 1
label_names = ['gatti_comprati', 'gatti_morti']

function = lambda x: np.exp(-1/2 * ((x - 10) / 5)**2)
data_error = 0.05
data_mean = function(x['time']) + data_error * np.random.randn(*x.shape)
data_mean[1] += 0.02 * time
data = gvar.gvar(data_mean, np.full_like(data_mean, data_error))

x = lgp.StructuredArray(x)
def makegp(params):
    kernel = lgp.Cauchy(scale=params['time_scale'], dim='time', beta=2)
    kernel *= lgp.ExpQuad(scale=params['label_scale'], dim='label')
    gp = lgp.GP(kernel)
    x['time'] = jnp.array([time, time - params['delay']])
    gp.addx(x, 'A')
    return gp

start = systime.time()
hyperprior = gvar.BufferDict({
    'log(time_scale)': gvar.log(gvar.gvar(10, 10)),
    'log(label_scale)': gvar.log(gvar.gvar(10, 10)),
    'delay': gvar.gvar(10, 20)
})
params = lgp.empbayes_fit(hyperprior, makegp, {'A': data}, raises=False, jit=True).p
end = systime.time()

print('minimization time = {:.2g} sec'.format(end - start))
print('time scale = {}'.format(params['time_scale']))
corr = lgp.ExpQuad(scale=gvar.mean(params['label_scale']))(0, 1)
print('correlation = {:.3g} (equiv. scale = {})'.format(corr, params['label_scale']))
print('delay = {}'.format(params['delay']))

gp = makegp(gvar.mean(params))

xpred = np.empty((2, 100), dtype=x.dtype)
time_pred = np.linspace(np.min(time), np.max(time) + 1.5 * (np.max(time) - np.min(time)), xpred.shape[1])
xpred['time'][0] = time_pred
xpred['time'][1] = time_pred - gvar.mean(params['delay'])
xpred['label'][0] = 0
xpred['label'][1] = 1
gp.addx(xpred, 'B')

pred = gp.predfromdata({'A': data}, 'B')

fig, ax = plt.subplots(num='t', clear=True)

colors = []
for i in range(2):
    m = gvar.mean(pred[i])
    s = gvar.sdev(pred[i])
    polys = ax.fill_between(time_pred, m - s, m + s, alpha=0.5, label=label_names[i])
    colors.append(polys.get_facecolor()[0])

for sample in gvar.raniter(pred, 3):
    for i in range(2):
        ax.plot(time_pred, sample[i], color=colors[i])

for i in range(2):
    ax.errorbar(time, gvar.mean(data[i]), yerr=gvar.sdev(data[i]), fmt='.', color=colors[i], alpha=1)

ax.legend(loc='best')
ax.set_xlabel('time')

fig.show()
