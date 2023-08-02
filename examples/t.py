# lsqfitgp/examples/t.py
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

                            EXAMPLE T.

    Where we pretend to discover that two series of events
    were in fact one the delayed and imperfect copy of the
    other.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
from jax import numpy as jnp
import numpy as np
import gvar

time = np.arange(21)
delay = 20

def makex(time, delay):
    x = np.empty((2, time.size), dtype=[
        ('time', float),
        ('label', int)
    ])
    x['time'][0] = time
    x['time'][1] = time - delay
    x['label'][0] = 0
    x['label'][1] = 1
    return lgp.StructuredArray(x)

x = makex(time, delay)

label_names = ['gatti_comprati', 'gatti_morti']

function = lambda x: np.exp(-1/2 * ((x - 10) / 5)**2)
data_mean = function(x['time'])
data_mean[1] += 0.02 * time
data = gvar.gvar(data_mean, np.full_like(data_mean, 0.05))
data = gvar.make_fake_data(data)

def makegp(params):
    kernel = lgp.Cauchy(scale=params['time_scale'], dim='time', beta=2)
    kernel *= lgp.ExpQuad(scale=params['label_scale'], dim='label')
    gp = lgp.GP(kernel)
    xmod = x.at['time'].set(jnp.array([time, time - params['delay']]))
    gp.addx(xmod, 'A')
    return gp

hyperprior = gvar.BufferDict({
    'log(time_scale)': gvar.log(gvar.gvar(10, 10)),
    'log(label_scale)': gvar.log(gvar.gvar(5, 5)),
    'delay': gvar.gvar(5, 20)
})

fit = lgp.empbayes_fit(hyperprior, makegp, {'A': data}, raises=False, verbosity=2)

print(f'time scale = {fit.p["time_scale"]}')
corr = lgp.ExpQuad(scale=gvar.mean(fit.p['label_scale']))(0, 1)
print(f'correlation = {corr:.3g} (equiv. scale = {fit.p["label_scale"]})')
print(f'delay = {fit.p["delay"]} (true = {delay})')

fig, ax = plt.subplots(num='t', clear=True)

time_pred = np.linspace(np.min(time), np.max(time) + 1.5 * (np.max(time) - np.min(time)), 100)

for style, params_sample in zip(['-', '--'], gvar.raniter(fit.p, 2)):
    
    gp = makegp(params_sample)
    xpred = makex(time_pred, params_sample['delay'])
    gp.addx(xpred, 'B')
    pred = gp.predfromdata({'A': data}, 'B')

    for sample in gvar.raniter(pred, 1):
        for i in range(2):
            label = f'{label_names[i]}, delay={params_sample["delay"]:.1f}'
            ax.plot(time_pred, sample[i], color=f'C{i}', alpha=0.5, label=label, linestyle=style)

for i in range(2):
    ax.errorbar(time, gvar.mean(data[i]), yerr=gvar.sdev(data[i]), fmt='.', color=f'C{i}')

ax.legend(loc='best')
ax.set_xlabel('time')

fig.show()
