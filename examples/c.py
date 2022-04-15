# lsqfitgp/examples/c.py
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

                            EXAMPLE C.

    Where a nonlinear transformation hides the true height of some
    crosses.

"""

import lsqfitgp as lgp
import lsqfit
from matplotlib import pyplot as plt
import numpy as np
import gvar

plot_simulated_lines = True

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 300)

gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata, 'data')
gp.addx(xpred, 'pred')

true_par = dict(
    phi=np.sin(xdata),
    y0=10
)

def fcn(data_or_pred, p):
    if data_or_pred == 'data':
        phi = p['phi']
    elif data_or_pred == 'pred':
        phi = gp.predfromfit({'data': p['phi']}, 'pred')
    else:
        raise KeyError(data_or_pred)
    
    return gvar.tanh(1 + phi) + p['y0']

yerr = 0.05
ysdev = yerr * np.ones(len(xdata))
true_y = fcn('data', true_par)
ymean = true_y + ysdev * np.random.randn(len(ysdev))
y = gvar.gvar(ymean, ysdev)

prior = dict(
    phi=gp.prior('data'),
    y0=gvar.gvar(0, 1000)
)

p0=dict(
    phi=np.random.multivariate_normal(np.zeros(len(xdata)), gvar.evalcov(prior['phi']))
)

fit = lsqfit.nonlinear_fit(data=('data', y), prior=prior, fcn=fcn, p0=p0)
print(fit.format(maxline=True))

ypred = fcn('pred', fit.p)
ypredalt = fcn('pred', fit.palt)

phipred = gp.predfromfit({'data': fit.p['phi']}, 'pred')

fig, axs = plt.subplots(1, 2, num='c', clear=True)

for ax, variable in zip(axs, ['y', 'phi']):
    ax.set_title(variable)
    
    for label in 'pred', 'predalt':
        if variable == 'phi' and label == 'predalt':
            continue
        
        pred = eval(variable + label)

        m = gvar.mean(pred)
        s = gvar.sdev(pred)

        patch = ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)
        color = patch.get_facecolor()[0]
    
        if plot_simulated_lines:
            cov = gvar.evalcov(pred)
            simulated_lines = np.random.multivariate_normal(m, cov, size=10)
            ax.plot(xpred, simulated_lines.T, '-', color=color)
    
    if variable == 'phi':
        ax.plot(xdata, true_par['phi'], 'rx', label='true')

axs[0].errorbar(xdata, gvar.mean(y), yerr=gvar.sdev(y), fmt='k.', label='data')
axs[0].plot(xdata, true_y, 'rx', label='true')

for ax in axs:
    ax.legend(loc='best')

fig.tight_layout()
fig.show()
