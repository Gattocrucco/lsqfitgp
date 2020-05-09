# lsqfitgp/examples/m.py
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

import lsqfitgp as lgp
from matplotlib import pyplot as plt
from autograd import numpy as np
import gvar
from scipy import optimize
import autograd

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 500)
y = np.sin(xdata)

def makegp(par):
    scale = np.exp(par[0])
    return lgp.GP(lgp.Matern52(scale=scale))

def fun(par):
    gp = makegp(par)
    gp.addx(xdata, 'data')
    return -gp.marginal_likelihood(y)

result = optimize.minimize(fun, np.log([5]), jac=autograd.grad(fun))
print(result)

gp = makegp(result.x)
gp.addx(xdata, 'data')
gp.addx(xpred, 'pred')

m, cov = gp.predfromdata({'data': y}, 'pred', raw=True)
s = np.sqrt(np.diag(cov))

fig = plt.figure('m')
fig.clf()
ax = fig.subplots(1, 1)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
