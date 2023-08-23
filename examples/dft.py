# lsqfitgp/examples/dft.py
#
# Copyright (c) 2022, Giacomo Petrillo
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
Constrain the discrete Fourier transform of a periodic process. Shows how
to use GP.addlintransf.
"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp
import gvar

class PORCO: pass
class DUO: pass

gp = lgp.GP(lgp.Zeta(scale=2 * np.pi, nu=0.5))
# We could do the same with a non-periodic prior, but it would not make as much
# sense. Moreover we use the Zeta kernel with nu=0.5, i.e., prior on the
# coefficients going down like 1/k. A smoother kernel would yield a strongly
# shrinking prior on the DFT coefficients. Try it.

xdata = np.linspace(0, 2 * np.pi, 20)
gp = gp.addx(xdata, PORCO)

xpred = np.linspace(-2 * np.pi, 4 * np.pi, 50 * 6 + 1)
# If we don't align the number of points to the period, the samples will not
# show as periodic because the function is very rough. The distribution would
# still be periodic though, it's just an artifact of looking at a coarse grid.
gp = gp.addx(xpred, DUO)

def transf(x):
    # use jax.numpy instead of numpy in this function
    f = jnp.fft.rfft(x)
    return jnp.stack([jnp.real(f), jnp.imag(f)])

gp = gp.addlintransf(transf, [PORCO], 'dft')

gp = gp.addlintransf(lambda x: x[0, 3], ['dft'], '3rd real coef')
gp = gp.addlintransf(lambda x: x[0, 3] - x[1, 3], ['dft'], '3rd coef pseudo-phase')
# We will force the 3rd spectrum coefficient to be high because the supervisor
# said that it always comes out high, so the plot can't be different from the
# other articles. The statistics professor said that the correct fitting method
# is "Bayesian statistics" and that everything is subjective, so we may as
# well hardcode the desiratum into the prior.

u = gp.predfromdata({
    '3rd real coef': gvar.gvar(10, 1),
    '3rd coef pseudo-phase': gvar.gvar(5, 1),
})
mean = gvar.mean(u)
sdev = gvar.sdev(u)
cov = gvar.evalcov(u)

fig, axs = plt.subplots(2, 1, num='dft', clear=True, figsize=[6.4, 7])

ax = axs[0]
ax.set_title('Function')
m = mean[DUO]
s = sdev[DUO]
patch = ax.fill_between(xpred, m - s, m + s, alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov[DUO, DUO])
ax.plot(xpred, simulated_lines, '-', color=color)
ax.plot(xdata, np.zeros_like(xdata), '.k', label='discrete lattice')

ax = axs[1]
ax.set_title('DFT')
simul = gvar.sample(u['dft'])
for i, label in enumerate(['real', 'imag']):
    m = mean['dft'][i]
    s = sdev['dft'][i]
    n = len(m)
    patch = ax.fill_between(np.arange(n), m - s, m + s, alpha=0.5, label=label)
    color = patch.get_facecolor()[0]
    ax.plot(np.arange(n), simul[i], color=color)

for ax in axs:
    ax.legend()
    ax.grid(linestyle='--')

fig.show()
