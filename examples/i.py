import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 300)
y = np.sin(xdata)
yerr = 0.1

gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata, 'pere')
gp.addx(xpred, 'banane')

uy = gvar.gvar(y + yerr * np.random.randn(len(y)), yerr * np.ones_like(y))
u = gp.predfromdata({'pere': uy}, 'banane', keepcorr=False)
assert gvar.cov(u[0], uy[0]) == 0
m = gvar.mean(u)
s = gvar.sdev(u)
cov = gvar.evalcov(u)

fig = plt.figure('i')
fig.clf()
ax = fig.subplots(1, 1)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.errorbar(xdata, gvar.mean(uy), yerr=gvar.sdev(uy), fmt='k.', label='data')
ax.legend(loc='best')

fig.show()
