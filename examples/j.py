import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 300)
y = np.sin(xdata)

gp = lgp.GP(lgp.Matern(scale=5, nu=3))
gp.addx(xdata, 'pere')
gp.addx(xpred, 'banane')

u = gp.predfromdata({'pere': y}, 'banane')
m = gvar.mean(u)
s = gvar.sdev(u)
cov = gvar.evalcov(u)

fig = plt.figure('testgp2j')
fig.clf()
ax = fig.subplots(1, 1)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
