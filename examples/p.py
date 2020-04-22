import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 1000)
xpred = np.linspace(-15, 25, 300)
y = np.sin(xdata)

gp = lgp.GP(lgp.ExpQuad(scale=3), solver='lowrank', rank=10, checkpos=False)
gp.addx(xdata, 'pere')
gp.addx(xpred, 'banane')

print('fit...')
m, cov = gp.predfromdata({'pere': y}, 'banane', raw=True)
# u = gp.predfromdata({'pere': y}, 'banane')
# print('extract cov...')
# m = gvar.mean(u)
# cov = gvar.evalcov(u)

s = np.sqrt(np.diag(cov))

print('plot...')
fig = plt.figure('p')
fig.clf()
ax = fig.subplots(1, 1)

patch = ax.fill_between(xpred, m - s, m + s, label='pred', alpha=0.5)
color = patch.get_facecolor()[0]
simulated_lines = np.random.multivariate_normal(m, cov, size=10)
ax.plot(xpred, simulated_lines.T, '-', color=color)
ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
