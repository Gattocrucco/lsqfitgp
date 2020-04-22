import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 200)
y = np.sin(xdata)

print('make GP...')
gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata, 'data')
gp.addx(xpred, 'pred', 0)
gp.addx(xpred, 'predderiv', 1)

print('fit...')
umean, ucov = gp.predfromdata({'data': y}, ['pred', 'predderiv'], raw=True)
ualt = gp.predfromdata({'data': y}, ['pred', 'predderiv'])

print('figure...')
fig = plt.figure('testgp2g')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for label in umean:
    m = umean[label]
    s = np.sqrt(np.diag(ucov[label, label]))
    patch = ax.fill_between(xpred, m - s, m + s, label=label + ' (raw)', alpha=0.5)
    colors[label] = patch.get_facecolor()[0]
    
for label in ualt:
    m = gvar.mean(ualt[label])
    s = gvar.sdev(ualt[label])
    ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)

print('samples...')
for label in umean:
    m = umean[label]
    cov = ucov[label, label]
    samples = np.random.multivariate_normal(m, cov, size=10)
    ax.plot(xpred, samples.T, '-', color=colors[label])

ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
