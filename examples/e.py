import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(-5, 5, 10)
xpred = np.linspace(-15, 25, 200)
y = np.sin(xdata)
y[1::2] = np.cos(xdata[1::2])

print('make GP...')
gp = lgp.GP(lgp.ExpQuad(scale=3))
gp.addx(xdata[0::2], 'data', 0)
gp.addx(xdata[1::2], 'dataderiv', 1)
gp.addx(xpred, 'pred', 0)
gp.addx(xpred, 'predderiv', 1)

print('fit...')
u = gp.predfromdata({'data': y[0::2], 'dataderiv': y[1::2]}, ['pred', 'predderiv'])

print('figure...')
fig = plt.figure('testgp2e')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for label in u:
    m = gvar.mean(u[label])
    s = gvar.sdev(u[label])
    patch = ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)
    colors[label] = patch.get_facecolor()[0]
    
print('samples...')
for i, sample in zip(range(30), gvar.raniter(u)):
    for label in u:
        ax.plot(xpred, sample[label], '-', color=colors[label])

for deriv, marker in (0, '+'), (1, 'x'):
    ax.plot(xdata[deriv::2], y[deriv::2], f'k{marker}', label=f'data deriv {deriv}')
ax.legend(loc='best')

fig.show()
