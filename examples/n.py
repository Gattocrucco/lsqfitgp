import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

x = np.linspace(-10, 10, 200)
derivs = [0, 1, 2]

gp = lgp.GP(lgp.ExpQuad(scale=2))
for d in derivs:
    gp.addx(x, d, d)

u = gp.prior()

fig = plt.figure('testgp2n')
fig.clf()
ax = fig.subplots(1, 1)

ax.axhline(0)

colors = dict()
for deriv in derivs:
    m = gvar.mean(u[deriv])
    s = gvar.sdev(u[deriv])
    patch = ax.fill_between(x, m - s, m + s, label=f'deriv {deriv}', alpha=0.5)
    colors[deriv] = patch.get_facecolor()[0]
    
for i, sample in zip(range(1), gvar.raniter(u)):
    for deriv in derivs:
        ax.plot(x, sample[deriv], '-', color=colors[deriv])

ax.legend(loc='best')
ax.grid(linestyle=':')

fig.show()
