import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure('testgp2l')
fig.clf()
ax = fig.subplots(1, 1)

x = np.linspace(0, 10, 1000)

for dv in 1, 3, 5:
    v = dv / 2
    y1 = lgp.Matern(nu=v)([0.], x)
    y2 = eval(f'lgp.Matern{dv}2')()([0.], x)
    line, = ax.plot(x, y1, label=f'{dv}/2', alpha=0.5)
    ax.plot(x, y2, '--', color=line.get_color(), alpha=0.5)

ax.legend(loc='best')
ax.set_yscale('log')
fig.show()
