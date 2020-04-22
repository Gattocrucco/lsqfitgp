import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

xdata = np.linspace(0, 10, 10)
xpred = np.linspace(-15, 25, 200)
y = np.sin(xdata)

print('make GP...')
gp = lgp.GP(lgp.Matern52(scale=3), checkpos=False)
gp.addx(xdata, 'data')
gp.addx(xpred, 'pred')
gp.addx(xpred, 'deriv', 1)

print('fit...')
u = gp.pred({'data': y}, ['pred', 'deriv'], fromdata=True)

print('figure...')
fig = plt.figure('k')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
for label in u:
    m = gvar.mean(u[label])
    s = gvar.sdev(u[label])
    patch = ax.fill_between(xpred, m - s, m + s, label=label, alpha=0.5)
    colors[label] = patch.get_facecolor()[0]
    
print('samples...')
for i, sample in zip(range(1), gvar.raniter(u)):
    for label in u:
        ax.plot(xpred, sample[label], '-', color=colors[label])
ax.plot(xdata, y, 'k.', label='data')
ax.legend(loc='best')

fig.show()
