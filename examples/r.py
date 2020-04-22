import lsqfitgp as lgp
from lsqfitgp import _linalg
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import gvar

xdata1d = np.linspace(-4, 4, 10)
xpred1d = np.linspace(-10, 10, 50)

def makegrid(array1d):
    x, y = np.meshgrid(array1d, array1d)
    out = np.empty(len(array1d) * len(array1d), [('x', float), ('y', float)])
    out['x'] = x.reshape(-1)
    out['y'] = y.reshape(-1)
    return out

xdata = makegrid(xdata1d)
xpred = makegrid(xpred1d)
y = np.cos(xdata['x']) * np.cos(xdata['y'])

gp = lgp.GP(lgp.ExpQuad(scale=3, dim='x') * lgp.ExpQuad(scale=1, dim='y'), checkpos=False, solver='gersh')
gp.addx(xdata, 'pere')
gp.addx(xpred, 'banane')

print('fit...')
m, cov = gp.predfromdata({'pere': y}, 'banane', raw=True)

print('samples...')
# samples = np.random.multivariate_normal(m, cov)
# dec = _linalg.LowRank(cov, rank=300)
# samples = m + dec._V @ (np.random.randn(len(dec._w)) * dec._w)
samples = m + _linalg.CholGersh(cov, eps=1e-5)._L @ np.random.randn(len(cov))

print('plot...')
fig = plt.figure('testgp2r')
fig.clf()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xdata['x'], xdata['y'], y, color='black')
plotxpred = xpred.reshape(len(xpred1d), len(xpred1d))
ax.plot_surface(plotxpred['x'], plotxpred['y'], samples.reshape(plotxpred.shape), alpha=0.85)

for axis in 'xyz':
    exec(f'ax.set_{axis}label("{axis}")')

fig.show()
