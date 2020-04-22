import lsqfitgp as lgp
from lsqfitgp import _linalg
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import gvar

xdata1d = np.linspace(-4, 4, 10)
xpred1d = np.linspace(-10, 10, 50)

def makexy(x1d, y1d):
    xy = np.empty((len(x1d), len(y1d)), dtype=[
        ('x', float),
        ('y', float)
    ])
    x, y = np.meshgrid(x1d, y1d)
    xy['x'] = x
    xy['y'] = y
    return xy
    
xdata = makexy(xdata1d, xdata1d)
xpred = makexy(xpred1d, xpred1d)
y = np.cos(xdata['x']) * np.cos(xdata['y'])

gp = lgp.GP(lgp.ExpQuad(scale=3, dim='x') * lgp.ExpQuad(scale=3, dim='y'), checkpos=False, solver='gersh')
gp.addx(xdata.reshape(-1), 'pere')
gp.addx(xpred.reshape(-1), 'banane')

print('fit...')
m, cov = gp.predfromdata({'pere': y.reshape(-1)}, 'banane', raw=True)

print('samples...')
# samples = np.random.multivariate_normal(m, cov)
# dec = _linalg.LowRank(cov, rank=300)
# samples = m + dec._V @ (np.random.randn(len(dec._w)) * dec._w)
sample = m + _linalg.CholGersh(cov, eps=1e-5)._L @ np.random.randn(len(m))
sample = sample.reshape(xpred.shape)

print('plot...')
fig = plt.figure('q')
fig.clf()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xdata['x'].reshape(-1), xdata['y'].reshape(-1), y.reshape(-1), color='black')
ax.plot_surface(xpred['x'], xpred['y'], sample, alpha=0.8)

fig.show()
