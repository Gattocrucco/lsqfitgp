import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

x = np.linspace(-10, 10, 200)
derivs = [0, 1, 2]

gp = lgp.GP(lgp.ExpQuad(scale=2))
for d in derivs:
    gp.addx(x, d, d)

cov = gp.prior(raw=True)

fig = plt.figure('testgp2o')
fig.clf()
ax = fig.subplots(1, 1)

colors = dict()
m = np.zeros(len(x))
for deriv in derivs:
    s = np.sqrt(np.diag(cov[deriv, deriv]))
    patch = ax.fill_between(x, m - s, m + s, label=f'deriv {deriv}', alpha=0.5)
    colors[deriv] = patch.get_facecolor()[0]
    
for deriv in derivs:
    samples = np.random.multivariate_normal(m, cov[deriv, deriv], size=5)
    ax.plot(x, samples.T, '-', color=colors[deriv])

ax.legend(loc='best')
ax.grid(linestyle=':')

fig.show()
