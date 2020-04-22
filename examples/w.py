import numpy as np
from matplotlib import pyplot as plt
import gvar

import lsqfitgp as lgp

time = np.arange(30)
time_pred = np.linspace(-30, 60, 200)

def makex(time, comp):
    x = np.empty(len(time), dtype=[
        ('time', float),
        ('comp', 'U8')
    ])
    x['time'] = time
    x['comp'] = comp
    return x

kshort = lgp.ExpQuad(scale=1, dim='time')
klong = lgp.ExpQuad(scale=10, dim='time')
kernel = lgp.where(lambda comp: comp == 'short', kshort, klong, dim='comp')
gp = lgp.GP(kernel)

def addcomps(key, time):
    gp.addx(makex(time, 'short'), key + 'short')
    gp.addx(makex(time, 'long'), key + 'long')
    gp.addtransf({key + 'short': 0.3, key + 'long': 1}, key)

addcomps('data', time)
addcomps('pred', time_pred)

print('generate data...')
prior = gp.prior(['data', 'datashort', 'datalong'])
data = next(gvar.raniter(prior))

print('prediction...')
pred = gp.predfromdata({'data': data['data']}, ['pred', 'predshort', 'predlong'])

print('sample posterior...')
mean = gvar.mean(pred)
sdev = gvar.sdev(pred)
samples = [sample for _, sample in zip(range(1), gvar.raniter(pred))]

print('figure...')
fig = plt.figure('testgp2w')
fig.clf()
fig.set_size_inches(6, 7)
axs = fig.subplots(3, 1)

for ax, comp in zip(axs, ['', 'short', 'long']):
    key = 'pred' + comp
    
    m = mean[key]
    s = sdev[key]
    ax.fill_between(time_pred, m - s, m + s, alpha=0.3, color='b')
    
    for sample in samples:
        ax.plot(time_pred, sample[key], alpha=0.2, color='b')
    
    ax.plot(time, data['data' + comp], '.k')

fig.tight_layout()
fig.show()
