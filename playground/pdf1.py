"""Fit of parton distributions functions (PDFs)"""

import lsqfitgp as lgp
import numpy as np
from matplotlib import pyplot as plt
import gvar

np.random.seed(20220416)

xtype = np.dtype([
    ('x'    , float),
    ('gluon', int  ),
])

kernel = lgp.ExpQuad(dim='x') * lgp.White(dim='gluon')

xdata = np.empty((8, 30), xtype)
for i in range(len(xdata)):
    xdata[i]['gluon'] = i + 1
    xdata[i]['x'] = np.linspace(0, 1, len(xdata[i]))
    
M = np.random.randn(20, 8, 30)

gp = lgp.GP(kernel)

gp.addx(xdata, 'xbase')
gp.addtransf({'xbase': M}, 'data', axes=2)

prior = gp.prior(['data', 'xbase'])
priorsample = next(gvar.raniter(prior))

datamean = priorsample['data']
dataerr = np.full_like(datamean, 1)
data = gvar.gvar(datamean + dataerr * np.random.randn(*dataerr.shape), dataerr)

pred = gp.predfromdata({'data': data}, ['xbase', 'data'])

fig, axs = plt.subplots(1, 2, num='pdf1', clear=True, figsize=[9, 4.5])
axs[0].set_title('PDFs')
axs[1].set_title('Data')

for i in range(len(xdata)):
    y = pred['xbase'][i]
    m = gvar.mean(y)
    s = gvar.sdev(y)
    axs[0].fill_between(xdata[i]['x'], m - s, m + s, alpha=0.6, facecolor=f'C{i}')
    y2 = priorsample['xbase'][i]
    axs[0].plot(xdata[i]['x'], y2, color=f'C{i}')

m = gvar.mean(pred['data'])
s = gvar.sdev(pred['data'])
x = np.arange(len(data))
axs[1].fill_between(x, m - s, m + s, step='mid', color='lightgray')
axs[1].errorbar(x, datamean, dataerr, color='black', linestyle='', capsize=2)

fig.show()
