"""Fit of parton distributions functions (PDFs)"""

import lsqfitgp as lgp
import numpy as np
from matplotlib import pyplot as plt
import gvar

np.random.seed(20220416)

#### DEFINE MODEL ####

xtype = np.dtype([
    ('x'    , float),
    ('gluon', int  ),
])

kernel = lgp.ExpQuad(dim='x') * lgp.White(dim='gluon')

xdata = np.empty((8, 30), xtype)
xdata['gluon'] = np.arange(8)[:, None]
xdata[    'x'] = np.linspace(0, 1, 30)
    
M = np.random.randn(20, 8, 30) # xdata -> data transform

xinteg = np.empty((8, 2), xtype)
xinteg['gluon'] = np.arange(8)[:, None]
xinteg[    'x'] = [0, 1]

suminteg = np.empty(xinteg.shape)
suminteg[:, 0] = -1
suminteg[:, 1] =  1

#### CREATE GP OBJECT ####

gp = lgp.GP(kernel)

gp.addx(xdata, 'xdata', deriv='x')
gp.addtransf({'xdata': M}, 'data', axes=2)

gp.addx(xinteg, 'xinteg')
gp.addtransf({'xinteg': suminteg}, 'suminteg', axes=2)

#### GENERATE FAKE DATA ####

prior = gp.predfromdata({
    'suminteg': 1,
}, ['data', 'xdata'])
priorsample = next(gvar.raniter(prior))

datamean = priorsample['data']
dataerr = np.full_like(datamean, 1)
datamean = datamean + dataerr * np.random.randn(*dataerr.shape)
data = gvar.gvar(datamean, dataerr)

#### FIT ####

pred = gp.predfromdata({
    'suminteg':    1,
    'data'    : data,
}, ['data', 'xdata'])

#### PLOT RESULTS ####

fig, axs = plt.subplots(1, 2, num='pdf1', clear=True, figsize=[9, 4.5])
axs[0].set_title('PDFs')
axs[1].set_title('Data')

# check the integral is one with trapezoid rule
checksum = np.sum((pred['xdata'][:, 1:] + pred['xdata'][:, :-1]) / 2 * np.diff(xdata['x'], axis=1))
print('should be one:', checksum)

for i in range(len(xdata)):
    y = pred['xdata'][i]
    m = gvar.mean(y)
    s = gvar.sdev(y)
    axs[0].fill_between(xdata[i]['x'], m - s, m + s, alpha=0.6, facecolor=f'C{i}')
    y2 = priorsample['xdata'][i]
    axs[0].plot(xdata[i]['x'], y2, color=f'C{i}')

m = gvar.mean(pred['data'])
s = gvar.sdev(pred['data'])
x = np.arange(len(data))
axs[1].fill_between(x, m - s, m + s, step='mid', color='lightgray')
axs[1].errorbar(x, datamean, dataerr, color='black', linestyle='', capsize=2)

fig.show()
