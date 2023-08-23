"""Fit of parton distributions functions (PDFs)

The difference from pdf1.py is that we define the transformation on processes
instead of on their finite realizations"""

import warnings

import lsqfitgp as lgp
import numpy as np
from matplotlib import pyplot as plt
import gvar

np.random.seed(20220416)
warnings.filterwarnings('ignore', r'total derivative orders \(\d+, \d+\) greater than kernel minimum \(\d+, \d+\)')

#### DEFINE MODEL ####
# for each gluon:
# h ~ GP
# f = h''
# int_0^1 dx f(x) = [h'(x)]_0^1
# int_0^1 dx x f(x) = [xh'(x) - h(x)]_0^1

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

gp = (lgp
    .GP()

    .defproc('h', kernel)
    .defproctransf('primitive of f', {'h': 1}, deriv='x')
    .defproctransf('f', {'h': 1}, deriv=(2, 'x'))
    .defproctransf('primitive of xf(x)', {'primitive of f': lambda x: x['x'], 'h': -1})

    .addx(xdata, 'xdata', proc='f')
    .addtransf({'xdata': M}, 'data', axes=2)

    .addx(xinteg, 'xinteg', proc='primitive of f')
    .addtransf({'xinteg': suminteg}, 'suminteg', axes=2)

    .addx(xinteg, 'xintegx', proc='primitive of xf(x)')
    .addtransf({'xintegx': suminteg}, 'sumintegx', axes=2)
)

#### GENERATE FAKE DATA ####

prior = gp.predfromdata({
    'suminteg' : 1,
    'sumintegx': 1,
}, ['data', 'xdata'])
priorsample = next(gvar.raniter(prior))

datamean = priorsample['data']
dataerr = np.full_like(datamean, 1)
datamean = datamean + dataerr * np.random.randn(*dataerr.shape)
data = gvar.gvar(datamean, dataerr)

# check the integral is one with trapezoid rule
x = xdata['x']
y = priorsample['xdata']
checksum = np.sum((      y[:, 1:] +       y[:, :-1]) / 2 * np.diff(x, axis=1))
print('sum_i int dx   f_i(x) =', checksum)
checksum = np.sum(((y * x)[:, 1:] + (y * x)[:, :-1]) / 2 * np.diff(x, axis=1))
print('sum_i int dx x f_i(x) =', checksum)

#### FIT ####

pred = gp.predfromdata({
    'suminteg' :    1,
    'sumintegx':    1,
    'data'     : data,
}, ['data', 'xdata'])

# check the integral is one with trapezoid rule
x = xdata['x']
y = pred['xdata']
checksum = np.sum((      y[:, 1:] +       y[:, :-1]) / 2 * np.diff(x, axis=1))
print('sum_i int dx   f_i(x) =', checksum)
checksum = np.sum(((y * x)[:, 1:] + (y * x)[:, :-1]) / 2 * np.diff(x, axis=1))
print('sum_i int dx x f_i(x) =', checksum)

#### PLOT RESULTS ####

fig, axs = plt.subplots(1, 2, num='pdf2', clear=True, figsize=[9, 4.5])
axs[0].set_title('PDFs')
axs[1].set_title('Data')

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

fig.tight_layout()
fig.show()
