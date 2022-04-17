"""Test of double integral constraint"""

import lsqfitgp as lgp
from autograd import numpy as np
from matplotlib import pyplot as plt
import gvar

np.random.seed(20220417)

#### DEFINE MODEL ####

gp = lgp.GP(lgp.ExpQuad())

x = np.linspace(0, 1, 30)
gp.addx(x, 'data', deriv=2)

gp.addx([0, 1], 'xinteg', deriv=1)
gp.addtransf({'xinteg': [-1, 1]}, 'integ')

gp.addx([0, 1], 'xintegx0')
gp.addx(1, 'xintegx1', deriv=1)
gp.addtransf({'xintegx1': 1, 'xintegx0': [1, -1]}, 'integx')

#### GENERATE FAKE DATA ####

prior = gp.predfromdata({
    'integ' : 1,
    'integx': 1,
}, ['data', 'integ', 'integx'])
priorsample = next(gvar.raniter(prior))

datamean = priorsample['data']
dataerr = np.full_like(datamean, 0.1)
datamean = datamean + dataerr * np.random.randn(*dataerr.shape)
data = gvar.gvar(datamean, dataerr)

# check the integral is one with trapezoid rule
print('prior:')
y = priorsample['data']
checksum = np.sum((      y[1:] +       y[:-1]) / 2 * np.diff(x))
print('sum_i int dx   f_i(x) =', checksum)
checksum = np.sum(((y * x)[1:] + (y * x)[:-1]) / 2 * np.diff(x))
print('sum_i int dx x f_i(x) =', checksum)

#### FIT ####

pred = gp.predfromdata({
    'integ' :    1,
    'integx':    1,
    'data'  : data,
}, ['data', 'integ', 'integx'])

# check the integral is one with trapezoid rule
print('posterior:')
y = pred['data']
checksum = np.sum((      y[1:] +       y[:-1]) / 2 * np.diff(x))
print('sum_i int dx   f_i(x) =', checksum)
checksum = np.sum(((y * x)[1:] + (y * x)[:-1]) / 2 * np.diff(x))
print('sum_i int dx x f_i(x) =', checksum)

#### PLOT RESULTS ####

fig, ax = plt.subplots(num='doubleint', clear=True)

y = pred['data']
m = gvar.mean(y)
s = gvar.sdev(y)
ax.fill_between(x, m - s, m + s, alpha=0.6)

y = priorsample['data']
ax.plot(x, y)

y = pred['data'] * x
m = gvar.mean(y)
s = gvar.sdev(y)
ax.fill_between(x, m - s, m + s, alpha=0.6)

y = priorsample['data'] * x
ax.plot(x, y)

ax.errorbar(x, datamean, dataerr, color='black', linestyle='', capsize=2)

fig.show()
