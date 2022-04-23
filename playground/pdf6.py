"""Fit of parton distributions functions (PDFs)

Like pdf5, but with uncertainties on M and M2"""

import lsqfitgp as lgp
import numpy as np
from matplotlib import pyplot as plt
import gvar
import lsqfit

np.random.seed(20220416)

#### COMPONENTS ####

flavor = np.array([
    ( 1, 'd'    ), # 0
    (-1, 'dbar' ), # 1
    ( 2, 'u'    ), # 2
    (-2, 'ubar' ), # 3
    ( 3, 's'    ), # 4
    (-3, 'sbar' ), # 5
    ( 4, 'c'    ), # 6
    (-4, 'cbar' ), # 7
    (21, 'gluon'), # 8
], 'i8, U16')

pid  = flavor['f0']
name = flavor['f1']

nflav  = len(flavor)
nx     = 30
ndata  = 10
nx2    = 30  # must be <= nx
ndata2 = 10

indices = dict(
    # quark, antiquark
    d = [0, 1],
    u = [2, 3],
    s = [4, 5],
    c = [6, 7],
)

#### MODEL ####
# for each component of the proton:
# h ~ GP
# f = h''  (f is the PDF)
# for the momentum sum rule:
# int_0^1 dx x f(x) = [xh'(x) - h(x)]_0^1
# for the flavor sum rules:
# int_0^1 dx (f_i(x) - f_j(x)) = [h_i'(x) - h_j'(x)]_0^1

xtype = np.dtype([
    ('x'  , float),
    ('pid', int  ),
])

kernel = lgp.ExpQuad(dim='x') * lgp.White(dim='pid')

xdata = np.empty((nflav, nx), xtype)
xdata['pid'] = pid[:, None]
xdata[  'x'] = np.linspace(0, 1, nx)

M = np.random.randn(ndata, nflav, nx) # linear map PDF(X) -> data
# Msdev = np.full_like(Mtrue, 0.01)
# Mmean = Mtrue + Msdev * np.random.randn(*Merr.shape)
# M = gvar.gvar(Mean, Msdev)

A = np.random.randn(ndata2, nflav, nx2, nx2)
M2 = np.einsum('dfxz,dfyz->dfxy', A, A) # quadratic map PDF(X) -> data2
# Why am I using a positive definite M2? Habit I guess.

xinteg = np.empty((nflav, 2), xtype)
xinteg['pid'] = pid[:, None]
xinteg[  'x'] = [0, 1]

suminteg = np.empty(xinteg.shape)
suminteg[:, 0] = -1
suminteg[:, 1] =  1

constraints = {
    'momrule': 1,
    'uubar'  : 2,
    'ddbar'  : 1,
    'ccbar'  : 0,
    'ssbar'  : 0,
}

#### CREATE GP OBJECT ####

gp = lgp.GP()

gp.addproc(kernel, 'h')
gp.addproctransf({'h': 1}, 'primitive', deriv='x'     )
gp.addproctransf({'h': 1}, 'f'        , deriv=(2, 'x'))
gp.addproctransf({
    'primitive': lambda x: x['x'],
    'h'        : -1,
}, 'primitive of xf(x)')

# data
gp.addx(xdata, 'xdata', proc='f')
gp.addtransf({'xdata': M}, 'data', axes=2)

# total momentum rule
gp.addx(xinteg, 'xmomrule', proc='primitive of xf(x)')
gp.addtransf({'xmomrule': suminteg}, 'momrule', axes=2)

# quark sum rules
qdiff = np.array([1, -1])[:, None]
for quark in 'ducs':
    idx = indices[quark]
    label = f'{quark}{quark}bar' # the one appearing in `constraints`
    xlabel = f'x{label}'
    gp.addx(xinteg[idx], xlabel, proc='primitive')
    gp.addtransf({xlabel: suminteg[idx] * qdiff}, label, axes=2)
    
#### DEFINE NONLINEAR FUNCTION ####

def fcn(params):
    
    xdata = params['xdata']
    data = np.tensordot(M, xdata, 2)
    # data2 = np.einsum('dfxy,fx,fy->d', M2, xdata, xdata)
    xdata2 = xdata[:, None, :nx2] * xdata[:, :nx2, None]
    data2 = np.tensordot(M2, xdata2, 3)
    
    return dict(data=data, data2=data2)

params_prior = gp.predfromdata(constraints, ['xdata'])

#### GENERATE FAKE DATA ####

priorsample = gvar.sample(params_prior)
fcnsample = fcn(priorsample)

datamean = gvar.BufferDict({
    'data' : fcnsample['data' ],
    'data2': fcnsample['data2'],
})
dataerr = gvar.BufferDict({
    'data' : np.full(ndata ,   1),
    'data2': np.full(ndata2, 100),
})
datamean.buf += dataerr.buf * np.random.randn(*dataerr.buf.shape)
data = gvar.gvar(datamean, dataerr)

# check sum rules approximately with trapezoid rule
def check_integrals(x, y):
    checksum = np.sum(((y * x)[:, 1:] + (y * x)[:, :-1]) / 2 * np.diff(x, axis=1))
    print('sum_i int dx x f_i(x) =', checksum)
    for q in 'ducs':
        idx = indices[q]
        qx = x[idx]
        qy = y[idx]
        checksum = np.sum(qdiff * (qy[:, 1:] + qy[:, :-1]) / 2 * np.diff(qx, axis=1))
        print(f'sum_i={q}{q}bar int dx f_i(x) =', checksum)

print('check integrals in fake data:')
check_integrals(xdata['x'], priorsample['xdata'])

#### FIT ####

# find the minimization starting point with a simplified version of the fit
easyfit = gp.predfromdata(dict(data=data['data'], **constraints), list(params_prior.keys()))
p0 = gvar.mean(easyfit)

fit = lsqfit.nonlinear_fit(data, fcn, params_prior, p0=p0, verbose=2)
print(fit.format(maxline=True))
print(fit.format(maxline=-1))

pred = fcn(fit.p)

print('check integrals in fit:')
check_integrals(xdata['x'], fit.p['xdata'])

#### PLOT RESULTS ####

fig, axs = plt.subplots(1, 3, num='pdf6', clear=True, figsize=[12, 4.5])
axs[0].set_title('PDFs')
axs[1].set_title('Data')
axs[2].set_title('Data (quadratic)')

for i in range(nflav):
    
    x = xdata[i]['x']
    ypdf = fit.p['xdata'][i]
    ydata = priorsample['xdata'][i]
    m = gvar.mean(ypdf)
    s = gvar.sdev(ypdf)
    
    color = 'C' + str(i // 2)
    if i % 2:
        kw = dict(hatch='//////', edgecolor=color, facecolor='none')
        kwp = dict(color=color, linestyle='--')
    else:
        kw = dict(alpha=0.6, facecolor=color)
        kwp = dict(color=color)
    
    axs[0].fill_between(x, m - s, m + s, label=name[i], **kw)
    axs[0].plot(x, ydata, **kwp)

axs[0].legend(fontsize='small')

for ax, label in zip(axs[1:], ['data', 'data2']):
    
    m = gvar.mean(pred[label])
    s = gvar.sdev(pred[label]) if len(m) else []
    x = np.arange(len(m))
    ax.fill_between(x, m - s, m + s, step='mid', color='lightgray')
    ax.errorbar(x, datamean[label], dataerr[label], color='black', linestyle='', capsize=2)
    ax.plot(x, fcnsample[label], drawstyle='steps-mid', color='black')

fig.tight_layout()
fig.show()
