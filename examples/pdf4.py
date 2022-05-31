"""Fit of parton distribution functions (PDFs)

Like pdf3, but with hyperparameters"""

import lsqfitgp as lgp
import numpy as np
from matplotlib import pyplot as plt
import gvar

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

nflav = len(flavor)
nx    = 30
ndata = 20

indices = dict(
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

# fitting both scale and variance with weak priors is a bad idea, but whatever
hyperprior = {
    'log(scale)': gvar.log(gvar.gvar(0.5, 0.5)),
    'log(ampl)' : gvar.log(gvar.gvar(1, 1)),
    'log(alpha)': gvar.log(gvar.gvar(2, 1)),
}
def makekernel(hp):
    return hp['ampl'] ** 2 * lgp.RatQuad(dim='x', scale=hp['scale'], alpha=hp['alpha']) * lgp.White(dim='pid')

xdata = np.empty((nflav, nx), xtype)
xdata['pid'] = pid[:, None]
xdata[  'x'] = np.linspace(0, 1, nx)

M = np.random.randn(ndata, nflav, nx) # transformation PDF(X) -> data

xinteg = np.empty((nflav, 2), xtype)
xinteg['pid'] = pid[:, None]
xinteg[  'x'] = [0, 1]

suminteg = np.empty(xinteg.shape)
suminteg[:, 0] = -1
suminteg[:, 1] =  1

qdiff = np.array([1, -1])[:, None]

constraints = {
    'momrule': 1,
    'uubar'  : 2,
    'ddbar'  : 1,
    'ccbar'  : 0,
    'ssbar'  : 0,
}

#### CREATE GP OBJECT ####

def makegp(hp):
    
    gp = lgp.GP()

    kernel = makekernel(hp)
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
    for quark in 'ducs':
        idx = indices[quark]
        label = f'{quark}{quark}bar' # the one appearing in `constraints`
        xlabel = f'x{label}'
        gp.addx(xinteg[idx], xlabel, proc='primitive')
        gp.addtransf({xlabel: suminteg[idx] * qdiff}, label, axes=2)

    return gp

#### GENERATE FAKE DATA ####

hpsample = next(gvar.raniter(hyperprior))
gp = makegp(hpsample)
prior = gp.predfromdata(constraints, ['data', 'xdata'])
priorsample = next(gvar.raniter(prior))

datamean = priorsample['data']
dataerr = np.full_like(datamean, 1)
datamean = datamean + dataerr * np.random.randn(*dataerr.shape)
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
information = dict(constraints)
information.update({
    'data': data,
})
fit = lgp.empbayes_fit(hyperprior, makegp, information, raises=False)

print('hyperparameters:')
for k in fit.p.all_keys():
    d = fit.p[k] - hpsample[k]
    print(f'{k:10}: fit {fit.p[k]} true {hpsample[k]:5.2g} diff {d}')

gp = makegp(gvar.mean(fit.p))
pred = gp.predfromdata(information, ['data', 'xdata'])

print('check integrals in fit:')
check_integrals(xdata['x'], pred['xdata'])

#### PLOT RESULTS ####

fig, axs = plt.subplots(1, 2, num='pdf4', clear=True, figsize=[9, 4.5])
axs[0].set_title('PDFs')
axs[1].set_title('Data')

for i in range(nflav):
    
    x = xdata[i]['x']
    ypdf = pred['xdata'][i]
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

m = gvar.mean(pred['data'])
s = gvar.sdev(pred['data'])
x = np.arange(len(data))
axs[1].fill_between(x, m - s, m + s, step='mid', color='lightgray')
axs[1].errorbar(x, datamean, dataerr, color='black', linestyle='', capsize=2)

fig.tight_layout()
fig.show()
