"""Fit of parton distributions functions (PDFs)

Like pdf6, but with hyperparameters"""

import lsqfitgp as lgp
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
import gvar
import lsqfit
import time

np.random.seed(20220416)

#### SETTINGS ####

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

indices = dict(
    # quark, antiquark
    d = [0, 1],
    u = [2, 3],
    s = [4, 5],
    c = [6, 7],
)

pid  = flavor['f0']
name = flavor['f1']

nflav  = len(flavor)

# linear data
nx        = 30 # number of PDF points used for the transformation
ndata     = 10 # number of datapoints
rankmcov  =  9 # rank of the covariance matrix of the theory error

# quadratic data
nx2       = 30 # must be <= nx
ndata2    = 10
rankmcov2 =  9

#### MODEL ####
# for each PDF:
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

hyperprior = {
    'log(scale)': np.log(gvar.gvar(0.5, 0.5)),
}
def makekernel(hp):
    return lgp.ExpQuad(dim='x', scale=hp['scale']) * lgp.White(dim='pid')

# grid of points to which we apply the transformation
xdata = np.empty((nflav, nx), xtype)
xdata['pid'] = pid[:, None]
xdata[  'x'] = np.linspace(0, 1, nx)

# linear map PDF(xdata) -> data
Mcomps = np.random.randn(rankmcov, ndata, nflav, nx)
Mcomps /= np.sqrt(Mcomps.size / ndata)
Mparams = gvar.gvar(np.random.randn(rankmcov), np.full(rankmcov, 0.1))
M = lambda params: np.tensordot(params, Mcomps, 1)

# quadratic map PDF(xdata) -> data2
M2comps = np.random.randn(rankmcov2, ndata2, nflav, nx2, nx2)
M2comps /= 2 * np.sqrt(M2comps.size / ndata2)
M2comps = (M2comps + np.swapaxes(M2comps, -1, -2)) / 2
M2params = gvar.gvar(np.random.randn(rankmcov2), np.full(rankmcov2, 0.1))
M2 = lambda params: np.tensordot(params, M2comps, 1)

# endpoints of the integral for each PDF
xinteg = np.empty((nflav, 2), xtype)
xinteg['pid'] = pid[:, None]
xinteg[  'x'] = [0, 1]

# matrix to subtract the endpoints
suminteg = np.empty(xinteg.shape)
suminteg[:, 0] = -1
suminteg[:, 1] =  1

# matrix to subtract two quarks
qdiff = np.array([1, -1])[:, None]

constraints = {
    'momrule': 1,
    'uubar'  : 2,
    'ddbar'  : 1,
    'ccbar'  : 0,
    'ssbar'  : 0,
}

#### GP OBJECT ####

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

    gp.addx(xdata, 'xdata', proc='f')

    # linear data (used for warmup fit)
    gp.addtransf({'xdata': M(gvar.mean(Mparams))}, 'data', axes=2)

    # total momentum rule
    gp.addx(xinteg, 'xmomrule', proc='primitive of xf(x)')
    gp.addtransf({'xmomrule': suminteg}, 'momrule', axes=2)

    # quark sum rules
    for quark in 'ducs':
        idx = indices[quark] # [quark, antiquark] indices
        label = f'{quark}{quark}bar' # the one appearing in `constraints`
        xlabel = f'x{label}'
        gp.addx(xinteg[idx], xlabel, proc='primitive')
        gp.addtransf({xlabel: suminteg[idx] * qdiff}, label, axes=2)
    
    return gp
    
#### NONLINEAR FUNCTION ####

def fcn(params):
    xdata = params['xdata']
    Mparams = params['Mparams']
    M2params = params['M2params']
    
    data = np.tensordot(M(Mparams), xdata, 2)
    
    # data2 = np.einsum('dfxy,fx,fy->d', M2, xdata, xdata)
    # np.einsum does not work with gvar
    xdata2 = xdata[:, None, :nx2] * xdata[:, :nx2, None]
    data2 = np.tensordot(M2(M2params), xdata2, 3)
    
    return dict(data=data, data2=data2)

def makeprior(gp):
    prior = gp.predfromdata(constraints, ['xdata'])
    prior['Mparams'] = Mparams
    prior['M2params'] = M2params
    return prior

#### FAKE DATA ####

truehp = gvar.sample(hyperprior)
truegp = makegp(truehp)
trueparams = gvar.sample(makeprior(truegp))
truedata = fcn(trueparams)

dataerr = {
    k: np.full_like(v, 0.1 * (np.max(v) - np.min(v)))
    for k, v in truedata.items()
}
data = gvar.make_fake_data(gvar.gvar(truedata, dataerr))

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
check_integrals(xdata['x'], trueparams['xdata'])

#### FIT ####

# find the minimization starting point with a simplified version of the fit
easyfit = truegp.predfromdata(dict(data=data['data'], **constraints), ['xdata'])
p0 = gvar.mean(easyfit)

hyperprior = gvar.BufferDict(hyperprior)
hypermean = gvar.mean(hyperprior.buf)
hypercov = gvar.evalcov(hyperprior.buf)
hyperchol = linalg.cholesky(hypercov)

i = 0
lasttime = time.time()
def analyzer(hp):
    global i, lasttime
    i += 1
    now = time.time()
    interval = now - lasttime
    print(f'iteration {i:3d} ({interval:#.2g} s): {hp}')
    lasttime = now

def fitargs(hp):
    analyzer(hp)
    gp = makegp(hp)
    prior = makeprior(gp)
    args = dict(
        data = data,
        fcn = fcn,
        prior = prior,
        p0 = p0,
    )
    residuals = linalg.solve_triangular(hyperchol, hp.buf - hypermean)
    plausibility = -1/2 * (residuals @ residuals)
    return args, plausibility

z0 = gvar.mean(hyperprior)
fit, fithp = lsqfit.empbayes_fit(z0, fitargs)
print(fit.format(maxline=True, pstyle='v'))
print(fit.format(maxline=-1))

gp = makegp(fithp)
prior = makeprior(gp)
pred = fcn(fit.p)

print('check integrals in fit:')
check_integrals(xdata['x'], fit.p['xdata'])

def allkeys(d):
    for k in d:
        yield k
        m = d.extension_pattern.match(k)
        if m and d.has_distribution(m.group(1)):
            yield m.group(2)

print('\nhyperparameters (true, fitted):')
for k in allkeys(fithp):
    print(f'{k}: {truehp[k]:#.2g}\t{fithp[k]:#.2g}')

#### PLOT RESULTS ####

fig, axs = plt.subplots(2, 3, num='pdf7', clear=True, figsize=[13, 8])
axs[0, 0].set_title('PDFs')
axs[1, 0].set_title('PDFs')
axs[0, 1].set_title('Data')
axs[1, 1].set_title('Data (quadratic)')
axs[0, 2].set_title('M parameters')
axs[1, 2].set_title('M2 parameters')

ax = axs[0, 0]

for i in range(nflav):
    
    if i >= 4:
        ax = axs[1, 0]
    
    x = xdata[i]['x']
    ypdf = fit.p['xdata'][i]
    ydata = trueparams['xdata'][i]
    m = gvar.mean(ypdf)
    s = gvar.sdev(ypdf)
    
    color = 'C' + str(i // 2)
    if i % 2:
        kw = dict(hatch='//////', edgecolor=color, facecolor='none')
        kwp = dict(color=color, linestyle='--')
    else:
        kw = dict(alpha=0.6, facecolor=color)
        kwp = dict(color=color)
    
    ax.fill_between(x, m - s, m + s, label=name[i], **kw)
    ax.plot(x, ydata, **kwp)

for ax in axs[:, 0]:
    ax.legend(fontsize='small')

for ax, label in zip(axs[:, 1], ['data', 'data2']):
    d = pred[label]
    m = gvar.mean(d)
    s = gvar.sdev(d)
    x = np.arange(len(m))
    ax.fill_between(x, m - s, m + s, step='mid', color='lightgray', label='fit')
    d = data[label]
    ax.errorbar(x, gvar.mean(d), gvar.sdev(d), color='black', linestyle='', capsize=2, label='data')
    ax.plot(x, truedata[label], drawstyle='steps-mid', color='black', label='truth')

for ax, label in zip(axs[:, 2], ['Mparams', 'M2params']):
    p = fit.p[label]
    m = gvar.mean(p)
    s = gvar.sdev(p)
    x = np.arange(len(m))
    ax.fill_between(x, m - s, m + s, step='mid', color='lightgray', label='fit')
    p = prior[label]
    ax.errorbar(x, gvar.mean(p), gvar.sdev(p), color='black', linestyle='', capsize=2, label='data')
    ax.plot(x, trueparams[label], drawstyle='steps-mid', color='black', label='truth')

for ax in axs[:, 1:].flat:
    ax.legend()

fig.tight_layout()
fig.show()
