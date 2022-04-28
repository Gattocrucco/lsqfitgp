"""Fit of parton distributions functions (PDFs)

Like pdf7, but with more realistic PDFs"""

import time

import lsqfitgp as lgp
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
import gvar
import lsqfit

np.random.seed(20220416)

#### DEFINITIONS ####

qtopm = np.array([
    #  d, db,  u, ub,  s, sb,  c, cb
    [  1,  1,  0,  0,  0,  0,  0,  0], # d+ = d + dbar
    [  1, -1,  0,  0,  0,  0,  0,  0], # d- = d - dbar
    [  0,  0,  1,  1,  0,  0,  0,  0], # u+ = etc.
    [  0,  0,  1, -1,  0,  0,  0,  0], # u-
    [  0,  0,  0,  0,  1,  1,  0,  0], # s+
    [  0,  0,  0,  0,  1, -1,  0,  0], # s-
    [  0,  0,  0,  0,  0,  0,  1,  1], # c+
    [  0,  0,  0,  0,  0,  0,  1, -1], # c-
])

pmtoev = np.array([
    # d+, d-, u+, u-, s+, s-, c+, c-
    [  1,  0,  1,  0,  1,  0,  1,  0], # Sigma = sum_q q+
    [  0,  1,  0,  1,  0,  1,  0,  1], # V     = sum_q q-
    [  0, -1,  0,  1,  0,  0,  0,  0], # V3    = u- - d-
    [  0,  1,  0,  1,  0, -2,  0,  0], # V8    = u- + d- - 2s-
    [  0,  1,  0,  1,  0,  1,  0, -3], # V15   = u- + d- + s- - 3c-
    [ -1,  0,  1,  0,  0,  0,  0,  0], # T3    = u+ - d+
    [  1,  0,  1,  0, -2,  0,  0,  0], # T8    = u+ + d+ - 2s+
    [  1,  0,  1,  0,  1,  0, -3,  0], # T15   = u+ + d+ + s+ - 3c+
])

qnames  = ['d' , 'dbar', 'u' , 'ubar', 's' , 'sbar', 'c' , 'cbar']
pmnames = ['d+', 'd-'  , 'u+', 'u-'  , 's+', 's-'  , 'c+', 'c-'  ]
evnames = ['Sigma', 'V', 'V3', 'V8', 'V15', 'T3', 'T8', 'T15']

nflav = 9

# linear data
ndata     = 10 # number of datapoints
rankmcov  =  9 # rank of the covariance matrix of the theory error

# quadratic data
ndata2    = 10
rankmcov2 =  9

grid = np.array([
    1.9999999999999954e-07, # start logspace
    3.034304765867952e-07,
    4.6035014748963906e-07,
    6.984208530700364e-07,
    1.0596094959101024e-06,
    1.607585498470808e-06,
    2.438943292891682e-06,
    3.7002272069854957e-06,
    5.613757716930151e-06,
    8.516806677573355e-06,
    1.292101569074731e-05,
    1.9602505002391748e-05,
    2.97384953722449e-05,
    4.511438394964044e-05,
    6.843744918967896e-05,
    0.00010381172986576898,
    0.00015745605600841445,
    0.00023878782918561914,
    0.00036205449638139736,
    0.0005487795323670796,
    0.0008314068836488144,
    0.0012586797144272762,
    0.0019034634022867384,
    0.0028738675812817515,
    0.004328500638820811,
    0.006496206194633799,
    0.009699159574043398,
    0.014375068581090129,
    0.02108918668378717,
    0.030521584007828916,
    0.04341491741702269,
    0.060480028754447364,
    0.08228122126204893,
    0.10914375746330703, # end logspace, start linspace
    0.14112080644440345,
    0.17802566042569432,
    0.2195041265003886,
    0.2651137041582823,
    0.31438740076927585,
    0.3668753186482242,
    0.4221667753589648,
    0.4798989029610255,
    0.5397572337880445,
    0.601472197967335,
    0.6648139482473823,
    0.7295868442414312,
    0.7956242522922756,
    0.8627839323906108,
    0.9309440808717544,
    1, # end linspace
])

nx = len(grid)

#### MODEL ####
# f1 ~ GP
# Sigma(x) = f1'(x) / x
# f2 ~ GP
# g(x) = f2'(x) / x
# f12 = f1 + f2
# f12(0) - f12(1) = 1
# fi ~ GP
# Vi = fi'
# f(1) - f(0) = 3
# f3(1) - f3(0) = 1
# f8(1) - f8(0) = 3
# f15(1) - f15(0) = 3
# Ti ~ GP
# [Sigma, g, V*, T*](1) = 0

# matrix to stack processes
stackgrid = np.einsum('ab,ij->abij', np.eye(nflav), np.eye(nx))

# transformation from evolution to flavor basis
evtoq = linalg.inv(pmtoev @ qtopm)

hyperprior = {
    'log(scale)': np.log(gvar.gvar(0.5, 0.5)),
}

def makegp(hp):
    gp = lgp.GP()

    kernel = lgp.ExpQuad(scale=hp['scale'])
    
    # define evolution basis PDFs (and their primitives)
    gp.addproc(kernel, 'f1')
    gp.addproctransf({'f1': 1}, "f1'", deriv=1)
    gp.addproctransf({"f1'": lambda x: 1/x}, 'Sigma')
    gp.addproc(kernel, 'f2')
    gp.addproctransf({'f2': 1}, "f2'", deriv=1)
    gp.addproctransf({"f2'": lambda x: 1/x}, 'g')
    gp.addproctransf({'f1': 1, 'f2': 1}, 'f12')
    for suffix in ['', '3', '8', '15']:
        gp.addproc(kernel, 'f' + suffix)
        gp.addproctransf({'f' + suffix: 1}, 'V' + suffix, deriv=1)
        if suffix != '':
            gp.addproc(kernel, 'T' + suffix)
    
    # define flavor basis PDFs
    for qi, qproc in enumerate(qnames):
        gp.addproctransf({
            eproc: evtoq[qi, ei]
            for ei, eproc in enumerate(evnames)
        }, qproc)
    
    # define a matrix of PDF values over the x grid
    for proc in evnames + ['g']:
        gp.addx(grid, proc + '-grid', proc=proc)
    gp.addtransf({
        proc + '-grid': stackgrid[i]
        for i, proc in enumerate(evnames + ['g'])
    }, 'grid', 1)

    # linear data (used for warmup fit)
    global M_mean
    gp.addtransf({'grid': M_mean}, 'data', axes=2)

    # definite integrals
    for proc in ['f12', 'f', 'f3', 'f8', 'f15']:
        gp.addx([0, 1], proc + '-endpoints', proc=proc)
        gp.addtransf({proc + '-endpoints': [-1, 1]}, proc + '-diff')
    
    # right endpoint
    for proc in evnames + ['g']:
        gp.addx(1, f'{proc}(1)', proc=proc)
    
    return gp

constraints = {
    'f12-diff': 1,
    'f-diff'  : 3,
    'f3-diff' : 1,
    'f8-diff' : 3,
    'f15-diff': 3,
    'Sigma(1)': 0,
    'V(1)'    : 0,
    'V3(1)'   : 0,
    'V8(1)'   : 0,
    'V15(1)'  : 0,
    'T3(1)'   : 0,
    'T8(1)'   : 0,
    'T15(1)'  : 0,
    'g(1)'    : 0,
}
    
#### NONLINEAR FUNCTION ####

# linear map PDF(xdata) -> data
Mcomps = np.random.randn(rankmcov, ndata, nflav, nx)
Mcomps /= np.sqrt(Mcomps.size / ndata)
Mparams = gvar.gvar(np.random.randn(rankmcov), np.full(rankmcov, 0.1))
M = lambda params: np.tensordot(params, Mcomps, 1)

M_mean = M(gvar.mean(Mparams)) # used in makegp()

# quadratic map PDF(xdata) -> data2
M2comps = np.random.randn(rankmcov2, ndata2, nflav, nx, nx)
M2comps /= 2 * np.sqrt(M2comps.size / ndata2)
M2comps = (M2comps + np.swapaxes(M2comps, -1, -2)) / 2
M2params = gvar.gvar(np.random.randn(rankmcov2), np.full(rankmcov2, 0.1))
M2 = lambda params: np.tensordot(params, M2comps, 1)

def fcn(params):
    grid = params['grid']
    Mparams = params['Mparams']
    M2params = params['M2params']
    
    data = np.tensordot(M(Mparams), grid, 2)
    
    # data2 = np.einsum('dfxy,fx,fy->d', M2, grid, grid)
    # np.einsum does not work with gvar
    grid2 = grid[:, None, :] * grid[:, :, None]
    data2 = np.tensordot(M2(M2params), grid2, 3)
    
    return dict(data=data, data2=data2)

def makeprior(gp):
    prior = gp.predfromdata(constraints, ['grid'])
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
def check_constraints(y):
    x = grid
    integ = np.sum((y[:, 1:] + y[:, :-1]) / 2 * np.diff(x), 1)
    xinteg = np.sum(((y * x)[:, 1:] + (y * x)[:, :-1]) / 2 * np.diff(x), 1)
    print('int dx x (Sigma(x) + g(x)) =', xinteg[0] + xinteg[-1])
    for i in range(1, 5):
        print(f'int dx {evnames[i]}(x) =', integ[i])
    for i, name in enumerate(evnames + ['g']):
        print(f'{name}(1) =', y[i, -1])

print('\ncheck constraints in fake data:')
check_constraints(trueparams['grid'])

#### FIT ####

# find the minimization starting point with a simplified version of the fit
easyfit = truegp.predfromdata(dict(data=data['data'], **constraints), ['grid'])
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

print('\ncheck constraints in fit:')
check_constraints(fit.p['grid'])

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

fig, axs = plt.subplots(2, 3, num='pdf8', clear=True, figsize=[13, 8])
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
    
    x = grid
    ypdf = fit.p['grid'][i]
    ydata = trueparams['grid'][i]
    m = gvar.mean(ypdf)
    s = gvar.sdev(ypdf)
    
    color = 'C' + str(i // 2)
    if i % 2:
        kw = dict(hatch='//////', edgecolor=color, facecolor='none')
        kwp = dict(color=color, linestyle='--')
    else:
        kw = dict(alpha=0.6, facecolor=color)
        kwp = dict(color=color)
    
    ax.fill_between(x, m - s, m + s, label=(evnames + ['g'])[i], **kw)
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
