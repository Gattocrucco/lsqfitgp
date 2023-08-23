"""Fit of parton distributions functions (PDFs)

Like pdf7, but with more realistic PDFs"""

import time

import lsqfitgp as lgp
import numpy as np
from scipy import linalg, interpolate
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
pnames  = evnames + ['g']
tpnames = ['xSigma'] + evnames[1:] + ['xg']

nflav = len(pnames)

# linear data
ndata     = 10 # number of datapoints
rankmcov  =  9 # rank of the covariance matrix of the theory error

# quadratic data
ndata2    = 10
rankmcov2 =  9

# grid used for DGLAP evolution
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

# grid used for data
datagrid = grid[:-1] # exclude 1 since f(1) = 0 and zero errors upset the fit
nx = len(datagrid)

# grid used for plot
gridinterp = interpolate.interp1d(np.linspace(0, 1, len(grid)), grid)
plotgrid = gridinterp(np.linspace(0, 1, 200))

#### GAUSSIAN PROCESS ####
# Ti ~ GP
#
# fi ~ GP with sdev ~ x to compensate the scale ~ 1/x
# Vi = fi'
# f(1) - f(0) = 3
# f3(1) - f3(0) = 1
# f8(1) - f8(0) = 3
# f15(1) - f15(0) = 3
#
# f1 ~ GP   (WITHOUT scale compensation)
# tf1(x) = x^(a+1)/(a+2) f1(x)   <--- scale comp. is x^(a+1) instead of x^a,
#                                     to avoid doing x^a with a < 0 in x = 0
# Sigma(x) = tf1'(x) / x    (such that x Sigma(x) ~ x^a)
# the same with f2, tf2, g
# tf12 = tf1 + tf2
# tf12(0) - tf12(1) = 1
#
# [Sigma, g, V*, T*](1) = 0

# matrix to stack processes
stackplotgrid = np.einsum('ab,ij->abij', np.eye(nflav), np.eye(len(plotgrid)))
stackdatagrid = np.einsum('ab,ij->abij', np.eye(nflav), np.eye(len(datagrid)))

# transformation from evolution to flavor basis
evtoq = linalg.inv(pmtoev @ qtopm)

hyperprior = lgp.copula.makedict({
    'log(scale)' : np.log(gvar.gvar(0.5, 0.5)),
    'alpha_Sigma': lgp.copula.uniform(-0.5, 0.5),
    'alpha_g'    : lgp.copula.uniform(-0.5, 0.5),
})

def makegp(hp, quick=False):
    gp = lgp.GP(checkpos=False)
    
    eps = grid[0]
    scalefun = lambda x: hp['scale'] * (x + eps)
    kernel = lgp.Gibbs(scalefun=scalefun)
    kernel_prim = kernel.rescale(scalefun, scalefun)
    
    # define Ts and Vs
    for suffix in ['', '3', '8', '15']:
        if suffix != '':
            gp.defproc('T' + suffix, kernel)
        gp.defproc('f' + suffix, kernel_prim)
        gp.defproctransf('V' + suffix, {'f' + suffix: 1}, deriv=1)
    
    # define xSigma
    gp.defproc('f1', kernel)
    a = hp['alpha_Sigma']
    gp.defproctransf('tf1', {'f1': lambda x: x ** (a + 1) / (a + 2)})
    gp.defproctransf('xSigma', {'tf1': 1}, deriv=1)
    
    # define xg
    gp.defproc('f2', kernel)
    b = hp['alpha_g']
    gp.defproctransf('tf2', {'f2': lambda x: x ** (b + 1) / (b + 2)})
    gp.defproctransf('xg', {'tf2': 1}, deriv=1)
    
    # define primitive of xSigma + xg
    gp.defproctransf('tf12', {'tf1': 1, 'tf2': 1})
    
    # define a matrix of PDF values over the x grid
    for proc in tpnames:
        gp.addx(datagrid, proc + '-datagrid', proc=proc)
    gp.addtransf({
        proc + '-datagrid': stackdatagrid[i]
        for i, proc in enumerate(tpnames)
    }, 'datagrid', 1)

    # definite integrals
    for proc in ['tf12', 'f', 'f3', 'f8', 'f15']:
        gp.addx([0, 1], proc + '-endpoints', proc=proc)
        gp.addtransf({proc + '-endpoints': [-1, 1]}, proc + '-diff')
    
    # right endpoint
    for proc in tpnames:
        gp.addx(1, f'{proc}(1)', proc=proc)
    
    if not quick:
        
        # linear data (used for warmup fit)
        global M_mean
        gp.addtransf({'datagrid': M_mean}, 'data', axes=2)
    
        # define flavor basis PDFs
        gp.defproctransf('Sigma', {'xSigma': lambda x: 1 / x})
        gp.defproctransf('g', {'xg': lambda x: 1 / x})
        for qi, qproc in enumerate(qnames):
            gp.defproctransf(qproc, {
                eproc: evtoq[qi, ei]
                for ei, eproc in enumerate(evnames)
            })
    
        # define a matrix of PDF values over the plot grid
        for proc in tpnames:
            gp.addx(plotgrid, proc + '-plotgrid', proc=proc)
        gp.addtransf({
            proc + '-plotgrid': stackplotgrid[i]
            for i, proc in enumerate(tpnames)
        }, 'plotgrid', 1)

    return gp

constraints = {
    'tf12-diff': 1,
    'f-diff'   : 3,
    'f3-diff'  : 1,
    'f8-diff'  : 3,
    'f15-diff' : 3,
    'xSigma(1)': 0,
    'V(1)'     : 0,
    'V3(1)'    : 0,
    'V8(1)'    : 0,
    'V15(1)'   : 0,
    'T3(1)'    : 0,
    'T8(1)'    : 0,
    'T15(1)'   : 0,
    'xg(1)'    : 0,
}
    
#### NONLINEAR FUNCTION ####

# linear map PDF(grid) -> data
Mcomps = np.random.randn(rankmcov, ndata, nflav, nx)
Mcomps /= np.sqrt(Mcomps.size / ndata)
Mparams = gvar.gvar(np.random.randn(rankmcov), np.full(rankmcov, 0.1))
M = lambda params: np.tensordot(params, Mcomps, 1)

M_mean = M(gvar.mean(Mparams)) # used in makegp()

# quadratic map PDF(grid) -> data2
M2comps = np.random.randn(rankmcov2, ndata2, nflav, nx, nx)
M2comps /= 2 * np.sqrt(M2comps.size / ndata2)
M2comps = (M2comps + np.swapaxes(M2comps, -1, -2)) / 2
M2params = gvar.gvar(np.random.randn(rankmcov2), np.full(rankmcov2, 0.1))
M2 = lambda params: np.tensordot(params, M2comps, 1)

def fcn(params):
    datagrid = params['datagrid']
    Mparams = params['Mparams']
    M2params = params['M2params']
    
    data = np.tensordot(M(Mparams), datagrid, 2)
    
    # data2 = np.einsum('dfxy,fx,fy->d', M2, datagrid, datagrid)
    # np.einsum does not work with gvar
    grid2 = datagrid[:, None, :] * datagrid[:, :, None]
    data2 = np.tensordot(M2(M2params), grid2, 3)
    
    return dict(data=data, data2=data2)

def makeprior(gp, plot=False):
    out = ['datagrid']
    if plot:
        out += ['plotgrid']
    prior = gp.predfromdata(constraints, out)
    prior['Mparams'] = Mparams
    prior['M2params'] = M2params
    return prior

#### FAKE DATA ####

truehp = gvar.sample(hyperprior)
truegp = makegp(truehp)
trueprior = makeprior(truegp, plot=True)
trueparams = gvar.sample(trueprior)
truedata = fcn(trueparams)

dataerr = {
    k: np.full_like(v, 0.1 * (np.max(v) - np.min(v)))
    for k, v in truedata.items()
}
data = gvar.make_fake_data(gvar.gvar(truedata, dataerr))

def check_constraints(y):
    # integrate approximately with trapezoid rule
    integ = np.sum((y[:, 1:] + y[:, :-1]) / 2 * np.diff(plotgrid), 1)
    print('int dx x (Sigma(x) + g(x)) =', integ[0] + integ[-1])
    for i in range(1, 5):
        print(f'int dx {tpnames[i]}(x) =', integ[i])
    for i, name in enumerate(tpnames):
        print(f'{name}(1) =', y[i, -1])

print('\ncheck constraints in fake data:')
check_constraints(trueparams['plotgrid'])

#### FIT ####

# find the minimization starting point with a simplified version of the fit
meangp = makegp(gvar.mean(hyperprior))
easyfit = meangp.predfromdata(dict(data=data['data'], **constraints), ['datagrid'])
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
    gp = makegp(hp, quick=True)
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
    
print('\ntrue hyperparameters:')
print(truehp)

print('\nfit:')
z0 = gvar.sample(hyperprior)
fit, fithp = lsqfit.empbayes_fit(z0, fitargs)
print(fit.format(maxline=True, pstyle='v'))
print(fit.format(maxline=-1))

gp = makegp(fithp)
prior = makeprior(gp)
pred = fcn(fit.p)
fitgrid = gp.predfromfit(dict(datagrid=fit.p['datagrid'], **constraints), 'plotgrid')

print('\ncheck constraints in fit:')
check_constraints(fitgrid)

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

fig, axs = plt.subplots(2, 3, num='pdf8', clear=True, figsize=[10.5, 7], gridspec_kw=dict(width_ratios=[2, 1, 1]))
axs[0, 0].set_title('PDFs')
axs[1, 0].set_title('PDFs')
axs[0, 1].set_title('Data')
axs[1, 1].set_title('Data (quadratic)')
axs[0, 2].set_title('M parameters')
axs[1, 2].set_title('M2 parameters')

for i in range(nflav):
    
    if i < 5:
        ax = axs[0, 0]
    else:
        ax = axs[1, 0]
    
    ypdf = fitgrid[i]
    m = gvar.mean(ypdf)
    s = gvar.sdev(ypdf)
    ax.fill_between(plotgrid, m - s, m + s, label=tpnames[i], alpha=0.6, facecolor=f'C{i}')

    ax.plot(plotgrid, trueparams['plotgrid'][i], color=f'C{i}')

    # ax.set_xscale('log')

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
