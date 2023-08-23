"""Fit of parton distributions functions (PDFs)

Like pdf8, but only with linear data"""

import warnings

import lsqfitgp as lgp
import numpy as np
from jax import numpy as jnp
from scipy import linalg, interpolate
from matplotlib import pyplot as plt, gridspec
import gvar

# set random seeds
seed = np.random.SeedSequence(202308011825)
s1, s2 = seed.spawn(2)
np.random.seed(s1.generate_state(1))
gvar.ranseed(s2.generate_state(1))

# warnings to ignore
warnings.filterwarnings('ignore', r'total derivative orders \(\d+, \d+\) greater than kernel minimum \(\d+, \d+\)')
warnings.filterwarnings('ignore', r'Attempt to set non-positive xlim on a log-scaled axis will be ignored')


#### DEFINE CONSTANTS ####

ndata = 3000 # number of datapoints

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
datagrid = grid[15:-1] # exclude 1 since f(1) = 0 and zero errors upset the fit
nx = len(datagrid)

# grid used for plot
gridinterp = interpolate.interp1d(np.linspace(0, 1, len(grid)), grid)
plotgrid = gridinterp(np.linspace(0, 1, 200))

# generated linear map PDF(grid) -> data
i = np.arange(ndata)[:, None, None]
j = np.arange(nx)[None, None, :]
intensity_diagonal = np.exp(-1/2 * (i / ndata - j / nx) ** 2 * ndata * nx)
intensity_flat = 1
intensity = 0.9 * intensity_diagonal + 0.1 * intensity_flat
intensity = np.broadcast_to(intensity, (ndata, nflav, nx))
dof = 3
M = intensity * np.random.chisquare(dof, intensity.shape) / dof


#### DEFINE GAUSSIAN PROCESS MODEL ####

# Ti ~ GP   (i = 3, 8, 15)
#
# fi ~ GP with sdev ~ x to compensate the scale ~ 1/x   (i = 3, 8, 15)
# Vi = fi'
# f(1) - f(0) = 3
# f3(1) - f3(0) = 1
# f8(1) - f8(0) = 3
# f15(1) - f15(0) = 3
#
# f1 ~ GP   (without scale compensation)
# tf1(x) = x^(a+1)/(a+2) f1(x)   <--- scale comp. is x^(a+1) instead of x^a,
#                                     to avoid doing x^a with a < 0 in x = 0
# Sigma(x) = tf1'(x) / x    (such that x Sigma(x) ~ x^a)
# the same with f2, tf2, g
# tf12 = tf1 + tf2
# tf12(0) - tf12(1) = 1
#
# [Sigma, g, V*, T*](1) = 0

# transformation from evolution to flavor basis
evtoq = linalg.inv(pmtoev @ qtopm)

hyperprior = lgp.copula.makedict({
    # correlation length of the prior at x = 1
    'log(scale)' : np.log(gvar.gvar(0.5, 0.2)),
    # exponents of x Sigma(x) and x g(x) for x -> 0
    'alpha_Sigma': lgp.copula.uniform(-0.5, 0.5),
    'alpha_g'    : lgp.copula.uniform(-0.5, 0.5),
})

makegpkw = dict(
    grid=grid,
    datagrid=datagrid,
    plotgrid=plotgrid,
    M=M,
    evtoq=evtoq,
)

def makegp(hp, **kw):

    # avoid global array variables because jax folds them in compiled code
    grid = kw['grid']
    datagrid = kw['datagrid']
    plotgrid = kw['plotgrid']
    M = kw['M']
    evtoq = kw['evtoq']

    gp = lgp.GP(checkpos=False, checksym=False, solver='chol')
    
    eps = grid[0]
    scalefun = lambda x: hp['scale'] * (x + eps) # = 1 / log'(x)
    kernel = lgp.Gibbs(scalefun=scalefun)
    kernel_prim = kernel.rescale(scalefun, scalefun)
    
    # define Ts and Vs
    for suffix in ['', '3', '8', '15']:
        if suffix != '':
            gp = gp.defproc('T' + suffix, kernel)
        gp = gp.defproc('f' + suffix, kernel_prim)
        gp = gp.defprocderiv('V' + suffix, 1, 'f' + suffix)
    
    # define xSigma
    gp = gp.defproc('f1', kernel)
    a = hp['alpha_Sigma']
    gp = gp.defprocrescale('tf1', lambda x: x ** (a + 1) / (a + 2), 'f1')
    gp = gp.defprocderiv('xSigma', 1, 'tf1')
    
    # define xg
    gp = gp.defproc('f2', kernel)
    b = hp['alpha_g']
    gp = gp.defprocrescale('tf2', lambda x: x ** (b + 1) / (b + 2), 'f2')
    gp = gp.defprocderiv('xg', 1, 'tf2')
    
    # define primitive of xSigma + xg
    gp = gp.defproctransf('tf12', {'tf1': 1, 'tf2': 1})
    
    # definite integrals
    for proc in ['tf12', 'f', 'f3', 'f8', 'f15']:
        gp = gp.addx([0, 1], proc + '-endpoints', proc=proc)
        gp = gp.addlintransf(lambda x: x[1] - x[0], [proc + '-endpoints'], proc + '-diff')
    
    # right endpoint
    for proc in tpnames:
        gp = gp.addx(1, f'{proc}(1)', proc=proc)
    
    # define a matrix of PDF values over the x grid
    for proc in tpnames:
        gp = gp.addx(datagrid, proc + '-datagrid', proc=proc)
    gp = gp.addlintransf(lambda *args: jnp.stack(args), [proc + '-datagrid' for proc in tpnames], 'datagrid')

    # linear data
    gp = gp.addtransf({'datagrid': M}, 'datalatent', axes=2)

    # define flavor basis PDFs
    gp = gp.defprocrescale('Sigma', lambda x: 1 / x, 'xSigma')
    gp = gp.defprocrescale('g', lambda x: 1 / x, 'xg')
    for qi, qproc in enumerate(qnames):
        gp = gp.defproctransf(qproc, {
            eproc: evtoq[qi, ei]
            for ei, eproc in enumerate(evnames)
        })

    # define a matrix of PDF values over the plot grid
    for proc in tpnames:
        gp = gp.addx(plotgrid, proc + '-plotgrid', proc=proc)
    gp = gp.addlintransf(lambda *args: jnp.stack(args), [proc + '-plotgrid' for proc in tpnames], 'plotgrid')

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


#### GENERATE FAKE DATA ACCORDING TO MODEL ####

truehp = gvar.sample(hyperprior)

# rescale M to avoid having data depend almost uniquely on divergent functions
M[:, 0, :] /= datagrid ** truehp['alpha_Sigma']
M[:, -1, :] /= datagrid ** truehp['alpha_g']

truegp = makegp(truehp, **makegpkw)
trueprior, trueprior_cov = truegp.predfromdata(constraints, ['datalatent', 'plotgrid'], raw=True)
# no gvars because it's slow with >1000 datapoints
truedata = lgp.sample(trueprior, trueprior_cov, eps=1e-10)

v = truedata['datalatent']
dataerr = np.full_like(v, 0.1 * (np.max(v) - np.min(v)))
data = gvar.make_fake_data(gvar.gvar(v, dataerr))

def check_constraints(y):
    # integrate approximately with trapezoid rule
    integ = np.sum((y[:, 1:] + y[:, :-1]) / 2 * np.diff(plotgrid), 1)
    print(f'int dx x (Sigma(x) + g(x)) = {integ[0] + integ[-1]:.2g}')
    for i in range(1, 5):
        print(f'int dx {tpnames[i]}(x) = {integ[i]:.2g}')
    for i, name in enumerate(tpnames):
        print(f'{name}(1) = {y[i, -1]:.2g}')

print('\ncheck constraints in fake data:')
check_constraints(truedata['plotgrid'])
print()


#### FIT HYPERPARAMETERS ####

information = gvar.gvar(dict(datalatent=data, **constraints))
fitkw = dict(
    gpfactorykw=makegpkw,
    raises=False,
    verbosity=3,
    minkw=dict(method='l-bfgs-b'),
    forward=True,
)
fit = lgp.empbayes_fit(hyperprior, makegp, information, **fitkw)

print('\nhyperparameters (true, fitted, prior):')
hyperprior = gvar.BufferDict(hyperprior)
for k in fit.p.all_keys():
    print(f'{k:15}{truehp[k]:>#10.2g}{str(fit.p[k]):>15}{str(hyperprior[k]):>15}')

gp = makegp(gvar.mean(fit.p), **makegpkw)
pred, predcov = gp.predfromdata(information, ['datalatent', 'plotgrid'], raw=True)
# use raw because with gvars it becomes slow above ~1000 datapoints

print('\ncheck constraints in fit:')
check_constraints(pred['plotgrid'])


#### PLOT RESULTS ####

fig = plt.figure(num='pdf9', figsize=[13, 16], clear=True, layout='constrained')
grid = gridspec.GridSpec(4, 2, figure=fig)
axs = [
    fig.add_subplot(grid[0, :])
]
axs += [
    fig.add_subplot(grid[1, :], sharex=axs[0]),
    fig.add_subplot(grid[2, :], sharex=axs[0]),
    fig.add_subplot(grid[3, 0]), fig.add_subplot(grid[3, 1]),
]

for i in range(nflav):
    
    label = tpnames[i]
    if label in ['xSigma', 'xg', 'V']:
        ax = axs[0]
    elif label.startswith('T'):
        ax = axs[1]
    else:
        ax = axs[2]
    
    if label.startswith('x'):
        expon = fit.p['alpha_' + label[1:]]
        label += f' $\\sim x^{{{expon}}}$'
    
    ypdf = pred['plotgrid'][i, :]
    ypdfcov = predcov['plotgrid', 'plotgrid'][i, :, i, :]
    m = ypdf
    s = np.sqrt(np.diag(ypdfcov))
    ax.fill_between(plotgrid, m - s, m + s, label=label, alpha=0.4, facecolor=f'C{i}')

    ax.plot(plotgrid, truedata['plotgrid'][i], color=f'C{i}')

    ax.set_xscale('log')

for ax in axs[:3]:
    ax.axvline(datagrid[0], linestyle='--', color='black')

axs[0].set_yscale('symlog', linthresh=10, subs=[2, 3, 4, 5, 6, 7, 8, 9])

ax = axs[3]

zero = truedata['datalatent']
x = np.arange(len(zero))

# decimate the data to be displayed
sl = np.s_[::len(x) // 250 + 1]
zero = zero[sl]
x = x[sl]

ax.plot(x, truedata['datalatent'][sl] - zero, drawstyle='steps-mid', color='black', label='truth')

d = gvar.mean(data[sl]) - zero
ax.errorbar(x, d, dataerr[sl], color='black', linestyle='', linewidth=1, capsize=2, label='data')

d = pred['datalatent'][sl] - zero
dcov = predcov['datalatent', 'datalatent'][sl, sl]
m = d
s = np.sqrt(np.diag(dcov))
ax.fill_between(x, m - s, m + s, step='mid', color='gray', alpha=0.8, label='fit', zorder=10)

ax = axs[4]

x = list(range(len(hyperprior)))
keys = list(hyperprior.keys())
yprior = list(hyperprior.values())
ypost = list(fit.p.values())
ytrue = list(truehp.values())

ax.set(xticks=x, xticklabels=keys)

m = gvar.mean(yprior)
s = gvar.sdev(yprior)
ax.fill_between(x, m - s, m + s, label='prior', color='lightgray')
ax.errorbar(x, gvar.mean(ypost), gvar.sdev(ypost),
    label='posterior', color='black', linestyle='', capsize=3, marker='.')
ax.plot(ytrue, drawstyle='steps-mid', label='true value', color='red')

legkw = dict(loc='best', title_fontsize='large')
for ax in axs[:3]:
    ax.legend(title='PDFs', **legkw)
axs[3].legend(title='Data', **legkw)
axs[4].legend(title='Hyperparameters', **legkw)

for ax in axs[:3]:
    ax.set(xlabel='x',
           ylabel='PDF(x)')
    
axs[3].set(xlabel='Datapoint index',
           ylabel='Datapoint value')

axs[4].set(xlabel='Hyperparameter name',
           ylabel='Transformed hyperparameter value')

fig.show()
