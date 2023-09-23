"""Fit of parton distributions functions (PDFs)

Like pdf9, but with per-function hyperparameters"""

import lsqfitgp as lgp
import numpy as np
from jax import numpy as jnp
from scipy import linalg, interpolate
from matplotlib import pyplot as plt, gridspec
import gvar

# set global random seeds
seed = np.random.SeedSequence([2023, 9, 23, 12, 55])
s1, s2 = seed.spawn(2)
np.random.seed(s1.generate_state(1))
gvar.ranseed(s2.generate_state(1))

#### DEFINITIONS ####

ndata = 3000 # number of datapoints

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

#### GAUSSIAN PROCESS ####
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
hyperprior = lgp.copula.makedict({
    # correlation length of the priors at x = 1
    'scale_T3':    lgp.copula.lognorm(np.log(0.5), 0.4),
    'scale_T8':    lgp.copula.lognorm(np.log(0.5), 0.4),
    'scale_T15':   lgp.copula.lognorm(np.log(0.5), 0.4),
    'scale_V':     lgp.copula.lognorm(np.log(0.5), 0.4),
    'scale_V3':    lgp.copula.lognorm(np.log(0.5), 0.4),
    'scale_V8':    lgp.copula.lognorm(np.log(0.5), 0.4),
    'scale_V15':   lgp.copula.lognorm(np.log(0.5), 0.4),
    'scale_Sigma': lgp.copula.lognorm(np.log(0.5), 0.4),
    'scale_g':     lgp.copula.lognorm(np.log(0.5), 0.4),
    # prior variances of primitives
    'sigma_T3':    lgp.copula.lognorm(np.log(0.5), 0.4),
    'sigma_T8':    lgp.copula.lognorm(np.log(0.5), 0.4),
    'sigma_T15':   lgp.copula.lognorm(np.log(0.5), 0.4),
    'sigma_V':     lgp.copula.lognorm(np.log(0.5), 0.4),
    'sigma_V3':    lgp.copula.lognorm(np.log(0.5), 0.4),
    'sigma_V8':    lgp.copula.lognorm(np.log(0.5), 0.4),
    'sigma_V15':   lgp.copula.lognorm(np.log(0.5), 0.4),
    'sigma_Sigma': lgp.copula.lognorm(np.log(0.5), 0.4),
    'sigma_g':     lgp.copula.lognorm(np.log(0.5), 0.4),
    # exponents of x Sigma(x) and x g(x) for x -> 0
    'alpha_Sigma': lgp.copula.uniform(-0.5, 0.5),
    'alpha_g'    : lgp.copula.uniform(-0.5, 0.5),
})

def makegp(hp):
    gp = lgp.GP(checkpos=False, checksym=False, solver='chol', epsabs=1e-10, epsrel=0)
    
    def kernel(scale, *, prim=False):
        eps = grid[0]
        scalefun = lambda x: scale * (x + eps) # = 1 / log'(x)
        kernel = lgp.Gibbs(scalefun=scalefun)
        if prim:
            kernel = kernel.linop('rescale', scalefun)
        return kernel
    
    # define Ts and Vs
    for suffix in ['', '3', '8', '15']:
        if suffix != '':
            var = hp['sigma_T' + suffix] ** 2
            gp = gp.defproc('T' + suffix, var * kernel(hp['scale_T' + suffix]))
        var = hp['sigma_V' + suffix] ** 2
        gp = gp.defproc('f' + suffix, var * kernel(hp['scale_V' + suffix], prim=True))
        gp = gp.defderiv('V' + suffix, 1, 'f' + suffix)
    
    # define xSigma
    var = hp['sigma_Sigma'] ** 2
    gp = gp.defproc('f1', var * kernel(hp['scale_Sigma']))
    a = hp['alpha_Sigma']
    gp = gp.defrescale('tf1', lambda x: x ** (a + 1) / (a + 2), 'f1')
    gp = gp.defderiv('xSigma', 1, 'tf1')
    
    # define xg
    var = hp['sigma_g'] ** 2
    gp = gp.defproc('f2', var * kernel(hp['scale_g']))
    b = hp['alpha_g']
    gp = gp.defrescale('tf2', lambda x: x ** (b + 1) / (b + 2), 'f2')
    gp = gp.defderiv('xg', 1, 'tf2')
    
    # define primitive of xSigma + xg
    gp = gp.deftransf('tf12', {'tf1': 1, 'tf2': 1})
    
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

#### FAKE DATA ####

truehp = gvar.sample(hyperprior)

# rescale M to avoid having data depend almost uniquely on divergent functions
M[:, 0, :] /= datagrid ** truehp['alpha_Sigma']
M[:, -1, :] /= datagrid ** truehp['alpha_g']

truegp = makegp(truehp)
trueprior, trueprior_cov = truegp.predfromdata(constraints, ['datalatent', 'plotgrid'], raw=True)
# no gvars because it's slow with >1000 datapoints
truedata = lgp.sample(trueprior, trueprior_cov, eps=1e-10)

v = truedata['datalatent']
dataerr = np.full_like(v, 0.1 * (np.max(v) - np.min(v)))
data = gvar.make_fake_data(gvar.gvar(v, dataerr))
dataerrcov = gvar.evalcov(data)
datamean = gvar.mean(data)

def check_constraints(y):
    # integrate approximately with trapezoid rule
    integ = np.sum((y[:, 1:] + y[:, :-1]) / 2 * np.diff(plotgrid), 1)
    print(f'int dx x (Sigma(x) + g(x)) = {integ[0] + integ[-1]:.2g}')
    for i in range(1, 5):
        print(f'int dx {tpnames[i]}(x) = {integ[i]:.2g}')
    for i, name in enumerate(tpnames):
        print(f'{name}(1) = {y[i, -1]:.2g}')
    print()

print('\ncheck constraints in fake data:')
check_constraints(truedata['plotgrid'])

#### FIT ####

information = gvar.gvar(dict(datalatent=data, **constraints))

fitkw = dict(
    forward=True,
    minkw=dict(method='l-bfgs-b'),
    raises=False,
    method='gradient',
    verbosity=3,
    # covariance='fisher',
    # initial=truehp,
    # fix={'alpha_Sigma': True, 'alpha_g': True},
)
fit = lgp.empbayes_fit(hyperprior, makegp, information, **fitkw)

gp = makegp(gvar.mean(fit.p))
pred, predcov = gp.predfromdata(information, ['datalatent', 'plotgrid'], raw=True)
# use raw because with gvars it becomes slow above ~1000 datapoints

print('\ncheck constraints in fit:')
check_constraints(pred['plotgrid'])

#### PLOT RESULTS ####

legkw = dict(loc='best', title_fontsize='large')
figkw = dict(figsize=[11, 8], clear=True)#, layout='constrained')

plt.close('all')
figa = plt.figure(num='pdf10-a', **figkw)
figb = plt.figure(num='pdf10-b', **figkw)
figc = plt.figure(num='pdf10-c', **figkw)
dataax = figa.add_subplot(211)
axs = [figa.add_subplot(212)]
axs += [
    figb.add_subplot(211, sharex=axs[0]),
    figb.add_subplot(212, sharex=axs[0]),
]
hypax = figc.add_subplot()

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

for ax in axs:
    ax.axvline(datagrid[0], linestyle='--', color='black')
    ax.set_xscale('log')

if np.max(np.abs(axs[0].get_ylim())) > 10:
    axs[0].set_yscale('symlog', linthresh=10, subs=[2, 3, 4, 5, 6, 7, 8, 9])

for ax in axs:
    ax.set_xlabel('x')
    ax.set_ylabel('PDF(x)')
    
for ax in axs:
    ax.legend(title='PDFs', **legkw)

ax = dataax
ax.set(xlabel='Datapoint index', ylabel='Datapoint value')

zero = truedata['datalatent']
x = np.arange(len(zero))

# decimate the data to be displayed
sl = np.s_[::len(x) // 250 + 1]
zero = zero[sl]
x = x[sl]

ax.plot(x, truedata['datalatent'][sl] - zero, drawstyle='steps-mid', color='black', label='truth')

d = datamean[sl] - zero
ax.errorbar(x, d, dataerr[sl], color='black', linestyle='', linewidth=1, capsize=2, label='data')

d = pred['datalatent'][sl] - zero
dcov = predcov['datalatent', 'datalatent'][sl, sl]
m = d
s = np.sqrt(np.diag(dcov))
ax.fill_between(x, m - s, m + s, step='mid', color='gray', alpha=0.8, label='fit', zorder=10)

ax.legend(title='Data', **legkw)

ax = hypax
ax.set(ylabel='Hyperparameter name', xlabel='Transformed hyperparameter value (standard Normal prior)')

x = list(range(len(hyperprior)))
keys = [
    hyperprior.extension_pattern.fullmatch(k).group(2)
    for k in hyperprior.keys()
]
yprior = list(hyperprior.values())
ypost = list(fit.p.values())
ytrue = list(truehp.values())

ax.set_yticks(x)
ax.set_yticklabels(keys)

m = gvar.mean(yprior)
s = gvar.sdev(yprior)
ax.fill_betweenx(x, m - s, m + s, label='prior ($\\pm 1\\sigma$)', color='lightgray')
ax.errorbar(gvar.mean(ypost), x, xerr=gvar.sdev(ypost), label='posterior ($\\pm 1\\sigma$)', color='black', linestyle='', capsize=3, marker='.')
ax.plot(ytrue, x, 'x', label='true value', color='red')

ax.legend(title='Hyperparameters', **legkw)
ax.grid(linestyle=':', axis='y')

figa.show()
figb.show()
figc.show()
