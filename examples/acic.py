import pprint
import pathlib

import polars as pl
import lsqfitgp as lgp
import numpy as np
import gvar
from scipy import stats
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import tqdm

"""

Analyze a dataset from the ACIC 2022 Data Challenge using the BART kernel.

Website: https://acic2022.mathematica.org

Article: Dan R.C. Thal and Mariel M. Finucane, "Causal Methods Madness: Lessons
Learned from the 2022 ACIC Competition to Estimate Health Policy Impacts,"
Observational Studies, Volume 9, Issue 3, 2023, pp. 3-27,
https://doi.org/10.1353/obs.2023.0023

"""

# config
dataset = 1
datapath = pathlib.Path('examples') / 'acic'
nsamples = 100
bartkw = dict(fitkw=dict(verbosity=0))

# load data
print('load data...')
df_p = pl.read_csv(datapath / 'track2_20220404' / 'practice' / f'acic_practice_{dataset:04d}.csv')
df_py = pl.read_csv(datapath / 'track2_20220404' / 'practice_year' / f'acic_practice_year_{dataset:04d}.csv')
df = df_p.join(df_py, on='id.practice')

# drop data to keep the script fast
df = (df
    .filter((pl
        .col('id.practice')
        .is_in(pl.col('id.practice').unique().sample(250, seed=20230623))
    ))
)

# compute effect as if Z was randomized, i.e., without adjustment
print('least squares fit...')
data = (df
    .filter(pl.col('post') == 1)
    .select(['Y', 'year', 'Z', 'n.patients'])
    .to_pandas()
)
model = smf.wls('Y ~ 1 + Z', data, weights=data['n.patients'])
result = model.fit()
ate_unadjusted = gvar.gvar(result.params['Z'], result.bse['Z'])

# move pretreatment outcomes and per-year covariates to new columns: we are
# going to use standard unconfoundedness given pretreatment outcomes instead
# of the DID-like assumption given by the competition rules
posttreatment = df.filter(pl.col('post') == 1)
for year in [1, 2]:
    posttreatment = posttreatment.join(
        df.filter(pl.col('year') == year).select([
            c for c in df.columns
            if c.startswith('V') or c in ['Y', 'id.practice']
        ]), on='id.practice', suffix=f'_year{year}',
    )

# extract outcome, weights and covariates
y = posttreatment['Y'].to_numpy()
npatients_obs = posttreatment['n.patients'].to_numpy() # to scale error variance
Xobs = (posttreatment
    .drop(['Y', 'id.practice', 'post', 'n.patients'])
    .to_dummies(columns=['X2', 'X4'])
)
Xmis = (Xobs
    .filter(pl.col('Z') == 1)
    .with_columns(Z=1 - pl.col('Z'))
)
npatients_mis = posttreatment.filter(pl.col('Z') == 1)['n.patients'].to_numpy()

# fit outcome using BART as GP
print('\nfit outcome (w/o PS)...')
fit_outcome = lgp.bayestree.bart(Xobs, y, weights=npatients_obs, **bartkw)
print(fit_outcome)

# fit treatment with linear probability modeling on the probit scale
print('\nfit treatment...')
X = Xobs.drop('Z')
z = Xobs['Z'].cast(pl.Boolean).to_numpy()
p = 1 / len(z)
z_prob = np.where(z, 1 - p, p)
z_continuous = stats.norm.ppf(z_prob)
fit_treatment = lgp.bayestree.bart(X, z_continuous, weights=npatients_obs, **bartkw)
print(fit_treatment)

# compute propensity score using the probit integral:
# P(z) = int dx Phi(x) N(x; mu, sigma) = Phi(mu / sqrt(1 + sigma^2))
mu, Sigma = fit_treatment.pred(x_test=X, weights=npatients_obs, error=True)
sigma2 = np.diag(Sigma)
ps = stats.norm.cdf(mu / np.sqrt(1 + sigma2))

# fit outcome with propensity score as covariate
print('\nfit outcome (w/ PS)...')
Xobs_ps = Xobs.with_columns(ps=pl.lit(ps))
Xmis_ps = (Xobs_ps
    .filter(pl.col('Z') == 1)
    .with_columns(Z=1 - pl.col('Z'))
)
fit_outcome_ps = lgp.bayestree.bart(Xobs_ps, y, weights=npatients_obs, **bartkw)
print(fit_outcome_ps)

# define groups of units for conditional SATT
strata = (df
    .filter((pl.col('Z') == 1) & (pl.col('post') == 1))
    .select([f'X{i}' for i in range(1, 6)] + ['year'])
    .rename({'year': 'Yearly'})
    .with_row_count('index')
)

class SwitchGVar:
    """ Creating new primary gvars fills up memory permanently. This context
    manager keeps the gvars created within its context in a separate pool that
    is freed when all such gvars are deleted. They can not be mixed in
    operations with other gvars created outside of the context. """

    def __enter__(self):
        gvar.switch_gvar()

    def __exit__(self, *_):
        gvar.restore_gvar()

def compute_satt(fit, Xmis, *, rng=None):

    # create GP at MAP/sampled hypers and get imputed outcomes for the treated
    kw = dict(x_test=Xmis, weights=npatients_mis, error=True, format='gvar')
    if rng is not None:
        with SwitchGVar():
            ymis = fit.pred(**kw, hp='sample', rng=rng)
    else:
        ymis = fit.pred(**kw)

    # compute effects on the treated
    yobs = y[z]
    n = npatients_obs[z]
    effect = yobs - ymis
    satt = {}
    satt['Overall'] = np.average(effect, weights=n)
    for variable in strata.columns:
        if variable == 'index':
            continue
        for level, stratum in strata.groupby(variable):
            indices = stratum['index'].to_numpy()
            key = f'{variable}={level}'
            satt[key] = np.average(effect[indices], weights=n[indices])

    if rng is not None:
        satt = lgp.sample(gvar.mean(satt), gvar.evalcov(satt), rng=rng)

    satt = {k: satt[k] for k in sorted(satt)}
    return satt

print('\ncompute satt...')

# compute the SATT at marginal MAP hyperparameters
satt = compute_satt(fit_outcome, Xmis)
satt_ps = compute_satt(fit_outcome_ps, Xmis_ps)

# compute the SATT sampling the hyperparameters with the Laplace approx
rng = np.random.default_rng(202307081315)
satt_ps_samples = {}
for _ in tqdm.tqdm(range(nsamples)):
    satt_sample = compute_satt(fit_outcome_ps, Xmis_ps, rng=rng)
    for k, v in satt_sample.items():
        satt_ps_samples.setdefault(k, []).append(v)
satt_ps_samples_quantiles = {}
satt_ps_samples_meanstd = {}
for k, samples in satt_ps_samples.items():
    cl = np.diff(stats.norm.cdf([-1, 1])).item()
    satt_ps_samples_quantiles[k] = np.quantile(samples, [(1 - cl) / 2, 0.5, (1 + cl) / 2])
    satt_ps_samples_meanstd[k] = gvar.gvar(np.mean(samples), np.std(samples))

# print results
print(f'\nATE (unadjusted) = {ate_unadjusted}')
print(f'\nSATT (no PS, MAP) =\n{pprint.pformat(satt)}')
print(f'\nSATT (w/ PS, MAP) =\n{pprint.pformat(satt_ps)}')
print(f'\nSATT (w/ PS, Laplace) =\n{pprint.pformat(satt_ps_samples_meanstd)}')

# load actual true effect
df_results = (pl
    .read_csv(datapath / 'results' / 'ACIC_estimand_truths.csv', null_values='NA')
    .filter(pl.col('dataset.num') == dataset)
    .filter(pl.col('variable').is_not_null())
    .with_columns(level=(pl
        .when(pl.col('variable') == 'Yearly')
        .then(pl.col('year'))
        .otherwise(pl.col('level'))
    ))
)

# collect and show truth
satt_true = {}
for variable, group in df_results.groupby('variable'):
    if len(group) > 1:
        for row in group.iter_rows(named=True):
            level = row['level']
            satt_true[f'{variable}={level}'] = row['SATT']
    else:
        satt_true[variable] = group['SATT'][0]
print(f'\nSATT (truth) =\n{pprint.pformat(satt_true)}')

# plot comparison with truth
fig, ax = plt.subplots(num='acic', clear=True, layout='constrained')

estimates = {
    'w/o PS, MAP': satt,
    'w/ PS, MAP': satt_ps,
    'w/ PS, Laplace': satt_ps_samples_quantiles,
}
artist_estimate = [None] * len(estimates)

for i, (label, satt_dict) in enumerate(estimates.items()):
    tick = 0
    for stratum, estimate in satt_dict.items():
        width = 0.4
        shift = -width / 2 + width * i / (len(estimates) - 1)
        if isinstance(estimate, np.ndarray):
            x = estimate[1]
            xerr = np.reshape([
                estimate[1] - estimate[0],
                estimate[2] - estimate[1]
            ], (2, 1))
        else:
            x = gvar.mean(estimate)
            xerr = gvar.sdev(estimate)
        artist_estimate[i] = ax.errorbar(x, tick - shift, xerr=xerr, fmt='.',
            capsize=3, color=f'C{i}', label=label)
        artist_truth, = ax.plot(satt_true[stratum], tick, 'kx', label='Truth')
        tick -= 1

ax.set(
    xlabel='SATT',
    ylabel='Stratum',
    yticks=np.arange(0, tick, -1),
    yticklabels=list(satt),
    title='SATT posterior (0.16, 0.50, 0.84 quantiles)',
)
ax.legend(
    handles=[artist_truth, *artist_estimate],
    loc='upper right',
)

fig.show()
