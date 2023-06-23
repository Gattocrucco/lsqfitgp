import pprint
import pathlib

import polars as pl
import lsqfitgp as lgp
import numpy as np
import gvar
from scipy import stats
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

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

# load data
print('load data...')
df_p = pl.read_csv(datapath / 'track2_20220404' / 'practice' / f'acic_practice_{dataset:04d}.csv')
df_py = pl.read_csv(datapath / 'track2_20220404' / 'practice_year' / f'acic_practice_year_{dataset:04d}.csv')
df = df_p.join(df_py, on='id.practice')

# drop data to keep the script fast
df = df.filter(pl.col('id.practice').is_in(pl.col('id.practice').unique().sample(250, seed=20230623)))

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
            c for c in df.columns if c.startswith('V') or c in ['Y', 'id.practice']
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
fit_outcome = lgp.bayestree.bart(Xobs, y, weights=npatients_obs)
print(fit_outcome)

# fit treatment with linear probability modeling on the probit scale
print('\nfit treatment...')
X = Xobs.drop('Z')
z = Xobs['Z'].cast(pl.Boolean).to_numpy()
p = 1 / len(z)
z_prob = np.where(z, 1 - p, p)
z_continuous = stats.norm.ppf(z_prob)
fit_treatment = lgp.bayestree.bart(X, z_continuous, weights=npatients_obs)
print(fit_treatment)

# compute propensity score using the probit integral:
# P(z) = int dx Phi(x) N(x; mu, sigma) = Phi(mu / sqrt(1 + sigma^2))
gp = fit_treatment.gp(x_test=X, weights=npatients_obs)
mu, Sigma = gp.predfromdata(fit_treatment.info, 'test', raw=True)
mu += fit_treatment.mu
sigma2 = np.diag(Sigma)
ps = stats.norm.cdf(mu / np.sqrt(1 + sigma2))

# fit outcome with propensity score as covariate
print('\nfit outcome (w/ PS)...')
Xobs_ps = Xobs.with_columns(ps=pl.lit(ps))
Xmis_ps = (Xobs_ps
    .filter(pl.col('Z') == 1)
    .with_columns(Z=1 - pl.col('Z'))
)
fit_outcome_ps = lgp.bayestree.bart(Xobs_ps, y, weights=npatients_obs)
print(fit_outcome_ps)

def compute_satt(fit, Xmis):

    # create GP at MAP hyperparameters and get imputed outcomes for the treated
    gp = fit.gp(x_test=Xmis, weights=npatients_mis)
    ymis = fit.mu + gp.predfromdata(fit.info, 'test', keepcorr=False)

    # compute effects on the treated
    yobs = y[z]
    n = npatients_obs[z]
    effect = yobs - ymis
    strata = (df
        .filter((pl.col('Z') == 1) & (pl.col('post') == 1))
        .select([f'X{i}' for i in range(1, 6)] + ['year'])
        .rename({'year': 'Yearly'})
        .with_row_count('index')
    )
    satt = {}
    satt['Overall'] = np.average(effect, weights=n)
    for variable in strata.columns:
        if variable == 'index':
            continue
        subdict = {}
        for level, stratum in strata.groupby(variable):
            indices = stratum['index'].to_numpy()
            subdict[level] = np.average(effect[indices], weights=n[indices])
        satt[variable] = {k: subdict[k] for k in sorted(subdict)}

    return satt

satt = compute_satt(fit_outcome, Xmis)
satt_ps = compute_satt(fit_outcome_ps, Xmis_ps)

# print results
print(f'\nATE (unadjusted) = {ate_unadjusted}')
print(f'\nSATT (no PS) =\n{pprint.pformat(satt)}')
print(f'\nSATT (w/ PS) =\n{pprint.pformat(satt_ps)}')

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

# show truth
satt_true = {}
for variable, group in df_results.groupby('variable'):
    if len(group) > 1:
        for row in group.iter_rows(named=True):
            level = row['level']
            if level.isnumeric():
                level = int(level)
            satt_true.setdefault(variable, {})[level] = row['SATT']
    else:
        satt_true[variable] = group['SATT'][0]
print(f'\nSATT (truth) =\n{pprint.pformat(satt_true)}')

# plot comparison with truth
fig, ax = plt.subplots(num='acic', clear=True, layout='constrained')
labels = []
tick = 0
for variable, estimate in satt_ps.items():
    truth = satt_true[variable]
    if isinstance(estimate, dict):
        for level, estimate in estimate.items():
            truth = satt_true[variable][level]
            artist_estimate = ax.errorbar(gvar.mean(estimate), tick,
                xerr=gvar.sdev(estimate), fmt='.k', capsize=3,
                label='Estimate ($\\pm$1 sd)')
            artist_truth, = ax.plot(truth, tick, 'rx', label='Truth')
            labels.append(f'{variable}={level}')
            tick -= 1
    else:
        ax.errorbar(gvar.mean(estimate), tick, xerr=gvar.sdev(estimate), fmt='.k', capsize=3)
        ax.plot(truth, tick, 'rx')
        labels.append(f'{variable}')
        tick -= 1
ax.set(
    xlabel='SATT',
    ylabel='Stratum',
    yticks=np.arange(0, tick, -1),
    yticklabels=labels,
)
ax.legend(handles=[artist_estimate, artist_truth], loc='upper right')
fig.show()
