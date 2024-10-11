import pprint
import pathlib

import polars as pl
import lsqfitgp as lgp
import numpy as np
import gvar
from scipy import stats, special
import statsmodels.formula.api as smf
import statsmodels.api as sm
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
artificial_effect_shift = 0
laplace = False

# load data
print('load data...')
prefix = datapath / 'track2_20220404'
df_p = pl.read_csv(prefix / 'practice' / f'acic_practice_{dataset:04d}.csv')
df_py = pl.read_csv(prefix / 'practice_year' / f'acic_practice_year_{dataset:04d}.csv')
df = df_p.join(df_py, on='id.practice')

# shift treated units by a fixed amount for testing purposes
df = df.with_columns(
    pl.col('Y') + pl
    .when((pl.col('Z') == 1) & (pl.col('post') == 1))
    .then(artificial_effect_shift)
    .otherwise(0)
)

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
    .select(['Y', 'Z', 'n.patients'])
    .to_pandas()
)
model = smf.wls('Y ~ 1 + Z', data, weights=data['n.patients'])
result = model.fit()
ate_unadjusted = gvar.gvar(result.params['Z'], result.bse['Z'])

# move pretreatment outcomes and per-year covariates to new columns: we are
# going to use standard unconfoundedness given pretreatment outcomes instead
# of the DID-like assumption given by the competition rules
V_columns = [c for c in df.columns if c.startswith('V')]
posttreatment = df.filter(pl.col('post') == 1).drop(*V_columns, 'post')
for year, stratum in df.filter(pl.col('post') == 0).group_by('year'):
    posttreatment = posttreatment.join(
        stratum.select(
            'id.practice',
            pl.col(['Y', 'n.patients'] + V_columns).name.suffix(f'_year{year}')
        ), on='id.practice'
    )

# extract outcome, weights and covariates
y = posttreatment['Y'].to_numpy()
npatients_obs = posttreatment['n.patients'].to_numpy() # to scale error variance
Xobs = (posttreatment
    .drop(['Y', 'id.practice', 'n.patients'])
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

# fit treatment with a GLM
print('\nfit treatment...')
X = Xobs.drop('Z').select(pl.lit(1).alias('Intercept'), pl.all())
z = Xobs['Z'].cast(pl.Boolean).to_numpy()
model = sm.GLM(z, X.to_pandas(), family=sm.families.Binomial())
result = model.fit()
print(result.summary().tables[0])
ps = result.predict()

# fit outcome with propensity score as covariate
print('\nfit outcome (w/ PS)...')
Xobs_ps = Xobs.with_columns(ps=pl.lit(ps))
Xmis_ps = (Xobs_ps
    .filter(pl.col('Z') == 1)
    .with_columns(Z=1 - pl.col('Z'))
)
fit_outcome_ps = lgp.bayestree.bart(Xobs_ps, y, weights=npatients_obs, **bartkw)
print(fit_outcome_ps)

# fit outcome with propensity score as covariate using BCF
print('\nfit outcome (BCF)...')
fit_outcome_bcf = lgp.bayestree.bcf(
    y=y,
    z=Xobs['Z'],
    x_mu=Xobs.drop('Z'),
    pihat=Xobs_ps['ps'],
    weights=npatients_obs,
    **bartkw,
)
print(fit_outcome_bcf)

# define groups of units for conditional SATT
strata = (df
    .filter((pl.col('Z') == 1) & (pl.col('post') == 1))
    .select([f'X{i}' for i in range(1, 6)] + ['year'])
    .rename({'year': 'Yearly'})
    .with_row_index('index')
)

def compute_satt(fit, *, rng=None, **predkw):

    # create GP at MAP/sampled hypers and get imputed outcomes for the treated
    kw = dict(weights=npatients_mis, error=True, **predkw)
    if isinstance(fit, lgp.bayestree.bcf):
        kw['gvars'] = True
    elif isinstance(fit, lgp.bayestree.bart):
        kw['format'] = 'gvar'
    if rng is not None:
        with lgp.switchgvar():
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
        for (level,), stratum in strata.group_by([variable]):
            indices = stratum['index'].to_numpy()
            key = f'{variable}={level}'
            satt[key] = np.average(effect[indices], weights=n[indices])

    if rng is not None:
        satt = lgp.sample(gvar.mean(satt), gvar.evalcov(satt), rng=rng)

    satt = {k: satt[k] for k in sorted(satt)}
    return satt

print('\ncompute satt...')

# compute the SATT at marginal MAP hyperparameters
satt = compute_satt(fit_outcome, x_test=Xmis)
satt_ps = compute_satt(fit_outcome_ps, x_test=Xmis_ps)
bcf_predkw = dict(z=Xmis['Z'], x_mu=Xmis.drop('Z'), pihat=Xmis_ps['ps'])
satt_bcf = compute_satt(fit_outcome_bcf, **bcf_predkw)

# compute the SATT sampling the hyperparameters with the Laplace approx
if laplace:
    rng = np.random.default_rng(202307081315)
    
    satt_bcf_samples = {}
    for _ in tqdm.tqdm(range(nsamples)):
        satt_sample = compute_satt(fit_outcome_bcf, rng=rng, **bcf_predkw)
        for k, v in satt_sample.items():
            satt_bcf_samples.setdefault(k, []).append(v)
    
    satt_bcf_samples_quantiles = {}
    satt_bcf_samples_meanstd = {}
    for k, samples in satt_bcf_samples.items():
        q = stats.norm.cdf([-2, -1, 0, 1, 2])
        satt_bcf_samples_quantiles[k] = np.quantile(samples, q)
        satt_bcf_samples_meanstd[k] = gvar.gvar(np.mean(samples), np.std(samples))

# print results
print(f'\nATE (unadjusted) = {ate_unadjusted}')
print(f'\nSATT (no PS, MAP) =\n{pprint.pformat(satt)}')
print(f'\nSATT (w/ PS, MAP) =\n{pprint.pformat(satt_ps)}')
print(f'\nSATT (BCF, MAP) =\n{pprint.pformat(satt_bcf)}')
if laplace:
    print(f'\nSATT (BCF, Laplace) =\n{pprint.pformat(satt_bcf_samples_meanstd)}')

# load actual true effect
df_results = (pl
    .read_csv(datapath / 'results' / 'ACIC_estimand_truths.csv', null_values='NA')
    .filter(pl.col('dataset.num') == dataset)
    .filter(pl.col('variable').is_not_null())
    .with_columns(
        level=pl
            .when(pl.col('variable') == 'Yearly')
            .then(pl.col('year'))
            .otherwise(pl.col('level')),
        SATT=pl.col('SATT') + artificial_effect_shift,
    )
)

# collect and show truth
satt_true = {}
for (variable,), group in df_results.group_by(['variable']):
    if len(group) > 1:
        for row in group.iter_rows(named=True):
            level = row['level']
            satt_true[f'{variable}={level}'] = row['SATT']
    else:
        satt_true[variable] = group['SATT'][0]
print(f'\nSATT (truth) =\n{pprint.pformat(satt_true)}')

# create figure
fig, axs = plt.subplots(2, 1,
    num='acic',
    clear=True,
    layout='constrained',
    figsize=[6.4, 8],
    height_ratios=[2, 1],
)
ax_satt, ax_ps = axs

# plot propensity score distribution
ps_by_group = (Xobs_ps
    .select('Z', pl.col('ps').rank('dense'))
    .group_by('Z')
    .all()
    .sort('Z')
    .get_column('ps')
    .to_numpy()
    .tolist()
)
ax_ps.hist(ps_by_group, bins='auto', histtype='barstacked', label=['Z=0', 'Z=1'])
ax_ps.set(
    title='Propensity score distribution',
    xlabel='rank(propensity score)',
    ylabel='Bin count',
)
ax_ps.legend()

# plot comparison with truth
estimates = {
    'w/o PS, MAP': satt,
    'w/ PS, MAP': satt_ps,
    'BCF, MAP': satt_bcf,
}
if laplace:
    estimates.update({'BCF, Laplace': satt_bcf_samples_quantiles})
artist_estimate = [None] * len(estimates)

for i, (label, satt_dict) in enumerate(estimates.items()):
    tick = 0
    for stratum, estimate in satt_dict.items():
        width = 0.4
        shift = -width / 2 + width * i / (len(estimates) - 1)
        if isinstance(estimate, np.ndarray):
            x = estimate[2]
            xerr1 = np.reshape([
                estimate[3] - estimate[2],
                estimate[4] - estimate[3]
            ], (2, 1))
            xerr2 = np.reshape([
                estimate[4] - estimate[2],
                estimate[2] - estimate[0],
            ], (2, 1))
        else:
            x = gvar.mean(estimate)
            xerr1 = gvar.sdev(estimate)
            xerr2 = 2 * xerr1
        args = (x, tick - shift)
        kw = dict(fmt='.', capsize=3, color=f'C{i}')
        ax_satt.errorbar(*args, xerr=xerr2, elinewidth=1, capthick=1, **kw)
        artist_estimate[i] = ax_satt.errorbar(*args, xerr=xerr1, elinewidth=2, capthick=2, **kw, label=label)
        artist_truth, = ax_satt.plot(satt_true[stratum], tick, 'kx', label='Truth')
        tick -= 1

m = ate_unadjusted.mean
s = ate_unadjusted.sdev
artist_ate = ax_satt.axvspan(m - 2 * s, m + 2 * s, color='#eee', label='Unadjusted')
artist_ate = ax_satt.axvspan(m - s, m + s, color='lightgray', label='Unadjusted')
ax_satt.axvline(m, color='darkgray')

ax_satt.set(
    xlabel='SATT',
    ylabel='Stratum',
    yticks=np.arange(0, tick, -1),
    yticklabels=list(satt),
    title='SATT posterior (0.05, 0.16, 0.50, 0.84, 0.95 quantiles)',
)
ax_satt.legend(
    handles=[artist_truth, artist_ate, *artist_estimate],
    loc='upper right',
)

fig.show()
