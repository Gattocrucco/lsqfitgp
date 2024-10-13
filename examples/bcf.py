import pprint
import pathlib

import polars as pl
import lsqfitgp as lgp
import numpy as np
import gvar
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt
import tqdm

"""

Analyze a dataset from the ACIC 2022 Data Challenge [1]_ using GP-BCF.

.. [1] Dan R.C. Thal and Mariel M. Finucane, "Causal Methods Madness: Lessons
       Learned from the 2022 ACIC Competition to Estimate Health Policy
       Impacts," Observational Studies, Volume 9, Issue 3, 2023, pp. 3-27,
       https://doi.org/10.1353/obs.2023.0023, https://acic2022.mathematica.org

"""

# config
less_data = True # whether to halve the data for execution speed
artificial_effect_shift = 0 # shift the treated outcome by this amount
nsamples_hp = 20 # samples from hyper posterior
nsamples_per_hp = 50 # samples from gp posterior for each hyper sample

# fixed config
datapath = pathlib.Path(__file__).parent / 'acic'

# load data
print('load data...')
prefix = datapath / 'track2_20220404'
df_p = pl.read_csv(prefix / 'practice' / f'acic_practice_0001.csv')
df_py = pl.read_csv(prefix / 'practice_year' / f'acic_practice_year_0001.csv')
df = df_p.join(df_py, on='id.practice')

# shift treated units by a fixed amount for testing purposes
df = df.with_columns(
    pl.col('Y') + pl
    .when((pl.col('Z') == 1) & (pl.col('post') == 1))
    .then(artificial_effect_shift)
    .otherwise(0)
)

# drop randomly selected observations to keep the script fast
if less_data:
    df = df.filter(
        pl.col('id.practice')
        .is_in(
            pl.col('id.practice')
            .unique()
            .sample(250, seed=20230623)
        )
    )

# compute effect as if Z was randomized, i.e., without adjustment
print('unadjusted fit...')
data = (df
    .filter(pl.col('post') == 1)
    .select(['Y', 'Z'])
    .to_pandas()
)
model = smf.wls('Y ~ 1 + Z', data)
result = model.fit()
ate_unadjusted = gvar.gvar(result.params['Z'], result.bse['Z'])

# move pretreatment outcomes and per-year covariates to new columns: we are
# going to use standard unconfoundedness given pretreatment outcomes instead
# of the parallel trends assumption given by the competition rules
V_columns = [c for c in df.columns if c.startswith('V')]
posttreatment = df.filter(pl.col('post') == 1).drop(*V_columns, 'post')
for (year,), stratum in df.filter(pl.col('post') == 0).group_by('year'):
    posttreatment = posttreatment.join(
        stratum.select(
            'id.practice',
            pl.col(['Y', 'n.patients'] + V_columns).name.suffix(f'_year{year}')
        ), on='id.practice'
    )

# add pre-treatment trend as covariate
posttreatment = posttreatment.with_columns(
    pre_trend=pl.col('Y_year2') - pl.col('Y_year1'))

# split data in outcome and predictors
y = posttreatment['Y'].to_numpy()
Xobs = (posttreatment
    .drop(['Y', 'id.practice', 'n.patients'])
    .to_dummies(columns=['X2', 'X4'])
)
npatients_obs = posttreatment['n.patients'].to_numpy() # for SATT average

# fit treatment status with a GLM to get propensity score
print('\nfit treatment...')
X = Xobs.drop('Z').select(pl.lit(1).alias('Intercept'), pl.all())
z = Xobs['Z'].cast(pl.Boolean).to_numpy()
model = sm.GLM(z, X.to_pandas(), family=sm.families.Binomial())
result = model.fit()
print(result.summary().tables[0])
Xobs = Xobs.with_columns(ps=pl.lit(result.predict()))

# fit outcome with propensity score as predictor using BCF
print('\nfit outcome...')
bcf = lgp.bayestree.bcf(
    y=y,
    z=Xobs['Z'],
    x_mu=Xobs.drop('Z', 'ps'),
    pihat=Xobs['ps'],
    transf=['standardize', 'yeojohnson'],
)
print(bcf)

# negate treatment status to impute counterfactual outcomes
Xmis = (Xobs
    .filter(pl.col('Z') == 1) # only on the treated because we want the SATT
    .with_columns(Z=1 - pl.col('Z'))
)

# define groups of units for conditional SATT
strata = (df
    .filter((pl.col('Z') == 1) & (pl.col('post') == 1))
    .select([f'X{i}' for i in range(1, 6)] + ['year'])
    .rename({'year': 'Yearly'})
    .with_row_index('index')
)

def impute_counterfactual(hp, rng, nsamples):
    return bcf.pred(
        z=Xmis.get_column('Z'),
        x_mu=Xmis.drop('ps', 'Z'),
        pihat=Xmis.get_column('ps'),
        error=True,
        samples=nsamples,
        transformed=False,
        hp=hp,
        rng=rng,
    )

def compute_satt(ymis):
    """ compute in-sample average effect on the treated given imputed couterfactual outcomes """
    yobs = y[z]
    n = npatients_obs[z]
    effect = yobs - ymis

    satt = {}

    satt['Overall'] = np.average(effect, weights=n, axis=-1)
    for variable in strata.drop('index').columns:
        for (level,), stratum in strata.group_by(variable):
            indices = stratum['index'].to_numpy()
            key = f'{variable}={level}'
            satt[key] = np.average(effect[..., indices], weights=n[indices], axis=-1)

    return sortdict(satt)

def sortdict(d):
    return {k: d[k] for k in sorted(d)}

def posterior_summaries(satt_samples):
    quantiles = {}
    meanstd = {}
    for k, samples in satt_samples.items():
        q = stats.norm.cdf([-2, -1, 0, 1, 2])
        quantiles[k] = np.quantile(samples, q)
        meanstd[k] = gvar.gvar(np.mean(samples), np.std(samples))
    return quantiles, meanstd

print('\ncompute satt...')

rng = np.random.default_rng(202307081315)

# compute the SATT at the hyperparameters map
ymis_sample = impute_counterfactual('map', rng, nsamples_hp * nsamples_per_hp)
satt_map_samples = compute_satt(ymis_sample)

# compute the SATT sampling the hyperparameters posterior
satt_samples = {}
for _ in range(nsamples_hp):
    ymis_sample = impute_counterfactual('sample', rng, nsamples_per_hp)
    satt_sample = compute_satt(ymis_sample)
    for k, v in satt_sample.items():
        satt_samples.setdefault(k, []).append(v)

# concatenate samples taken at different hyperparameters
for k, v in satt_samples.items():
    satt_samples[k] = np.concatenate(v)

# compute posterior summaries from the samples
satt_map_quantiles, satt_map_meanstd = posterior_summaries(satt_map_samples)
satt_quantiles, satt_meanstd = posterior_summaries(satt_samples)

# print results
print(f'\nATE (unadjusted) = {ate_unadjusted}')
print(f'\nSATT (BCF, MAP) =\n{pprint.pformat(satt_map_meanstd)}')
print(f'\nSATT (BCF, Laplace) =\n{pprint.pformat(satt_meanstd)}')

# load actual true effect
df_results = (pl
    .read_csv(datapath / 'results' / 'ACIC_estimand_truths.csv', null_values='NA')
    .filter(pl.col('dataset.num') == 1)
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
for (variable,), group in df_results.group_by('variable'):
    if len(group) > 1:
        for row in group.iter_rows(named=True):
            level = row['level']
            satt_true[f'{variable}={level}'] = row['SATT']
    else:
        satt_true[variable] = group['SATT'][0]
print(f'\nSATT (truth) =\n{pprint.pformat(satt_true)}')

# create figure
fig, axs = plt.subplots(2, 1,
    num='bcf',
    clear=True,
    figsize=[6.4, 8],
    height_ratios=[2, 1],
    layout='constrained',
)
ax_satt, ax_ps = axs

# plot propensity score distribution
ps_by_group = (Xobs
    .select('Z', pl.col('ps').rank('dense'))
    .group_by('Z')
    .all()
    .sort('Z')
    .get_column('ps')
    .to_numpy()
    .tolist()
)
_, _, (z0, z1) = ax_ps.hist(ps_by_group, bins='auto', histtype='barstacked', label=['Z=0', 'Z=1'])
ax_ps.set(
    title='Propensity score distribution',
    xlabel='rank(propensity score)',
    ylabel='Bin count',
)
ax_ps.legend(handles=[z1[0], z0[0]])

# plot comparison with truth
estimates = {
    'BCF, MAP': satt_map_quantiles,
    'BCF, Laplace': satt_quantiles,
}
artist_estimate = [None] * len(estimates)

for i, (label, satt_dict) in enumerate(estimates.items()):
    tick = 0
    for stratum, estimate in satt_dict.items():
        
        width = 0.2
        if len(estimates) >= 2:
            shift = -width / 2 + width * i / (len(estimates) - 1)
        else:
            shift = 0
        
        x = estimate[2]
        xerr1 = np.reshape([
            estimate[3] - estimate[2],
            estimate[4] - estimate[3]
        ], (2, 1))
        xerr2 = np.reshape([
            estimate[4] - estimate[2],
            estimate[2] - estimate[0],
        ], (2, 1))
        
        args = (x, tick - shift)
        kw = dict(fmt='.', capsize=3, color=f'C{i}')
        ax_satt.errorbar(*args, xerr=xerr2, elinewidth=1, capthick=1, **kw)
        artist_estimate[i] = ax_satt.errorbar(*args, xerr=xerr1, elinewidth=2, capthick=2, **kw, label=label)
        artist_truth, = ax_satt.plot(satt_true[stratum], tick, 'kx', markersize=10, label='Truth')
        tick -= 1

m = ate_unadjusted.mean
s = ate_unadjusted.sdev
ax_satt.axvspan(m - 2 * s, m + 2 * s, color='#eee')
artist_ate = ax_satt.axvspan(m - s, m + s, color='lightgray', label='Unadjusted')
ax_satt.axvline(m, color='darkgray')

ax_satt.set(
    xlabel='SATT',
    ylabel='Stratum',
    yticks=np.arange(0, tick, -1),
    yticklabels=list(satt_samples),
    title='SATT posterior (0.05, 0.16, 0.50, 0.84, 0.95 quantiles)',
)
ax_satt.legend(
    handles=[artist_truth, artist_ate, *artist_estimate],
    loc='upper right',
)

fig.show()
