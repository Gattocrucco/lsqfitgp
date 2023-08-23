# lsqfitgp/examples/bart.py
#
# Copyright (c) 2023, Giacomo Petrillo
#
# This file is part of lsqfitgp.
#
# lsqfitgp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# lsqfitgp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.

"""
Example usage of the BART kernel to replace the standard BART MCMC algorithm.
"""

import pathlib

import lsqfitgp as lgp
import numpy as np
from numpy.lib import recfunctions
import polars as pl
import gvar
from jax import numpy as jnp

# Load and preprocess data

datafile = pathlib.Path('examples') / 'bart-data.txt'

columns = """
    Sex
    Length
    Diameter
    Height
    Whole weight
    Shucked weight
    Viscera weight
    Shell weight
    Rings
"""
columns = [x for x in [x.strip() for x in columns.split('\n')] if x]

df = pl.read_csv(datafile, new_columns=columns, has_header=False, dtypes={
    'Sex': pl.Categorical,
    'Rings': pl.Float64,
}).to_dummies(columns='Sex')

df = df[:500] # drop most data to keep the script fast

X = lgp.StructuredArray.from_dataframe(df.drop('Rings'))
y = df['Rings'].to_numpy()

# Compute fixed hyperparameters values

ymin = np.min(y)
ymax = np.max(y)
mu_mu = (ymax + ymin) / 2 # prior mean
k_sigma_mu = (ymax - ymin) / 2 # prior standard deviation times k

# Gaussian copula prior on the free hyperparameters

hyperprior = lgp.copula.makedict({
    'alpha': lgp.copula.beta(2, 1), # base of tree gen prob
    'beta': lgp.copula.invgamma(1, 1), # exponent of tree gen prob
    'log(k)': gvar.gvar(np.log(2), 2), # denominator of prior sdev
    'log(sigma2)': gvar.gvar(np.log(1), 2), # i.i.d. error variance
})

# Splitting points and indices

splits = lgp.BART.splits_from_coord(X)
def toindices(x):
    ix = lgp.BART.indices_from_coord(x, splits)
    return lgp.unstructured_to_structured(ix, names=x.dtype.names)
X_indices = toindices(X)

# GP factory

def makegp(hp, *, i_train, i_test, splits):
    kw = dict(alpha=hp['alpha'], beta=hp['beta'], maxd=10, reset=[2,4,6,8], gamma=0.95)
    kernel = lgp.BART(splits=splits, indices=True, **kw)
    kernel *= (k_sigma_mu / hp['k']) ** 2
    
    return (lgp
        .GP(kernel, checkpos=False, checksym=False, solver='chol')
        .addx(i_train, 'data_latent')
        .addcov(hp['sigma2'] * jnp.eye(i_train.size), 'noise')
        .addtransf({'data_latent': 1, 'noise': 1}, 'data')
        .addx(i_test, 'test')
    )

# Fit hyperparameters (Laplace approximation of the marginal posterior)

info = {'data': y - mu_mu}
gpkw = dict(i_train=X_indices, i_test=X_indices, splits=splits)
options = dict(
    verbosity=3,
    raises=False,
    minkw=dict(method='l-bfgs-b', options=dict(maxls=4, maxiter=100)),
    mlkw=dict(epsrel=0),
    forward=True,
    gpfactorykw=gpkw,
)
fit = lgp.empbayes_fit(hyperprior, makegp, info, **options)

# Print hyperparameters report

print()
print('alpha =', fit.p['alpha'], '(0 -> force depth=0 trees, 1 -> allow deep trees)')
print('beta =', fit.p['beta'], '(0 -> allow deep trees, ∞ -> force depth<=1 trees)')
print('latent sdev =', k_sigma_mu / fit.p['k'], '(large -> conservative extrapolation)')
print('error sdev =', gvar.sqrt(fit.p['sigma2']))
print(f'data total sdev = {np.std(y):.3f}')

# Extract mean predictions from GP at hyperparameters MAP

gp = makegp(fit.pmean, **gpkw)
yhat_mean, yhat_cov = gp.predfromdata(info, 'test', raw=True)
yhat_mean += mu_mu
