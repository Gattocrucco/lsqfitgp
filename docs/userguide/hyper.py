# lsqfitgp/docs/userguide/hyper.py
#
# Copyright (c) 2020, 2023, 2026, Giacomo Petrillo
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

import gvar  ####
import numpy as np  ####
import pymc3 as pm  # ty: ignore[unresolved-import]
from scipy import stats  ####

x = np.linspace(-5, 5, 11)  ####
y = np.sin(x)  ####

model = pm.Model()
with model:
    logscale = pm.Normal('logscale', mu=np.log(3), sigma=1)
    logsdev = pm.Normal('logsdev', mu=np.log(1), sigma=1)
    kernel = pm.math.exp(logsdev) ** 2 * pm.gp.cov.ExpQuad(1, ls=pm.math.exp(logscale))
    gp = pm.gp.Marginal(cov_func=kernel)
    gp.marginal_likelihood('data', x[:, None], y, 0)

with model:
    mp = pm.find_MAP()
    trace = pm.sample(10000, cores=1)

print('\nMaximum a posteriori (must be the same as lsqfitgp):')
print('log(sdev) {:.2f}'.format(mp['logsdev']))
print('log(scale) {:.2f}'.format(mp['logscale']))

df = pm.trace_to_dataframe(trace)
mean = df.mean()
cov = df.cov()

meandict = {}
covdict = {}
for label1 in df.columns:
    meandict[label1] = mean[label1]
    for label2 in df.columns:
        covdict[label1, label2] = cov[label1][label2]

params = gvar.gvar(meandict, covdict)
print('\nPosterior mean and standard deviation:')
print('log(sdev)', params['logsdev'])
print('log(scale)', params['logscale'])

p = params['logsdev']
prob_gauss = stats.norm.cdf(np.log(1), loc=gvar.mean(p), scale=gvar.sdev(p))
true_prob = np.sum(df['logsdev'] <= np.log(1)) / len(df)
print('\nProbability of having sdev < 1:')
print(f'prob_gauss {prob_gauss:.3g}')
print(f'true_prob {true_prob:.3g}')
