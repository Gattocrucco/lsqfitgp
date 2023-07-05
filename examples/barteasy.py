# lsqfitgp/examples/barteasy.py
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
BART with the simplified subpackage.
"""

import pathlib

import lsqfitgp as lgp
import numpy as np
import polars as pl
import gvar
from matplotlib import pyplot as plt

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
columns = list(filter(None, map(lambda x: x.strip(), columns.split('\n'))))

n = 500
df = (pl
    .read_csv(datafile, new_columns=columns, has_header=False)
    .to_dummies(columns='Sex')
    .sample(2 * n, seed=20230605) # drop most data to keep the script fast
)

X = df.drop('Rings')
y = df['Rings'].cast(pl.Float64)

X_data = X[:n]
y_data = y[:n]
X_test = X[n:]
y_test = y[n:]

# Fit BART

bart = lgp.bayestree.bart(X_data, y_data)
print()
print(bart)

# Compute predictions

yhat_mean, yhat_cov = bart.pred(x_test=X_test, error=True)

# Compare predictions with truth

fig, ax = plt.subplots(num='barteasy', clear=True)
ax.errorbar(yhat_mean, y_test, xerr=np.sqrt(np.diag(yhat_cov)), fmt='.')
ax.plot(y_test, y_test, 'k-')
ax.set(
    ylabel='truth',
    xlabel='prediction',
)
fig.show()
