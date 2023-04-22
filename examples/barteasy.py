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

import lsqfitgp as lgp
import numpy as np
from numpy.lib import recfunctions
import polars as pl
import gvar
from jax import numpy as jnp

# Load and preprocess data

datafile = 'examples/bart-data.txt'

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

X = df.drop('Rings')
y = df['Rings']

# Fit BART

bart = lgp.bayestree.bart(X, y)
fit = bart.fit

# Print hyperparameters report

print()
print(f'alpha = {bart.alpha} (0 -> intercept only, 1 -> any)')
print(f'beta = {bart.beta} (0 -> any, ∞ -> no interactions)')
print(f'latent sdev = {bart.meansdev} (large -> conservative extrapolation)')
print(f'error sdev = {bart.sigma}')
print(f'data total sdev = {y.std():.1f}')
