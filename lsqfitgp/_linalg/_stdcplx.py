# lsqfitgp/_linalg/_stdcplx.py
#
# Copyright (c) 2022, Giacomo Petrillo
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
Module to estimate the time taken by standard linear algebra operations.
"""

__all__ = [
    'predtime',
    'listops',
]

import timeit
import inspect

from jax import random
from jax import numpy as jnp
from jax.scipy import linalg as jlinalg

def benchmark(func, *args, **kwargs):
    timer = timeit.Timer('func(*args, **kwargs)', globals=locals())
    n, _ = timer.autorange()
    times = timer.repeat(5, n)
    time = min(times) / n
    return time

ops = {
    'chol': [
        lambda x: jnp.linalg.cholesky(x), # function performing the operation
        lambda s: s[0] ** 3, # complexity in terms of arguments' shapes
    ],
    'eigh': [
        lambda x: jnp.linalg.eigh(x),
        lambda s: s[0] ** 3,
    ],
    'qr-red': [
        lambda x: jnp.linalg.qr(x, mode='reduced'),
        lambda s: min(s) ** 2 * max(s),
    ],
    'qr-full': [
        lambda x: jnp.linalg.qr(x, mode='full'),
        lambda s: max(s) ** 3,
    ],
    'svd-red': [
        lambda x: jnp.linalg.svd(x, full_matrices=False),
        lambda s: min(s) ** 2 * max(s),
    ],
    'svd-full': [
        lambda x: jnp.linalg.svd(x, full_matrices=True),
        lambda s: max(s) ** 3,
    ],
    'solve_triangular': [
        lambda x, y: jlinalg.solve_triangular(x, y),
        lambda s, t: s[0] ** 2 * t[1],
    ],
}

def gen_ops_factors(n):
    key = random.PRNGKey(202208101236)
    factors = {}
    for op, (job, est) in ops.items():
        print(f'{op}({n})... ', end='', flush=True)
        nparams = len(inspect.signature(job).parameters)
        key, subkey = random.split(key)
        m = random.normal(subkey, (nparams, n, n), jnp.float32)
        args = m @ m
        time = benchmark(job, *args)
        print(f'{time:.2g} s')
        factors[op] = time / est(*(a.shape for a in args))
    return factors

ops_factors = {'chol': 8.984154160134494e-13,
 'eigh': 1.7393400007858872e-10,
 'qr-red': 1.1879412503913046e-10,
 'qr-full': 1.2299797893501818e-10,
 'svd-red': 4.287901660427451e-10,
 'svd-full': 4.346819589845836e-10,
 'solve_triangular': 4.7264866810292004e-12} # = gen_ops_factors(1000)

ops_consts = {'chol': 1.8038220843300223e-06,
 'eigh': 2.4166233302094043e-06,
 'qr-red': 2.6811220799572766e-06,
 'qr-full': 2.7157124993391336e-06,
 'svd-red': 3.675635000690818e-06,
 'svd-full': 3.6995087494142353e-06,
 'solve_triangular': 2.2041862504556774e-06} # = gen_ops_factors(1)

def predtime(op, shapes, types):
    """
    Estimate the time taken by a linear algebra operation.
    
    Parameters
    ----------
    op : str
        The identifier of the operation, see `listops`.
    shapes : sequence of tuples of integers
        The shapes of the arguments.
    types : sequence of numpy data types
        The types of the arguments. They are promoted according to JAX rules.
    
    Returns
    -------
    time : float
        The estimated time. Not accurate. The unit of measure is seconds on a
        particular laptop cpu used to calibrate the estimate with 1000x1000
        matrices.
    """
    _, est = ops[op]
    factor = ops_factors[op]
    const = ops_consts[op]
    dt = jnp.sin(jnp.empty(0, jnp.result_type(*types))).dtype
    if dt == jnp.float64:
        factor *= 2
    return const + factor * est(*shapes)

def listops():
    """
    List available linear algebra operations.
    
    Returns
    -------
    ops : dict
        A dictionary operation identifier -> number of arguments.
    """
    return {
        op: len(inspect.signature(job).parameters)
        for op, (job, _) in ops.items()
    }
