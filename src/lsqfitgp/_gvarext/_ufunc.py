# lsqfitgp/_gvarext/_ufunc.py
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

import functools
import string

import jax
import gvar
from jax import numpy as jnp

from .. import _signature

from ._jacobian import jacobian, from_jacobian
from ._tabulate import tabulate_together

def gvar_gufunc(func, *, signature=None):
    """

    Wraps a jax-traceable generalized ufunc with one argument to support gvars.

    Parameters
    ----------
    func : callable
        A function from one array to one array. It must be a generalized ufunc,
        and differentiable one time with `jax`.
    signature : str, optional
        The signature of the generalized ufunc. If not specified, it is assumed
        to be scalar to scalar (normal ufunc).

    Returns
    -------
    decorated_func : callable
        A function that, in addition to numerical arrays, accepts gvars and
        returns gvars.

    See also
    --------
    numpy.vectorize

    """

    # parse signature
    if signature is None:
        signature = '()->()'
    sig = _signature.Signature(signature)
    inp, = sig.incores
    out, = sig.outcores
    jac_sig = _signature.Signature.from_tuples([inp], [out + inp])
    
    # make jacobian function
    deriv = jnp.vectorize(jax.jacfwd(func), signature=jac_sig.signature)

    # get indices for summation
    ninp = len(inp)
    nout = len(out)
    head_indices = '...'
    out_indices = string.ascii_letters[:nout]
    in_indices = string.ascii_letters[nout:nout + ninp]
    gvar_indices = string.ascii_letters[nout + ninp]

    # make summation formula
    jac_indices = head_indices + out_indices + in_indices
    in_jac_indices = head_indices + in_indices + gvar_indices
    out_indices = head_indices + out_indices + gvar_indices
    formula = f'{jac_indices},{in_jac_indices}->{out_indices}'

    def gvar_function(x):

        # unpack the gvars
        in_mean = gvar.mean(x)
        in_jac, indices = jacobian(x)

        # apply function
        out_mean = func(in_mean)
        jac = deriv(in_mean)

        # check shapes match
        head_ndim = jac.ndim - nout - ninp
        assert jac.shape[:head_ndim] == in_jac.shape[:in_jac.ndim - 1 - ninp]

        # contract
        out_jac = jnp.einsum(formula, jac, in_jac)

        # pack output
        return from_jacobian(out_mean, out_jac, indices)

    @functools.wraps(func)
    def decorated_func(x):
        if isinstance(x, gvar.GVar):
            out = gvar_function(x)
            if not out.ndim:
                out = out.item()
            return out
        elif getattr(x, 'dtype', None) == object:
            return gvar_function(x)
        else:
            return func(x)

    return decorated_func

    # TODO add more than one argument or output. Possibly without taking
    # derivatives when it's not a gvar, i.e., merge the wrappers and cycle over
    # args. Also implement excluded => note that jnp.vectorize only supports
    # positional arguments, excluded takes in only indices, not names
