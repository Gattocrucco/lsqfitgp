# lsqfitgp/_gvarext.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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
import textwrap
import string
import contextlib

import jax
import gvar
from jax import numpy as jnp
import numpy

from . import _signature

__all__ = [
    'jacobian',
    'from_jacobian',
    'gvar_gufunc',
    'switchgvar',
]

def _getsvec(x):
    """
    Get the sparse vector of derivatives of a GVar.
    """
    if isinstance(x, gvar.GVar):
        return x.internaldata[1]
    else:
        return gvar.svec(0)

def _merge_svec(gvlist, start=None, stop=None):
    if start is None:
        return _merge_svec(gvlist, 0, len(gvlist))
    n = stop - start
    if n <= 0:
        return gvar.svec(0)
    if n == 1:
        return _getsvec(gvlist[start])
    left = _merge_svec(gvlist, start, start + n // 2)
    right = _merge_svec(gvlist, start + n // 2, stop)
    return left.add(right, 1, 1)

def jacobian(g):
    """
    Extract the jacobian of gvars w.r.t. primary gvars.
    
    Parameters
    ----------
    g : array_like
        An array of numbers or gvars.
    
    Returns
    -------
    jac : array
        The shape is g.shape + (m,), where m is the total number of primary
        gvars that g depends on.
    indices : (m,) int array
        The indices that map the last axis of jac to primary gvars in the
        global covariance matrix.

    See also
    --------
    from_jacobian
    """
    g = numpy.asarray(g)
    v = _merge_svec(g.flat)
    indices = v.indices()
    jac = numpy.zeros((g.size, len(indices)), float)
    for i, x in enumerate(g.flat):
        v = _getsvec(x)
        ind = numpy.searchsorted(indices, v.indices())
        jac[i, ind] = v.values()
    jac = jac.reshape(g.shape + indices.shape)
    return jac, indices

def from_jacobian(mean, jac, indices):
    """
    Create new gvars from a jacobian w.r.t. primary gvars.
    
    Parameters
    ----------
    mean : array_like
        An array of numbers with the means of the new gvars.
    jac : mean.shape + (m,) array
        The derivatives of each new gvar w.r.t. m primary gvars.
    indices : (m,) int array
        The indices of the primary gvars.
    
    Returns
    -------
    g : mean.shape array
        The new gvars.

    See also
    --------
    jacobian
    """
    cov = gvar.gvar.cov
    mean = numpy.asarray(mean)
    shape = mean.shape
    mean = mean.flat
    jac = numpy.asarray(jac)
    jac = jac.reshape(len(mean), len(indices))
    g = numpy.zeros(len(mean), object)
    for i, jacrow in enumerate(jac):
        der = gvar.svec(len(indices))
        der._assign(jacrow, indices)
        g[i] = gvar.GVar(mean[i], der, cov)
    return g.reshape(shape)

def tabulate_together(*gs, headers=True, offset='', ndecimal=None, keys=None):
    """
    
    Format a table comparing side by side various collections of gvars.
    
    Parameters
    ----------
    *gs : sequence of arrays or dictionaries of gvars
        The variables to be tabulated. The structures of arrays and dictionaries
        must match.
    headers : bool or sequence of strings
        If True (default), add automatically an header. If False, don't add an
        header. If a sequence with length len(gs) + 1, it contains the column
        names for the keys/indices and for each set of variables.
    offset : str
        Prefix to each line, default empty.
    ndecimal : int, optional
        Number of decimal places. If not specified (default), keep two error
        digits.
    keys : sequence, optional
        If ``gs`` are dictionaries, a subset of keys to be extracted from each
        dictionary. Ignored if they are arrays.
    
    Examples
    --------
    >>> print(tabulate_together(gvar.gvar(dict(a=1)), gvar.gvar(dict(a=2))))
    key/index   value1   value2
    ---------------------------
            a    1 (0)    2 (0)
    
    See also
    --------
    gvar.tabulate
    
    """
    if not gs:
        return ''
    gs = [g if hasattr(g, 'keys') else numpy.asarray(g) for g in gs]
    assert all(hasattr(g, 'keys') for g in gs) or all(not hasattr(g, 'keys') for g in gs)
    if keys is not None and hasattr(gs[0], 'keys'):
        gs = [{k: g[k] for k in keys} for g in gs]
    g0 = gs[0]
    if hasattr(g0, 'keys'):
        assert all(set(g.keys()) == set(g0.keys()) for g in gs[1:])
        gs = [{k: g[k] for k in g0} for g in gs]
    else:
        assert all(g.shape == g0.shape for g in gs[1:])
        if g0.shape == ():
            gs = [{'--': g} for g in gs]
    tables = [
        _splittable(gvar.tabulate(g, headers=['@', ''], ndecimal=ndecimal))
        for g in gs
    ]
    columns = list(tables[0]) + [t[1] for t in tables[1:]]
    if not hasattr(headers, '__len__'):
        if headers:
            headers = ['key/index'] + [f'value{i+1}' for i in range(len(gs))]
        else:
            headers = None
    else:
        assert len(headers) == len(columns)
    if headers is not None:
        columns = (_head(col, head) for col, head in zip(columns, headers))
    return textwrap.indent(_join(columns), offset)

def _splittable(table):
    lines = table.split('\n')
    header = lines[0]
    col = header.find('@') + 1
    contentlines = lines[2:]
    col1 = '\n'.join(line[:col] for line in contentlines)
    col2 = '\n'.join(line[col:] for line in contentlines)
    return col1, col2

def _head(col, head):
    head = str(head)
    width = col.find('\n')
    if width < 0:
        width = len(col)
    hwidth = len(head)
    if hwidth > width:
        col = textwrap.indent(col, (hwidth - width) * ' ')
    else:
        head = (width - hwidth) * ' ' + head
    return head + '\n' + len(head) * '-' + '\n' + col

def _join(cols):
    split = (col.split('\n') for col in cols)
    return '\n'.join(''.join(lines) for lines in zip(*split))

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

@contextlib.contextmanager
def switchgvar():
    """
    Context manager to keep new gvars in a separate pool.

    Creating new primary gvars fills up memory permanently. This context manager
    keeps the gvars created within its context in a separate pool that is freed
    when all such gvars are deleted. They can not be mixed in operations with
    other gvars created outside of the context.
    
    Returns
    -------
    gvar : gvar.GVarFactory
        The new gvar-creating function that uses a new pool. The change is also
        reflected in the global `gvar.gvar`.

    See also
    --------
    gvar.switch_gvar, gvar.restore_gvar

    Examples
    --------

    >>> x = gvar.gvar(0, 1)
    >>> with lgp.switchgvar():
    >>>     y = gvar.gvar(0, 1)
    >>>     z = gvar.gvar(0, 1)
    >>> w = gvar.gvar(0, 1)
    >>> q = y + z  # allowed, y and z created in the same pool
    >>> p = x + w  # allowed, x and w created in the same pool
    >>> h = x + y  # x and y created in different pools: this will silently
    ...            # fail and possibly crash python immediately or later on

    """
    try:
        yield gvar.switch_gvar()
    finally:
        gvar.restore_gvar()
