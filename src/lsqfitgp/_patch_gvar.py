# lsqfitgp/_patch_gvar.py
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

import jax
import gvar
from jax import numpy as jnp
from jax.scipy import special as jspecial
from jax import tree_util
import numpy
from scipy import linalg

from . import _patch_jax

gvar_ufuncs = [
    'sin',
    'cos',
    'tan',
    'exp',
    'log',
    'sqrt',
    'fabs',
    'sinh',
    'cosh',
    'tanh',
    'arcsin',
    'arccos',
    'arctan',
    'arctan2',
    'arcsinh',
    'arccosh',
    'arctanh',
    'square',
    'erf',
]

for fname in gvar_ufuncs:
    fgvar = getattr(gvar, fname)
    fjax = getattr(jnp, fname, getattr(jspecial, fname, NotImplemented))
    fboth = functools.singledispatch(fgvar)
    fboth.register(jnp.ndarray, fjax)
    setattr(gvar, fname, fboth)

# reset transformations to support jax arrays
gvar.BufferDict.del_distribution('log')
gvar.BufferDict.del_distribution('sqrt')
gvar.BufferDict.del_distribution('erfinv')
gvar.BufferDict.add_distribution('log', gvar.exp)
gvar.BufferDict.add_distribution('sqrt', gvar.square)
gvar.BufferDict.add_distribution('erfinv', gvar.erf)

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
    
    Return
    ------
    jac : array
        The shape is g.shape + (m,), where m is the total number of primary
        gvars that g depends on.
    indices : (m,) int array
        The indices that map the last axis of jac to primary gvars in the
        global covariance matrix.
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
    
    Return
    ------
    g : mean.shape array
        The new gvars.
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

def _skeleton(bd):
    return gvar.BufferDict(
        bd,
        buf=numpy.broadcast_to(numpy.empty((), bd.dtype), bd.buf.shape),
        # buf is not copied, so the memoryless broadcast is kept
    )

def bufferdict_flatten(bd):
    return (bd.buf,), _skeleton(bd)

def bufferdict_unflatten(skeleton, children):
    buf, = children
    new = _skeleton(skeleton)
    # copy the skeleton to permit multiple unflattening
    new._extension = {}
    new._buf = buf
    return new

# register BufferDict as a pytree
tree_util.register_pytree_node(gvar.BufferDict, bufferdict_flatten, bufferdict_unflatten)

# TODO the current implementation of BufferDict as pytree is not really
# consistent with how JAX handles trees, because JAX expects to be allowed to
# put arbitrary objects in the leaves; in particular, internally it sometimes
# creates dummy trees filled with None. Maybe the current impl is fine with
# this; _buf gets set to None, and assuming the BufferDict is never really
# used in that crooked state, everything goes fine. The thing that this breaks
# is a subsequent flattening of the dummy, I think JAX never does this. (The
# reason for switching to buf-as-leaf in place of dict-values-as-leaves is that
# the latter breaks tracing.)

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

def add_gvar_support(func):
    """ Wraps a jax-traceable ufunc to support gvars """
    
    dfdx = _patch_jax.elementwise_grad(func)
    
    def gvar_function(x):
        m = gvar.mean(x)
        return gvar.gvar_function(x, func(m), dfdx(m))
    
    gvar_function_vectorized = numpy.vectorize(gvar_function)

    @functools.wraps(func)
    def decorated_func(x):
        if isinstance(x, gvar.GVar):
            return gvar_function(x)
        elif getattr(x, 'dtype', None) == object:
            return gvar_function_vectorized(x)
        else:
            return func(x)
    
    return decorated_func

    # TODO make public? => To make it public, I need it to support arbitrary
    # arguments, with some configuration (imitate vectorize), and pick a name
    # that makes it clear it convert ufuncs from jax to gvar. => Maybe I should
    # make a more general version starting from the code in GP that propagates
    # gvars through lintransf.
