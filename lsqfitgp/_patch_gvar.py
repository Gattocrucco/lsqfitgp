# lsqfitgp/_patch_gvar.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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

import jax
import gvar
from jax import numpy as jnp
from jax.scipy import special as jspecial
from jax import tree_util
import numpy as np
from scipy import linalg

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

def jaxsupport(jax_ufunc):
    """
    Create a wrapper of a function that will dispatch to another specified
    function if the argument is a jax array.
    """
    def decorator(gvar_ufunc):
        @functools.wraps(gvar_ufunc)
        def newfunc(x):
            if isinstance(x, jnp.ndarray):
                return jax_ufunc(x)
            else:
                return gvar_ufunc(x)
        return newfunc
    return decorator

for fname in gvar_ufuncs:
    fgvar = getattr(gvar, fname)
    fjax = getattr(jspecial if fname == 'erf' else jnp, fname)
    # fboth = functools.singledispatch(fgvar)
    # fboth.register(jnp.ndarray, fjax)
    # python dispatch mechanism does not work, see jax issue #11075
    fboth = jaxsupport(fjax)(fgvar)
    setattr(gvar, fname, fboth)

# reset transformations to support jax arrays
gvar.BufferDict.del_distribution('log')
gvar.BufferDict.del_distribution('sqrt')
gvar.BufferDict.del_distribution('erfinv')
gvar.BufferDict.add_distribution('log', gvar.exp)
gvar.BufferDict.add_distribution('sqrt', gvar.square)
gvar.BufferDict.add_distribution('erfinv', gvar.erf)

def scipy_eigh(x):
    w, v = linalg.eigh(x)
    w = np.abs(w)
    si = np.argsort(w)
    return w[si], v.T[si]

gvar.SVD._analyzers = dict(scipy_eigh=scipy_eigh)
# default uses SVD instead of diagonalization because it once was more stable

def getsvec(x):
    """
    Get the sparse vector of derivatives of a GVar.
    """
    if isinstance(x, gvar.GVar):
        return x.internaldata[1]
    else:
        return gvar.svec(0)

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
    g = np.asarray(g)
    v = gvar.svec(0)
    for x in g.flat:
        v = v.add(getsvec(x), 1, 1)
    indices = v.indices()
    jac = np.zeros((g.size, len(indices)), float)
    for i, x in enumerate(g.flat):
        v = getsvec(x)
        ind = np.searchsorted(indices, v.indices())
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
    mean = np.asarray(mean)
    shape = mean.shape
    mean = mean.flat
    jac = np.array(jac) # TODO patches gvar issue #27
    jac = jac.reshape(len(mean), len(indices))
    g = np.zeros(len(mean), object)
    for i, jacrow in enumerate(jac):
        der = gvar.svec(len(indices))
        der._assign(jac[i], indices)
        g[i] = gvar.GVar(mean[i], der, cov)
    return g.reshape(shape)
    
def bufferdict_flatten(bd):
    return tuple(bd.values()), tuple(bd.keys())

def bufferdict_unflatten(keys, values):
    return gvar.BufferDict(zip(keys, values))

# register BufferDict as a pytree
tree_util.register_pytree_node(gvar.BufferDict, bufferdict_flatten, bufferdict_unflatten)
