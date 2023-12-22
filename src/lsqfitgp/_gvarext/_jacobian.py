# lsqfitgp/_gvarext/_jacobian.py
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

import gvar
import numpy

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
