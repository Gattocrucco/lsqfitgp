# lsqfitgp/_fit.py
#
# Copyright (c) 2020, Giacomo Petrillo
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

from ._imports import gvar
from ._imports import numpy as np
from ._imports import linalg
from ._imports import autograd
from ._imports import optimize

from . import _GP

__all__ = [
    'empbayes_fit'
]

def _asarrayorbufferdict(x):
    if hasattr(x, 'keys'):
        return gvar.BufferDict(x)
    else:
        return np.array(x, copy=False)

def _flat(x):
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    elif isinstance(x, gvar.BufferDict):
        return x.buf

def _unflat(x, original):
    if isinstance(original, np.ndarray):
        out = x.reshape(original.shape)
        return out if out.shape else out.item
    elif isinstance(original, gvar.BufferDict):
        return gvar.BufferDict(original, buf=x)

def empbayes_fit(hyperprior, gpfactory, data, raises=True):
    """
    
    Empirical bayes fit.
    
    Maximizes the marginal likelihood of the data with a gaussian process model
    that depends on hyperparameters.
    
    Parameters
    ----------
    hyperprior : scalar, array or dictionary of scalars/arrays
        A collection of gvars representing the prior for the hyperparameters.
    gpfactory : callable
        A function with signature gpfactory(hyperparams) -> GP object. The
        argument `hyperparams` has the same structure of the empbayes_fit
        argument `hyperprior`. gpfactory must be autograd-friendly, i.e. either
        use autograd.numpy, autograd.scipy, lsqfitgp.numpy, lsqfit.scipy or
        gvar instead of plain numpy/scipy.
    data : dictionary
        Dictionary of data that is passed to GP.marginal_likelihood on the
        GP object returned by `gpfactory`.
    raises : bool, optional
        If True (default), raise an error when the minimization fails.
        Otherwise, use the last point of the minimization as result.
    
    Returns
    -------
    hyperparams : scalar, array or dictionary of scalars/arrays
        A collection of gvars representing the hyperparameters that maximize
        the marginal likelihood. The covariance matrix is computed as the
        inverse of the hessian of the marginal likelihood. These gvars do
        not track correlations with the hyperprior or the data.

    Raises
    ------
    RuntimeError
        The minimization failed and `raises` is True.
    
    """
    assert isinstance(data, (dict, gvar.BufferDict)) # TODO duck typing of data
    assert callable(gpfactory)
    
    # TODO allow data to be a callable that builds data with the hypers,
    # if it returns only one thing it's `given`, if it returns two it's
    # `given` and `givencov`.
    
    # TODO add arguments/keyword arguments passed to makegp, and
    # keyword arguments passed to the minimizer.
    
    # TODO add correlation of the output gvars with the hyperprior, it should
    # be possible by using the implicit function theorem since we have both
    # the gradient and the inverse of the hessian.
    
    # TODO convert this function to a class, such that the raw minimizer
    # result is always accessible as an attribute.
    
    # TODO add the second order correction. It probably requires more than
    # the gradient and inv_hess, but maybe by getting a little help from
    # marginal_likelihood I can use the least-squares optimized second order
    # correction on the residuals term and invent something for the logdet
    # term.
    
    # TODO it raises very often with "Desired error not necessarily achieved
    # due to precision loss.". Change the default arguments of minimize to make
    # this less frequent, but only after implement the quad-specific
    # derivatives since maybe they will fix this. Another thing that may
    # matter is the vjp of the logdet, which computes a matrix inverse.
    # I could do the following: make an internal version of marginal_likelihood
    # that returns separately the residuals and the logdet term, and do
    # a backward derivative on the residuals and a forward on the logdet.
    
    hyperprior = _asarrayorbufferdict(hyperprior)
    flathp = _flat(hyperprior)
    hpcov = gvar.evalcov(flathp) # TODO use evalcov_blocks
    chol = linalg.cholesky(hpcov, lower=True)
    
    # TODO use a decomposition from _linalg after I've implemented
    # Decomposition.correlate.
    
    hpmean = gvar.mean(flathp)
    
    # TODO it may be inefficient that marginal_likelihood gets called every time
    # with the same data when data is not a callable because it uses evalcov
    # to extract the covariance of the data. It may easily become the
    # bottleneck with correlated data. Extract the covariance one time and
    # pass it to marginal_likelihood in `givencov`.
    
    def fun(p):
        gp = gpfactory(_unflat(p, hyperprior))
        assert isinstance(gp, _GP.GP)
        res = p - hpmean
        diagres = linalg.solve_triangular(chol, res, lower=True)
        return -gp.marginal_likelihood(data) + 1/2 * np.sum(np.square(diagres))
    
    result = optimize.minimize(autograd.value_and_grad(fun), hpmean, jac=True)
    if raises and not result.success:
        raise RuntimeError('minimization failed: {}'.format(result.message))
    uresult = gvar.gvar(result.x, result.hess_inv)
    return _unflat(uresult, hyperprior)
