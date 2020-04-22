from __future__ import division

import gvar
from autograd import numpy as np
from autograd.scipy import linalg
import autograd
from scipy import optimize

from . import _GP

__all__ = [
    'empbayes_fit'
]

def _asarrayorbufferdict(x):
    if isinstance(x, list):
        return np.array(x)
    elif isinstance(x, dict):
        return gvar.BufferDict(x)
    else:
        return x

def _flat(x):
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    elif isinstance(x, gvar.BufferDict):
        return x.buf
    else:
        raise TypeError('hyperprior must be array or dictionary of scalars/arrays')
    
_transf = {
    'arctanh': np.tanh,
    'log': np.exp
}
# TODO this should be integrated with gvar.BufferDict.add_distribution.

def _unflat(x, original, expand_transf):
    # this function must be autograd-friendly with x
    if isinstance(original, np.ndarray):
        return x.reshape(original.shape)
    elif isinstance(original, gvar.BufferDict):
        d = dict()
        for key in original:
            slic, shape = original.slice_shape(key)
            val = x[slic]
            if shape:
                val = val.reshape(shape)
            d[key] = val
            if expand_transf and isinstance(key, str):
                for transf in _transf:
                    if key.startswith(transf + '(') and key.endswith(')') and len(key) > 2 + len(transf):
                        d[key[len(transf) + 1:-1]] = _transf[transf](d[key])
        return d

def empbayes_fit(hyperprior, gpfactory, data):
    """
    Empirical bayes fit. Maximizes the marginal likelihood of the data with
    a gaussian process model that depends on hyperparameters.
    
    The function `gpfactory` must build and return a GP object using the
    hyperparameters. Although `hyperprior` must be a collection of `gvar`s,
    `gpfactory` receives ordinary scalars, and must operate on them using
    `autograd`'s numpy.
    
    Parameters
    ----------
    hyperprior : array or dictionary of arrays of gvars
        The prior for the hyperparameters.
    gpfactory : callable
        A function with signature gpfactory(hyperparams) -> GP object.
    data : dictionary
        Dictionary of data that is passed to GP.marginal_likelihood on the
        GP object returned by `gpfactory`.
    
    Returns
    -------
    hyperparams : array or dictionary of arrays of gvars
        The hyperparameters that maximize the marginal likelihood. The
        covariance matrix is computed as the inverse of the hessian of the
        marginal likelihood.
    """
    assert isinstance(data, (dict, gvar.BufferDict))
    assert callable(gpfactory)
    
    hyperprior = _asarrayorbufferdict(hyperprior)
    flathp = _flat(hyperprior)
    hpcov = gvar.evalcov(flathp) # TODO use gvar.evalcov_blocks
    chol = linalg.cholesky(hpcov, lower=True)
    hpmean = gvar.mean(flathp)
    
    def fun(p):
        gp = gpfactory(_unflat(p, hyperprior, True))
        assert isinstance(gp, _GP.GP)
        res = p - hpmean
        diagres = linalg.solve_triangular(chol, res, lower=True)
        return -gp.marginal_likelihood(data) + 1/2 * np.sum(diagres ** 2)
    
    result = optimize.minimize(autograd.value_and_grad(fun), hpmean, jac=True)
    uresult = gvar.gvar(result.x, result.hess_inv)
    shapedresult = _unflat(uresult, hyperprior, False)
    return _asarrayorbufferdict(shapedresult)
