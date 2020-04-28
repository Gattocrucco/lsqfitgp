from ._imports import gvar
from ._imports import numpy as np
from ._imports import linalg
from ._imports import autograd
from ._imports import optimize

from . import _GP

__all__ = [
    'empbayes_fit'
]

# TODO accept iterables other than list? Like, just rely on np.array? And
# mappings other than dict and BufferDict?
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

def _unflat(x, original):
    if isinstance(original, np.ndarray):
        return x.reshape(original.shape)
    elif isinstance(original, gvar.BufferDict):
        return gvar.BufferDict(original, buf=x)

def empbayes_fit(hyperprior, gpfactory, data):
    """
    Empirical bayes fit. Maximizes the marginal likelihood of the data with
    a gaussian process model that depends on hyperparameters.
    
    Parameters
    ----------
    hyperprior : array or dictionary of arrays of gvars
        The prior for the hyperparameters.
    gpfactory : callable
        A function with signature gpfactory(hyperparams) -> GP object. The
        argument `hyperparams` has the same structure of the empbayes_fit
        argument `hyperprior`. gpfactory must be autograd-friendly, i.e.
        either use autograd.numpy, autograd.scipy, lsqfitgp.numpy or gvar
        instead of plain numpy.
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
    hpcov = gvar.evalcov(flathp)
    # TODO use gvar.evalcov_blocks when it becomes fast enough
    chol = linalg.cholesky(hpcov, lower=True)
    # TODO regularize hpcov in case it is not numerically positive definite,
    # maybe since this is typically small we could do a diagonalization
    hpmean = gvar.mean(flathp)
    
    def fun(p):
        gp = gpfactory(_unflat(p, hyperprior))
        assert isinstance(gp, _GP.GP)
        res = p - hpmean
        diagres = linalg.solve_triangular(chol, res, lower=True)
        return -gp.marginal_likelihood(data) + 1/2 * np.sum(diagres ** 2)
    
    result = optimize.minimize(autograd.value_and_grad(fun), hpmean, jac=True)
    uresult = gvar.gvar(result.x, result.hess_inv)
    return _unflat(uresult, hyperprior)
