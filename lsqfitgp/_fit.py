# lsqfitgp/_fit.py
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

import warnings

from ._imports import gvar
from ._imports import numpy as np
from ._imports import autograd
from ._imports import optimize

from . import _GP
from . import _linalg

__all__ = [
    'empbayes_fit'
]

@autograd.wrap_util.unary_to_nary
def jac(fun, x):
    """Like autograd.jacobian but with forward mode"""
    jvp = autograd.core.make_jvp(fun, x)
    ans = fun(x)
    vs = autograd.extend.vspace(x)
    grads = map(lambda b: jvp(b)[1], vs.standard_basis())
    return np.reshape(np.stack(grads, axis=-1), ans.shape + vs.shape)

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

class empbayes_fit:

    def __init__(self, hyperprior, gpfactory, data, raises=True, minkw={}, gpfactorykw={}):
        """
    
        Empirical bayes fit.
    
        Maximizes the marginal likelihood of the data with a Gaussian process
        model that depends on hyperparameters.
    
        Parameters
        ----------
        hyperprior : scalar, array or dictionary of scalars/arrays
            A collection of gvars representing the prior for the
            hyperparameters.
        gpfactory : callable
            A function with signature gpfactory(hyperparams) -> GP object. The
            argument `hyperparams` has the same structure of the empbayes_fit
            argument `hyperprior`. gpfactory must be autograd-friendly, i.e.,
            either use autograd.numpy, autograd.scipy, lsqfitgp.numpy,
            lsqfitgp.scipy or gvar instead of plain numpy/scipy.
        data : dict or callable
            Dictionary of data that is passed to `GP.marginal_likelihood` on
            the GP object returned by `gpfactory`. If a callable, it is called
            with the same arguments of `gpfactory` and must return either a
            dictionary or a pair of dictionaries where the second dictionary is
            passed as `givencov` argument to `GP.marginal_likelihood`.
        raises : bool, optional
            If True (default), raise an error when the minimization fails.
            Otherwise, use the last point of the minimization as result.
        minkw : dict, optional
            Keyword arguments passed to `scipy.optimize.minimize`.
        gpfactorykw : dict, optional
            Keyword arguments passed to `gpfactory`, and also to `data` if it
            is a callable.
    
        Attributes
        ----------
        p : scalar, array or dictionary of scalars/arrays
            A collection of gvars representing the hyperparameters that
            maximize the marginal likelihood. The covariance matrix is computed
            as the inverse of the hessian of the marginal likelihood. These
            gvars do not track correlations with the hyperprior or the data.
        minresult : scipy.optimize.OptimizeResult
            The result object returned by `scipy.optimize.minimize`.

        Raises
        ------
        RuntimeError
            The minimization failed and `raises` is True.
    
        """
        assert callable(gpfactory)
    
        # TODO would it be meaningful to add correlation of the fit result with
        # the data and hyperprior?
    
        # TODO add the second order correction. It probably requires more than
        # the gradient and inv_hess, but maybe by getting a little help from
        # marginal_likelihood I can use the least-squares optimized second order
        # correction on the residuals term and invent something for the logdet
        # term.
    
        # TODO it raises very often with "Desired error not necessarily
        # achieved due to precision loss.". Change the default arguments of
        # minimize to make this less frequent, but only after implement the
        # quad-specific derivatives since maybe they will fix this. Another
        # thing that may matter is the vjp of the logdet, which computes a
        # matrix inverse. I could do the following: make an internal version of
        # marginal_likelihood that returns separately the residuals and the
        # logdet term, and do a backward derivative on the residuals and a
        # forward on the logdet. Other option: try another optimization
        # algorithm.
        
        # TODO I don't know really how much the inverse hessian estimated by
        # BFGS is accurate. Investigate computing the hessian with autograd or
        # using Gauss-Newton on the residuals and autograd on the logdet.
        
        # TODO read the minkw options to adapt the usage of the jacobian: if
        # jac is specified or if the method does not require derivatives do not
        # take derivatives with autograd.
        
        # TODO compute the logGBF for the whole fit.
    
        hyperprior = _asarrayorbufferdict(hyperprior)
        flathp = _flat(hyperprior)
        hpmean = gvar.mean(flathp)
        hpcov = gvar.evalcov(flathp) # TODO use evalcov_blocks (low priority)
        hpdec = _linalg.EigCutFullRank(hpcov)
        
        if callable(data):
            cachedargs = None
        elif _flat(_asarrayorbufferdict(data)).dtype == object:
            data = gvar.gvar(data)
            datamean = gvar.mean(data)
            datacov = gvar.evalcov(data)
            cachedargs = (datamean, datacov)
        else:
            cachedargs = (data,)
    
        def fun(p):
            priorchi2 = hpdec.quad(p - hpmean)

            hp = _unflat(p, hyperprior)
            gp = gpfactory(hp, **gpfactorykw)
            assert isinstance(gp, _GP.GP)
            
            if cachedargs:
                args = cachedargs
            else:
                args = data(hp, **gpfactorykw)
                if not isinstance(args, tuple):
                    args = (args,)
            ml = gp.marginal_likelihood(*args)
            
            return -ml + 1/2 * priorchi2
    
        result = optimize.minimize(autograd.value_and_grad(fun), hpmean, jac=True, **minkw)
        
        if not result.success:
            msg = 'minimization failed: {}'.format(result.message)
            if raises:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)
        
        uresult = gvar.gvar(result.x, result.hess_inv)
        
        self.p = _unflat(uresult, hyperprior)
        self.minresult = result
