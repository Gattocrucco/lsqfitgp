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

import gvar
import jax
from jax import numpy as jnp
import numpy as np
from scipy import optimize

from . import _GP
from . import _linalg

__all__ = [
    'empbayes_fit',
]

def _asarrayorbufferdict(x):
    if hasattr(x, 'keys'):
        return gvar.BufferDict(x)
    else:
        return np.array(x, copy=False)

def _flat(x):
    if hasattr(x, 'reshape'):
        return x.reshape(-1)
    elif isinstance(x, gvar.BufferDict):
        return x.buf

def _unflat(x, original):
    if isinstance(original, np.ndarray):
        out = x.reshape(original.shape)
        return out if out.shape else out.item()
    elif isinstance(original, gvar.BufferDict):
        # normally I would do BufferDict(original, buf=x) but it does not work
        # with JAX tracers
        b = gvar.BufferDict(original)
        b._extension = {}
        b._buf = x
        # TODO b.buf = x does not work because BufferDict checks that the
        # array is a numpy array.
        return b

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
        # achieved due to precision loss.". I tried doing a forward grad on
        # the logdet but does not fix the problem. I still suspect it's the
        # logdet, maybe the value itself and not the derivative, because as the
        # matrix changes the regularization can change a lot the value of the
        # logdet. How do I stabilize it?
        
        # TODO use the split marginal likelihood to compute the gradient of
        # the logdet in forward mode.
        
        # TODO I don't know really how much the inverse hessian estimated by
        # BFGS is accurate. Investigate computing the hessian with autograd or
        # using Gauss-Newton on the residuals and autograd on the logdet.
        
        # TODO read the minkw options to adapt the usage of the jacobian: if
        # jac is specified or if the method does not require derivatives do not
        # take derivatives with autograd.
        
        # TODO compute the logGBF for the whole fit (see the gpbart code)
    
        hyperprior = _asarrayorbufferdict(hyperprior)
        flathp = _flat(hyperprior)
        hpmean = gvar.mean(flathp)
        hpcov = gvar.evalcov(flathp) # TODO use evalcov_blocks
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
            logp = -ml + 1/2 * priorchi2
            
            return logp, logp
        
        jac = jax.jacfwd(fun, has_aux=True)
        def value_and_grad(p):
            j, f = jac(p)
            return f, j
    
        result = optimize.minimize(value_and_grad, hpmean, jac=True, **minkw)
        
        if not result.success:
            msg = 'minimization failed: {}'.format(result.message)
            if raises:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)
        
        uresult = gvar.gvar(result.x, result.hess_inv)
        
        self.p = _unflat(uresult, hyperprior)
        self.minresult = result
