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
import functools

import gvar
import jax
from jax import numpy as jnp
import numpy
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
        return numpy.asarray(x)

def _flat(x):
    if hasattr(x, 'reshape'):
        return x.reshape(-1)
    elif isinstance(x, gvar.BufferDict): # pragma: no branch
        return x.buf
    else:
        raise NotImplementedError # pragma: no cover

def _unflat(x, original):
    if isinstance(original, numpy.ndarray):
        out = x.reshape(original.shape)
        # if not out.shape:
        #     try:
        #         out = out.item()
        #     except jax.errors.ConcretizationTypeError:
        #         pass
        return out
    elif isinstance(original, gvar.BufferDict): # pragma: no branch
        # normally I would do BufferDict(original, buf=x) but it does not work
        # with JAX tracers
        b = gvar.BufferDict(original)
        b._extension = {}
        b._buf = x
        # TODO b.buf = x does not work because BufferDict checks that the
        # array is a numpy array.
        return b
    else:
        raise NotImplementedError # pragma: no cover

class empbayes_fit:

    def __init__(self, hyperprior, gpfactory, data, raises=True, minkw={}, gpfactorykw={}, jit=False, method='gradient'):
        """
    
        Maximum a posteriori fit.
    
        Maximizes the marginal likelihood of the data with a Gaussian process
        model that depends on hyperparameters, multiplied by a prior on the
        hyperparameters.
    
        Parameters
        ----------
        hyperprior : scalar, array or dictionary of scalars/arrays
            A collection of gvars representing the prior for the
            hyperparameters.
        gpfactory : callable
            A function with signature gpfactory(hyperparams) -> GP object. The
            argument `hyperparams` has the same structure of the empbayes_fit
            argument `hyperprior`. gpfactory must be JAX-friendly, i.e.,
            use jax.numpy and jax.scipy instead of plain numpy/scipy and avoid
            assignments to arrays.
        data : dict, tuple or callable
            Dictionary of data that is passed to `GP.marginal_likelihood` on
            the GP object returned by `gpfactory`. If a tuple, it contains the
            first two arguments to `GP.marginal_likelihood`. If a callable, it
            is called with the same arguments of `gpfactory` and must return
            the argument(s) for `GP.marginal_likelihood`.
        raises : bool, optional
            If True (default), raise an error when the minimization fails.
            Otherwise, use the last point of the minimization as result.
        minkw : dict, optional
            Keyword arguments passed to `scipy.optimize.minimize`.
        gpfactorykw : dict, optional
            Keyword arguments passed to `gpfactory`, and also to `data` if it
            is a callable.
        jit : bool
            If True, use jax's jit to compile the minimization target. Default
            False.
        method : str
            Minimization strategy. Options:
        
            'nograd'
                Use a gradient-free method.
            'gradient'
                Use a gradient-only method (default).
            'hessian'
                Use a Newton method with the Hessian.
            'fisher'
                Use a Newton method with the Fisher information matrix plus
                the hyperprior precision matrix.
            'hessmod'
                Use a Newton method with a modified Hessian where the
                second derivatives of the prior covariance matrix w.r.t. the
                hyperparameters are assumed to be zero.
    
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
        
        # TODO compute the logGBF for the whole fit (see the gpbart code)
        
        # Analyze the hyperprior.
        hyperprior = _asarrayorbufferdict(hyperprior)
        flathp = _flat(hyperprior)
        hpmean = gvar.mean(flathp)
        hpcov = gvar.evalcov(flathp) # TODO use evalcov_blocks
        hpdec = _linalg.EigCutFullRank(hpcov)
        precision = hpdec.inv()
        
        if isinstance(data, tuple) and len(data) == 1:
            data, = data
        
        if callable(data):
            cachedargs = None
        elif isinstance(data, tuple):
            cachedargs = data
        elif _flat(_asarrayorbufferdict(data)).dtype == object:
            data = gvar.gvar(data)
            datamean = gvar.mean(data)
            datacov = gvar.evalcov(data)
            cachedargs = (datamean, datacov)
        else:
            cachedargs = (data,)
        
        def make(p):
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
            
            return gp, args, priorchi2
        
        def dojit(f):
            return jax.jit(f) if jit else f
        
        @dojit
        def fun(p):
            gp, args, priorchi2 = make(p)
            ml = gp.marginal_likelihood(*args, stop_hessian=method == 'hessmod')
            logp = -ml + 1/2 * priorchi2
            return logp
                
        jac = dojit(jax.jacfwd(fun)) # can't change to rev due to, I guess, the priorchi2 term quad derivatives w.r.t. b
        hess = dojit(jax.jacfwd(jac))
        
        
        if not callable(data):
            
            @jax.jacfwd
            @jax.jacfwd
            # can't change inner jac to rev due to jax issue #10994
            def fisher(p):
                gp, args, _ = make(p)
                decomp, _ = gp._prior_decomp(*args, stop_hessian=True)
                return -1/2 * decomp.logdet()
            
            @dojit
            def fisherprec(p):
                return fisher(p) + precision
        
        else:
            
            # Must take into account that the data can depend on the        
            # hyperparameters. The statistical meaning is that the prior mean
            # depends on the hyperparameters.
            
            @functools.partial(jax.jacfwd, has_aux=True)
            def grad_logdet_and_aux(p):
                gp, args, _ = make(p)
                decomp, ymean = gp._prior_decomp(*args, stop_hessian=True)
                return -1/2 * decomp.logdet(), (decomp, ymean)
            
            @functools.partial(jax.jacfwd, has_aux=True)
            def fisher_and_jac_and_aux(p):
                gld, (decomp, ymean) = grad_logdet_and_aux(p)
                return (gld, ymean), decomp
            
            @dojit
            def fisherprec(p):
                (F, J), decomp = fisher_and_jac_and_aux(p)
                return F + precision + decomp.quad(J)
                            
        # TODO instead of recomputing everything many times, I can use nested
        # has_aux appropriately to compute all the things I need at once. The
        # problem is that scipy currently does not allow me to provide a
        # function that computes value, jacobian and hessian at once, only value
        # and jacobian. => Another problem is that the jacobian and hessian
        # need not be computed all the times, see scipy issue #9265. Check
        # if using value_and_jac is more efficient.
        
        args = (fun,)
        kwargs = dict(x0=hpmean, jac=jac)
        
        if method == 'nograd':
            kwargs.pop('jac')
            kwargs.update(method='nelder-mead')
        elif method == 'gradient':
            kwargs.update(method='bfgs')
        elif method == 'hessian':
            kwargs.update(hess=hess, method='trust-exact')
        elif method == 'fisher':
            kwargs.update(hess=fisherprec, method='dogleg')
            # dogleg requires positive definiteness
        elif method == 'hessmod':
            kwargs.update(hess=hess, method='trust-exact')
        else:
            raise KeyError(method)
        
        kwargs.update(minkw)
        result = optimize.minimize(*args, **kwargs)
        
        if not result.success:
            msg = 'minimization failed: {}'.format(result.message)
            if raises:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)
        
        if hasattr(result, 'hess_inv'):
            cov = result.hess_inv
        elif hasattr(result, 'hess'):
            hessdec = _linalg.EigCutFullRank(result.hess)
            cov = hessdec.inv()
        else:
            cov = jnp.full_like(hpcov, jnp.nan)
        
        uresult = gvar.gvar(result.x, cov)
        
        self.p = _unflat(uresult, hyperprior)
        self.minresult = result
