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
import time

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

class Logger:
    
    def __init__(self, verbosity=0):
        self._verbosity = verbosity
    
    def log(self, message, verbosity=1, level=None):
        """print a message"""
        if verbosity > self._verbosity:
            return
        if level is None:
            level = self.loglevel._level
        indent = 4 * level * ' '
        print(f'{indent}{message}')
    
    class _LogLevel:
        """shared context manager to indent messages"""
        
        _level = 0
                
        @classmethod
        def __enter__(cls):
            cls._level += 1
        
        @classmethod
        def __exit__(cls, *_):
            cls._level -= 1
    
    loglevel = _LogLevel()

class empbayes_fit(Logger):

    def __init__(
        self,
        hyperprior,
        gpfactory,
        data,
        raises=True,
        minkw={},
        gpfactorykw={},
        jit=False,
        method='gradient',
        initial='priormean',
        verbosity=0,
    ):
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
        initial : str, scalar, array, dictionary of scalars/arrays
            Starting point for the minimization, matching the format of
            `hyperprior`, or one of the following options:
            
            'priormean'
                Start from the hyperprior mean (default).
            'priorsample'
                Take a random sample from the hyperprior.
        verbosity : int
            An integer indicating how much information is printed on the
            terminal:
    
            0
                No logging (default).
            1
                Report starting point and result.
            2
                More detailed report.
            3
                Log each iteration.
            4
                More detailed iteration log.
    
        Attributes
        ----------
        p : scalar, array or dictionary of scalars/arrays
            A collection of gvars representing the hyperparameters that
            maximize the marginal likelihood. The covariance matrix is computed
            as the inverse of the hessian of the marginal likelihood. These
            gvars do not track correlations with the hyperprior or the data.
        pmean : scalar, array or dictionary of scalars/arrays
            Mean of `p`.
        pcov : scalar, array or dictionary of scalars/arrays
            Covariance matrix of `p`.
        minresult : scipy.optimize.OptimizeResult
            The result object returned by `scipy.optimize.minimize`.
        minargs : dict
            The arguments passed to `scipy.optimize.minimize`.

        Raises
        ------
        RuntimeError
            The minimization failed and `raises` is True.
    
        """

        Logger.__init__(self, verbosity)
        self.log('**** call lsqfitgp.empbayes_fit ****')
    
        assert callable(gpfactory)
    
        # Analyze the hyperprior.
        hyperprior = self._asarrayorbufferdict(hyperprior)
        flathp = self._flat(hyperprior)
        hpmean = gvar.mean(flathp)
        hpcov = gvar.evalcov(flathp) # TODO use evalcov_blocks
        hpdec = _linalg.EigCutFullRank(hpcov)
        precision = hpdec.inv()
        self.log(f'{hpdec.n} hyperparameters', 2)
        
        if isinstance(data, tuple) and len(data) == 1:
            data, = data
        
        if callable(data):
            self.log('data is callable', 2)
            cachedargs = None
        elif isinstance(data, tuple):
            self.log('data errors provided separately', 2)
            assert len(data) == 2
            cachedargs = data
        elif self._flat(self._asarrayorbufferdict(data)).dtype == object:
            self.log('data has errors as gvars', 2)
            data = gvar.gvar(data)
            datamean = gvar.mean(data)
            datacov = gvar.evalcov(data)
            cachedargs = (datamean, datacov)
        else:
            self.log('data has no errors', 2)
            cachedargs = (data,)
        
        def make(p):
            priorchi2 = hpdec.quad(p - hpmean)

            hp = self._unflat(p, hyperprior)
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
        # TODO a reverse mode jac would be very useful with many hyperparameters
        # and grad-only optimization.
        
        if not callable(data):
            
            @jax.jacfwd
            @jax.jacfwd
            # can't change inner jac to rev due to jax issue #10994
            # (stop_hessian)
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
        
        kwargs = dict(fun=fun, jac=jac)
        
        if not isinstance(initial, str):
            self.log('start from provided point', 2)
            initial = self._asarrayorbufferdict(initial)
            flatinitial = self._flat(initial)
            kwargs.update(x0=flatinitial)
        elif initial == 'priormean':
            self.log('start from prior mean', 2)
            kwargs.update(x0=hpmean)
        elif initial == 'priorsample':
            self.log('start from a random sample from the prior', 2)
            iid = np.random.randn(hpdec.n)
            x0 = hpdec.correlate(iid)
            kwargs.update(x0=x0)
        else:
            raise KeyError(initial)
        
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
        self.log(f'minimization method {method!r}', 2)
        
        class Callback:
            
            def __init__(self, this):
                self.it = 0
                self.stamp = time.time()
                self.this = this
        
            def __call__(self, p):
                self.it += 1
                now = time.time()
                duration = now - self.stamp
                self.stamp = now
                nicep = self._unflat(p, hyperprior)
                self.this.log(f'iteration {self.it} ({duration:.2g} s)', 3)
                self.this.log(f'parameters = {nicep}', 4)
                
            # TODO write a method to format the parameters nicely.

        if verbosity > 2:
            kwargs.update(callback=Callback(self))
        
        kwargs.update(minkw)
        with self.loglevel:
            start = time.time()
            result = optimize.minimize(**kwargs)
            end = time.time()
            total = end - start
        
        if not result.success:
            msg = 'minimization failed: {}'.format(result.message)
            if raises:
                raise RuntimeError(msg)
            elif verbosity == 0:
                warnings.warn(msg)
            else:
                self.log(msg)
        
        if hasattr(result, 'hess_inv'):
            self.log('use minimizer estimate of inverse hessian as covariance', 2)
            cov = result.hess_inv
        elif hasattr(result, 'hess'):
            self.log('use minimizer hessian as covariance', 2)
            hessdec = _linalg.EigCutFullRank(result.hess)
            cov = hessdec.inv()
        else:
            self.log('no covariance information', 2)
            cov = jnp.full_like(hpcov, jnp.nan)
        
        # TODO allow the user to choose the covariance estimation independently
        # of the minimization method, possibly with methods or cached
        # properties. Also give the Fisher and Hessian even if not used.
        
        uresult = gvar.gvar(result.x, cov)
        
        self.p = self._unflat(uresult, hyperprior)
        self.pmean = gvar.mean(self.p)
        self.pcov = gvar.evalcov(self.p)
        self.minresult = result
        self.minargs = kwargs

        self.log(f'posterior = {self.p}')
        self.log(f'prior = {hyperprior}', 2)
        self.log(f'total time: {total:.2g} s')
        self.log('**** exit lsqfitgp.empbayes_fit ****')

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
        
        # TODO instead of recomputing everything many times, I can use nested
        # has_aux appropriately to compute all the things I need at once. The
        # problem is that scipy currently does not allow me to provide a
        # function that computes value, jacobian and hessian at once, only value
        # and jacobian. => Another problem is that the jacobian and hessian
        # need not be computed all the times, see scipy issue #9265. Check
        # if using value_and_jac is more efficient.

    @staticmethod
    def _asarrayorbufferdict(x):
        if hasattr(x, 'keys'):
            return gvar.BufferDict(x)
        else:
            return numpy.asarray(x)

    @staticmethod
    def _flat(x):
        if hasattr(x, 'reshape'):
            return x.reshape(-1)
        elif isinstance(x, gvar.BufferDict): # pragma: no branch
            return x.buf
        else:
            raise NotImplementedError # pragma: no cover

    @staticmethod
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

