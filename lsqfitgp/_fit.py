# lsqfitgp/_fit.py
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

import warnings
import functools
import time
import textwrap
import datetime

import gvar
import jax
from jax import numpy as jnp
import numpy
from scipy import optimize

from . import _GP
from . import _linalg
from . import _patch_jax
from . import _patch_gvar

__all__ = [
    'empbayes_fit',
]

class Logger:
    
    def __init__(self, verbosity=0):
        self._verbosity = verbosity
    
    def indent(self, text, level=None):
        """ indent a text by provided level or by global current level """
        if level is None:
            level = self.loglevel._level
        prefix = 4 * level * ' '
        return textwrap.indent(text, prefix)
    
    def log(self, message, verbosity=1, level=None):
        """ print a message """
        if verbosity > self._verbosity:
            return
        print(self.indent(message, level))
    
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
        covariance='auto',
        fix=None,
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
        covariance : str
            Method to estimate the posterior covariance matrix of the
            hyperparameters:
    
            'hess'
                Use the hessian of the log posterior in the MAP as precision
                matrix.
            'fisher'
                Use the Fisher information in the MAP, plus the prior precision,
                as precision matrix.
            'minhess'
                Use the hessian returned by the minimizer as precision matrix,
                may be an estimate.
            'auto' (default)
                Use the minimizer hessian if applicable, Fisher otherwise.
            'none'
                Do not estimate the covariance matrix.
        fix : scalar, array or dictionary of scalars/arrays
            A set of booleans, with the same format as `hyperprior`, indicating
            which hyperparameters are kept fixed to their initial value.
            Scalars and arrays are broadcasted to the shape of `hyperprior`.
            If a dictionary, missing keys are treated as False.
    
        Attributes
        ----------
        p : scalar, array or dictionary of scalars/arrays
            A collection of gvars representing the hyperparameters that
            maximize their posterior. These gvars do not track correlations
            with the hyperprior or the data.
        prior : scalar, array or dictionary of scalars/arrays
            A copy of the hyperprior.
        initial : scalar, array or dictionary of scalars/arrays
            Starting point of the minimization, with the same format as `p`.
        fix : scalar, array or dictionary of scalars/arrays
            A set of booleans, with the same format as `p`, indicating which
            parameters were kept fixed to the values in `initial`.
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
        
        # analyze the hyperprior
        hpinitial, hpunflat = self._parse_hyperprior(hyperprior, initial, fix)
        del hyperprior, initial, fix
        
        # analyze data
        data, cachedargs = self._parse_data(data)
        # TODO log number of datapoints (not trivial, must be done in callback)
        
        def make(p):
            
            hp = hpunflat(p)
            gp = gpfactory(hp, **gpfactorykw)
            assert isinstance(gp, _GP.GP)
            
            if cachedargs:
                args = cachedargs
            else:
                args = data(hp, **gpfactorykw)
                if not isinstance(args, tuple):
                    args = (args,)
            
            return gp, args
        
        def dojit(f):
            return jax.jit(f) if jit else f
        
        @dojit
        def fun(p, *, stop_hessian=False):
            gp, args = make(p)
            ml = gp.marginal_likelihood(*args, stop_hessian=stop_hessian)
            logp = -ml + 1/2 * (p @ p)
            return logp
        
        fun_and_jac = dojit(_patch_jax.value_and_ops(fun, jax.jacrev))
        if method == 'hessmod':
            hess = dojit(jax.jacfwd(jax.jacfwd(functools.partial(fun, stop_hessian=True))))
            # can't change inner jacfwd to jacrev due to jax issue #10994
            # (stop_hessian)
        else:
            hess = dojit(jax.jacfwd(jax.jacrev(fun)))
                    
        if not callable(data):
            
            @jax.jacfwd
            @jax.jacfwd
            # can't change inner jac to rev due to jax issue #10994
            # (stop_hessian)
            def fisher(p):
                gp, args = make(p)
                decomp, _ = gp._prior_decomp(*args, stop_hessian=True)
                return -1/2 * decomp.logdet()
            
            def fisherprec(p):
                f = fisher(p)
                return f + jnp.eye(len(f), dtype=f.dtype)
        
        else:
            
            # Must take into account that the data can depend on the        
            # hyperparameters. The statistical meaning is that the prior mean
            # depends on the hyperparameters.
            
            @functools.partial(jax.jacfwd, has_aux=True)
            def grad_logdet_and_aux(p):
                gp, args = make(p)
                decomp, ymean = gp._prior_decomp(*args, stop_hessian=True)
                return -1/2 * decomp.logdet(), (decomp, ymean)
            
            @functools.partial(jax.jacfwd, has_aux=True)
            def fisher_and_jac_and_aux(p):
                gld, (decomp, ymean) = grad_logdet_and_aux(p)
                return (gld, ymean), decomp
            
            def fisherprec(p):
                (F, J), decomp = fisher_and_jac_and_aux(p)
                return F + jnp.eye(len(F), dtype=F.dtype) + decomp.quad(J)
        
        if method == 'fisher':
            fisherprec = dojit(fisherprec)
            # fisherprec may be used once for estimation of the posterior
            # precision, so do not jit unless it's used multiple times since
            # jit is somewhat slow
        
        # wrap functions to count number of calls
        fun = self._CountCalls(fun)
        fun_and_jac = self._CountCalls(fun_and_jac)
        hess = self._CountCalls(hess)
        fisherprec = self._CountCalls(fisherprec)
        
        # prepare minimizer arguments
        minargs = dict(fun=fun_and_jac, jac=True, x0=hpinitial)
        if method == 'nograd':
            minargs.update(fun=fun, jac=None, method='nelder-mead')
        elif method == 'gradient':
            minargs.update(method='bfgs')
        elif method == 'hessian':
            minargs.update(hess=hess, method='trust-exact')
        elif method == 'fisher':
            minargs.update(hess=fisherprec, method='dogleg')
            # dogleg requires positive definiteness
        elif method == 'hessmod':
            minargs.update(hess=hess, method='trust-exact')
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
                duration = datetime.timedelta(seconds=now - self.stamp)
                self.stamp = now
                nicep = hpunflat(p)
                self.this.log(f'iteration {self.it}, time: {duration}, calls: fun {fun.partial()}, funjac {fun_and_jac.partial()}, fisher {fisherprec.partial()}, hess {hess.partial()}', 3)
                self.this.log(f'parameters = {nicep}', 4)
                
            # TODO write a method to format the parameters nicely. => use
            # gvar.tabulate? => nope, need actual gvars
            
        if verbosity > 2:
            minargs.update(callback=Callback(self))
        
        minargs.update(minkw)
        with self.loglevel:
            start = time.time()
            result = optimize.minimize(**minargs)
            end = time.time()
        total = datetime.timedelta(seconds=end - start)
        self.log(f'totals: time: {total}, calls: fun {fun.total()}, funjac {fun_and_jac.total()}, fisher {fisherprec.total()}, hess {hess.total()}')
        
        # check the minimization was successful
        if not result.success:
            msg = 'minimization failed: {}'.format(result.message)
            if raises:
                raise RuntimeError(msg)
            elif verbosity == 0:
                warnings.warn(msg)
            else:
                self.log(msg)
        
        # determine method to compute posterior covariance
        if covariance == 'auto':
            if hasattr(result, 'hess_inv') or hasattr(result, 'hess'):
                covariance = 'minhess'
            else:
                covariance = 'fisher'
        
        # compute posterior covariance of the hyperparameters
        if covariance == 'fisher':
            self.log('use fisher plus prior precision as precision', 2)
            if method == 'fisher':
                prec = result.hess
            else:
                prec = fisherprec(result.x)
            cov = _linalg.EigCutFullRank(prec).inv()
        elif covariance == 'hess':
            self.log('use log posterior hessian as precision', 2)
            if method == 'hess':
                prec = result.hess
            else:
                prec = hess(result.x)
            cov = _linalg.EigCutFullRank(prec).inv()
        elif covariance == 'minhess':
            if hasattr(result, 'hess_inv'):
                self.log('use minimizer estimate of inverse hessian as covariance', 2)
                cov = result.hess_inv
            elif hasattr(result, 'hess'):
                self.log('use minimizer hessian as precision', 2)
                cov = _linalg.EigCutFullRank(result.hess).inv()
            else:
                raise RuntimeError('the minimizer did not return an estimate of the hessian')
        elif covariance == 'none':
            cov = numpy.full(result.x.size, numpy.nan)
        else:
            raise KeyError(covariance)
        
        uresult = gvar.gvar(result.x, cov)
        
        self.p = gvar.gvar(hpunflat(uresult))
        self.pmean = gvar.mean(self.p)
        self.pcov = gvar.evalcov(self.p)
        self.minresult = result
        self.minargs = minargs

        if verbosity >= 2:
            self.log(_patch_gvar.tabulate_together(
                self.prior, gvar.gvar(self.initial), self.p, self.p - self.prior, self.p - self.initial,
                headers=['param', 'prior', 'initial', 'posterior', 'post-prior', 'post-ini'],
            ))
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
        
        # TODO now that I have Decomposition.matrix(), I could write
        # by hand the gradient and Fisher matrix expressions to save on jax
        # tracing time. => wait for the new linalg system => don't wait, make
        # it non-crap right away!
    
    class _CountCalls:
        
        def __init__(self, func):
            self._func = func
            self._total = 0
            self._partial = 0
        
        def __call__(self, *args, **kw):
            self._total += 1
            self._partial += 1
            return self._func(*args, **kw)
        
        def partial(self):
            result = self._partial
            self._partial = 0
            return result
        
        def total(self):
            return self._total
    
    def _parse_hyperprior(self, hyperprior, initial, fix):
        
        # check fix against hyperprior and fill missing values
        hyperprior = self._copyasarrayorbufferdict(hyperprior)
        self._check_no_redundant_keys(hyperprior)
        fix = self._parse_fix(hyperprior, fix)
        flatfix = self._flatview(fix)

        # extract distribution of free hyperparameters
        flathp = self._flatview(hyperprior)
        freehp = flathp[~flatfix]
        mean = gvar.mean(freehp)
        cov = gvar.evalcov(freehp) # TODO use evalcov_blocks
        dec = _linalg.EigCutFullRank(cov)
        assert dec.n == freehp.size
        self.log(f'{freehp.size}/{flathp.size} free hyperparameters', 2)
        
        # determine starting point for minimization
        initial = self._parse_initial(hyperprior, initial, dec)
        flatinitial = self._flatview(initial)
        x0 = dec.decorrelate(flatinitial[~flatfix] - mean)
        # TODO for initial = 'priormean', x0 is zero, skip decorrelate
        # for initial = 'priorsample', x0 is iid normal, but I have to sync
        # it with the user-exposed unflattened initial in _parse_initial
        
        # make function to correlate, add fixed values, and reshape to original
        # format
        fixed_indices, = jnp.nonzero(flatfix)
        unfixed_indices, = jnp.nonzero(~flatfix)
        fixed_values = jnp.asarray(flatinitial[flatfix])
        def unflat(x):
            assert x.ndim == 1
            if x.dtype == object:
                jac, indices = _patch_gvar.jacobian(x)
                xmean = mean + dec.correlate(gvar.mean(x))
                xjac = dec.correlate(jac)
                x = _patch_gvar.from_jacobian(xmean, xjac, indices)
                y = numpy.empty(flatfix.size, x.dtype)
                y[unfixed_indices] = x
                y[fixed_indices] = fixed_values
            else:
                x = mean + dec.correlate(x)
                y = jnp.empty(flatfix.size, x.dtype)
                y = y.at[unfixed_indices].set(x)
                y = y.at[fixed_indices].set(fixed_values)
            return self._unflatview(y, hyperprior)
        
        self.prior = hyperprior
        return x0, unflat
    
    @staticmethod
    def _check_no_redundant_keys(hyperprior):
        if not hasattr(hyperprior, 'keys'):
            return
        for k in hyperprior:
            m = hyperprior.extension_pattern.match(k)
            if m and m.group(1) in hyperprior.invfcn:
                altk = m.group(2)
                assert altk not in hyperprior, f'duplicate keys {altk!r} and {k!r} in hyperprior'

    def _parse_fix(self, hyperprior, fix):
        
        if fix is None:
            if hasattr(hyperprior, 'keys'):
                fix = gvar.BufferDict(hyperprior, buf=numpy.zeros(hyperprior.size, bool))
            else:
                fix = numpy.zeros(hyperprior.shape, bool)
        else:
            fix = self._copyasarrayorbufferdict(fix)
            if hasattr(fix, 'keys'):
                assert hasattr(hyperprior, 'keys'), 'fix is dictionary but hyperprior is array'
                assert all(hyperprior.has_dictkey(k) for k in fix), 'some keys in fix are missing in hyperprior'
                newfix = {}
                for k, v in hyperprior.items():
                    key = None
                    m = hyperprior.extension_pattern.match(k)
                    if m and m.group(1) in hyperprior.invfcn:
                        altk = m.group(2)
                        if altk in fix:
                            assert k not in fix, f'duplicate keys {k!r} and {altk!r} in fix'
                            key = altk
                    if key is None and k in fix:
                        key = k
                    if key is None:
                        elem = numpy.zeros(v.shape, bool)
                    else:
                        elem = numpy.broadcast_to(fix[key], v.shape)
                    newfix[k] = elem
                fix = gvar.BufferDict(newfix, dtype=bool)
            else:
                assert not hasattr(hyperprior, 'keys'), 'fix is array but hyperprior is dictionary'
                fix = numpy.broadcast_to(fix, hyperprior.shape).astype(bool)
        
        self.fix = fix
        return fix
    
    def _parse_initial(self, hyperprior, initial, dec):
        
        if not isinstance(initial, str):
            self.log('start from provided point', 2)
            initial = self._copyasarrayorbufferdict(initial)
            if hasattr(hyperprior, 'keys'):
                assert hasattr(initial, 'keys'), 'hyperprior is dictionary but initial is array'
                assert set(hyperprior.keys()) == set(initial.keys())
                assert all(hyperprior[k].shape == initial[k].shape for k in hyperprior)
            else:
                assert not hasattr(initial, 'keys'), 'hyperprior is array but initial is dictionary'
                assert hyperprior.shape == initial.shape
        
        elif initial == 'priormean':
            self.log('start from prior mean', 2)
            initial = gvar.mean(hyperprior)
        
        elif initial == 'priorsample':
            self.log('start from a random sample from the prior', 2)
            if dec.size < hyperprior.size:
                flathp = self._flatview(hyperprior)
                cov = gvar.evalcov(flathp) # TODO use evalcov_blocks
                fulldec = _linalg.EigCutFullRank(cov)
            else:
                fulldec = dec
            iid = numpy.random.randn(fulldec.m)
            flatinitial = fulldec.correlate(iid)
            initial = self._unflatview(flatinitial, hyperprior)
        
        else:
            raise KeyError(initial)
        
        self.initial = initial
        return initial
    
    def _parse_data(self, data):
        
        if isinstance(data, tuple) and len(data) == 1:
            data, = data

        if callable(data):
            self.log('data is callable', 2)
            cachedargs = None
        elif isinstance(data, tuple):
            self.log('data errors provided separately', 2)
            assert len(data) == 2
            cachedargs = data
        elif self._flatview(self._copyasarrayorbufferdict(data)).dtype == object:
            self.log('data has errors as gvars', 2)
            data = gvar.gvar(data)
            datamean = gvar.mean(data)
            datacov = gvar.evalcov(data)
            cachedargs = (datamean, datacov)
        else:
            self.log('data has no errors', 2)
            cachedargs = (data,)
        
        return data, cachedargs

    @staticmethod
    def _copyasarrayorbufferdict(x):
        if hasattr(x, 'keys'):
            return gvar.BufferDict(x)
        else:
            return numpy.array(x)

    @staticmethod
    def _flatview(x):
        if hasattr(x, 'reshape'):
            return x.reshape(-1)
        elif hasattr(x, 'buf'):
            return x.buf
        else: # pragma: no cover
            raise NotImplementedError

    @staticmethod
    def _unflatview(x, original):
        if isinstance(original, numpy.ndarray):
            # TODO is this never applied to jax arrays?
            out = x.reshape(original.shape)
            # if not out.shape:
            #     try:
            #         out = out.item()
            #     except jax.errors.ConcretizationTypeError:
            #         pass
            return out
        elif isinstance(original, gvar.BufferDict):
            # normally I would do BufferDict(original, buf=x) but it does not
            # work with JAX tracers
            b = gvar.BufferDict(original)
            b._extension = {}
            b._buf = x
            # b.buf = x does not work because BufferDict checks that the
            # array is a numpy array
            # TODO maybe make a feature request to gvar to accept array_like
            # buf
            return b
        else: # pragma: no cover
            raise NotImplementedError
