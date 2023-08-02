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

import re
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
from jax import tree_util

from . import _GP
from . import _linalg
from . import _patch_jax
from . import _patch_gvar
from . import _array

__all__ = [
    'empbayes_fit',
]

# TODO the following token_ thing functionality may be provided by jax in the
# future, follow the developments

@functools.singledispatch
def token_getter(x):
    return x

@functools.singledispatch
def token_setter(x, token):
    return token

@token_getter.register(jnp.ndarray)
@token_getter.register(numpy.ndarray)
def _(x):
    return x[x.ndim * (0,)] if x.size else x

@token_setter.register(jnp.ndarray)
@token_setter.register(numpy.ndarray)
def _(x, token):
    x = jnp.asarray(x)
    return x.at[x.ndim * (0,)].set(token) if x.size else token

def token_map_leaf(func, x):
    if isinstance(x, (jnp.ndarray, numpy.ndarray)):
        token = token_getter(x)
        @jax.custom_jvp
        def jaxfunc(token):
            return jax.pure_callback(func, token, token, vectorized=True)
        @jaxfunc.defjvp
        def _(p, t):
            return (jaxfunc(*p), *t)
        token = jaxfunc(token)
        return token_setter(x, token)
    else:
        token = token_getter(x)
        token = func(token)
        return token_setter(x, token)

def token_map(func, x):
    return tree_util.tree_map(lambda x: token_map_leaf(func, x), x)

class Logger:
    
    def __init__(self, verbosity=0):
        self._verbosity = verbosity
    
    def _indent(self, text, level=0):
        """ indent a text by provided level or by global current level """
        level = max(0, level + self.loglevel._level)
        prefix = 4 * level * ' '
        return textwrap.indent(text, prefix)
    
    def log(self, message, verbosity=1, *, level=0):
        """ print a message """
        if verbosity > self._verbosity:
            return
        print(self._indent(message, level))

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

    SEPARATE_JAC = False

    def __init__(
        self,
        hyperprior,
        gpfactory,
        data,
        *,
        raises=True,
        minkw={},
        gpfactorykw={},
        jit=True,
        method='gradient',
        initial='priormean',
        verbosity=0,
        covariance='auto',
        fix=None,
        mlkw={},
        forward=False,
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
            argument ``hyperparams`` has the same structure of the empbayes_fit
            argument ``hyperprior``. gpfactory must be JAX-friendly, i.e.,
            use jax.numpy and jax.scipy instead of plain numpy/scipy and avoid
            assignments to arrays.
        data : dict, tuple or callable
            Dictionary of data that is passed to `GP.marginal_likelihood` on
            the GP object returned by ``gpfactory``. If a tuple, it contains the
            first two arguments to `GP.marginal_likelihood`. If a callable, it
            is called with the same arguments of ``gpfactory`` and must return
            the argument(s) for `GP.marginal_likelihood`.
        raises : bool, optional
            If True (default), raise an error when the minimization fails.
            Otherwise, use the last point of the minimization as result.
        minkw : dict, optional
            Keyword arguments passed to `scipy.optimize.minimize`, overwrites
            values specified by `empbayes_fit`.
        gpfactorykw : dict, optional
            Keyword arguments passed to ``gpfactory``, and also to ``data`` if
            it is a callable. If ``jit``, ``gpfactorykw`` crosses a JAX jit
            boundary, so it must contain objects understandable by JAX.
        jit : bool
            If True (default), use jax's jit to compile the minimization target.
        method : str
            Minimization strategy. Options:
        
            'nograd'
                Use a gradient-free method.
            'gradient'
                Use a gradient-only method (default).
            'fisher'
                Use a Newton method with the Fisher information matrix plus
                the hyperprior precision matrix.
        initial : str, scalar, array, dictionary of scalars/arrays
            Starting point for the minimization, matching the format of
            ``hyperprior``, or one of the following options:
            
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
                Minimal report.
            2
                Detailed report.
            3
                Log each iteration.
            4
                More detailed iteration log.
            5
                Print the current parameter values at each iteration.
        covariance : str
            Method to estimate the posterior covariance matrix of the
            hyperparameters:
    
            'fisher'
                Use the Fisher information in the MAP, plus the prior precision,
                as precision matrix.
            'minhess'
                Use the hessian estimate of the minimizer as precision matrix.
            'none'
                Do not estimate the covariance matrix.
            'auto' (default)
                'minhess' if applicable, 'none' otherwise.
        fix : scalar, array or dictionary of scalars/arrays
            A set of booleans, with the same format as ``hyperprior``,
            indicating which hyperparameters are kept fixed to their initial
            value. Scalars and arrays are broadcasted to the shape of
            ``hyperprior``. If a dictionary, missing keys are treated as False.
        mlkw : dict
            Additional arguments passed to `GP.marginal_likelihood`.
        forward : bool
            Use forward instead of backward (default) derivatives.
    
        Attributes
        ----------
        p : scalar, array or dictionary of scalars/arrays
            A collection of gvars representing the hyperparameters that
            maximize their posterior. These gvars do not track correlations
            with the hyperprior or the data.
        prior : scalar, array or dictionary of scalars/arrays
            A copy of the hyperprior.
        initial : scalar, array or dictionary of scalars/arrays
            Starting point of the minimization, with the same format as ``p``.
        fix : scalar, array or dictionary of scalars/arrays
            A set of booleans, with the same format as ``p``, indicating which
            parameters were kept fixed to the values in ``initial``.
        pmean : scalar, array or dictionary of scalars/arrays
            Mean of ``p``.
        pcov : scalar, array or dictionary of scalars/arrays
            Covariance matrix of ``p``.
        minresult : scipy.optimize.OptimizeResult
            The result object returned by `scipy.optimize.minimize`.
        minargs : dict
            The arguments passed to `scipy.optimize.minimize`.
        gpfactory : callable
            The ``gpfactory`` argument.
        gpfactorykw : dict
            The ``gpfactorykw`` argument.
        data : dict, tuple or callable
            The ``data`` argument.

        Raises
        ------
        RuntimeError
            The minimization failed and ``raises`` is True.
    
        """

        # TODO parameter float32: bool to use short float type. I think that
        # scipy's optimize may break down with short floats with default
        # options, I hope that changing termination tolerances does the trick.

        Logger.__init__(self, verbosity)
        del verbosity
        self.log('**** call lsqfitgp.empbayes_fit ****')
    
        assert callable(gpfactory)
        
        # analyze the hyperprior
        hpinitial, hpunflat = self._parse_hyperprior(hyperprior, initial, fix)
        del hyperprior, initial, fix
        
        # analyze data
        data, cachedargs = self._parse_data(data)

        # define functions
        timer, functions = self._prepare_functions(
            gpfactory=gpfactory, gpfactorykw=gpfactorykw, data=data,
            cachedargs=cachedargs, hpunflat=hpunflat, mlkw=mlkw, jit=jit,
            forward=forward,
        )
        del gpfactory, gpfactorykw, data, cachedargs, mlkw, forward

        # prepare minimizer arguments
        minargs = self._prepare_minargs(method, functions, hpinitial)

        # set up callback to time and log iterations
        callback = self._Callback(self, functions, timer, hpunflat)
        minargs.update(callback=callback)

        # check invalid argument before running minimizer
        if not covariance in ('auto', 'fisher', 'minhess', 'none'):
            raise KeyError(covariance)
        
        # add user arguments and minimize
        minargs.update(minkw)
        self.log(f'minimizer method {minargs["method"]!r}', 2)
        total = time.time()
        result = optimize.minimize(**minargs)

        # check the minimization was successful
        self._check_success(result, raises)
        
        # compute posterior covariance of the hyperparameters
        cov = self._posterior_covariance(method, covariance, result, functions['fisher'])    
        
        # log total timings and function calls
        total = time.time() - total
        self._log_totals(total, timer, callback, jit, functions)
        
        # join posterior mean and covariance matrix
        uresult = gvar.gvar(result.x, cov)
        
        # set attributes
        self.p = gvar.gvar(hpunflat(uresult))
        self.pmean = gvar.mean(self.p)
        self.pcov = gvar.evalcov(self.p)
        self.minresult = result
        self.minargs = minargs

        # tabulate hyperparameter prior and posterior
        if self._verbosity >= 2:
            self.log(_patch_gvar.tabulate_together(
                self.prior, self.p,
                headers=['param', 'prior', 'posterior'],
            )) # TODO replace tabulate_toegether with something more flexible I
            # can use for the callback as well. Maybe import TextMatrix from
            # miscpy.
            # TODO print the transformed parameters
        
        self.log('**** exit lsqfitgp.empbayes_fit ****')

        # TODO would it be meaningful to add correlation of the fit result with
        # the data and hyperprior?
    
        # TODO add the second order correction. It probably requires more than
        # the gradient and inv_hess, but maybe by getting a little help from
        # marginal_likelihood I can use the least-squares optimized second order
        # correction on the residuals term and invent something for the logdet
        # term.
    
        # TODO it raises very often with "Desired error not necessarily achieved
        # due to precision loss.". I tried doing a forward grad on the logdet
        # but does not fix the problem. I still suspect it's the logdet, maybe
        # the value itself and not the derivative, because as the matrix changes
        # the regularization can change a lot the value of the logdet. How do I
        # stabilize it? => scipy's l-bfgs-b seems to fail the linear search less
        # often
        
        # TODO compute the logGBF for the whole fit (see the gpbart code)
        
        # TODO empbayes_fit(autoeps=True) tries to double epsabs until the
        # minimization succedes, with some maximum number of tries.
        # autoeps=dict(maxrepeat=5, increaseby=2, initial=1e-16,
        # startfromzero=True) allows to configure the algorithm.

        # TODO empbayes_fit(maxiter=100) sets the maximum number of minimization
        # iterations. maxiter=dict(iter=100, calls=200, callsperiter=10) allows
        # to configure it more finely. The calls limits are cumulative on all
        # functions (need to make a class counter in _CountCalls), I can
        # probably implement them by returning nan when the limit is surpassed,
        # I hope the minimizer stops immediately on nan (test this).

        # TODO can I approximate the hessian with only function values and no
        # gradient, i.e., when using nelder-mead? => See Hare (2022), although I
        # would not know how to apply it properly to the optimization history.
        # Somehow I need to keep only the "last" iterations.

        # TODO is there a better algorithm than lbfgs for inaccurate functions?
        # consider SC-BFGS (https://github.com/frankecurtis/SCBFGS). See Basak 
        # (2022). And NonOpt (https://github.com/frankecurtis/NonOpt).

        # TODO can I estimate the error on the likelihood with the matrices? It
        # requires the condition number. Basak (2022) gives wide bounds. I could
        # try an upper bound and see how it compares to the true error, assuming
        # that the matrix was as ill-conditioned as possible, i.e., use eps as
        # the lowest eigenvalue, and gershgorin as the highest one.

        # TODO look into jaxopt: it has improved a lot since the last time I saw
        # it. In particular, it implements l-bfgs and has a "do not stop on
        # failed line search" option. And it probably supports float32.

        # TODO I could provide the option to use BFGS to estimate the covariance
        # independently of the minimizer, using the last n iterations (not all
        # the evaluations, just the iterations, to skip difficult line
        # searches.) covariance='bfgs(10)'. I still need, in any case, an
        # automatic policy of this sort for covariance='hessinv' when the linear
        # search fails.

        # TODO reimplement the timing system with host_callback.id_tap. It
        # should preserve the order because id_tap takes inputs and outputs. I
        # must take care to make all callbacks happen at runtime instead of
        # having some of them at compile time. I tried once but failed.
        # Currently host_callback is experimental, maybe wait until it isn't. =>
        # I think it fails because it's asynchronous and there is only one
        # device. Maybe host_callback.call would work?

        # TODO dictionary argument jitkw

    class _CountCalls:
        """ wrap a callable to count calls """
        
        def __init__(self, func):
            self._func = func
            self._total = 0
            self._partial = 0
            functools.update_wrapper(self, func)
        
        def __call__(self, *args, **kw):
            self._total += 1
            self._partial += 1
            return self._func(*args, **kw)
        
        def partial(self):
            """ return the partial counter and reset it """
            result = self._partial
            self._partial = 0
            return result
        
        def total(self):
            """ return the total number of calls """
            return self._total

        @staticmethod
        def fmtcalls(method, functions):
            """
            format summary of number of calls
            method : str
            functions: dict[str, _CountCalls]
            """
            def counts():
                for name, func in functions.items():
                    if count := getattr(func, method)():
                        yield f'{name} {count}'
            return ', '.join(counts())

    class _Timer:
        """ object to time likelihood computations """

        def __init__(self):
            self.totals = {}
            self.partials = {}
            self._last_start = False

        def start(self, token):
            return token_map(self._start, token)

        def _start(self, token):
            self.stamp = time.time()
            self.counter = 0
            assert not self._last_start # forbid consecutive start() calls
            self._last_start = True
            return token

        def reset(self):
            self.partials = {}

        def partial(self, token):
            return token_map(self._partial, token)

        def _partial(self, token):
            now = time.time()
            delta = now - self.stamp
            self.partials[self.counter] = self.partials.get(self.counter, 0) + delta
            self.totals[self.counter] = self.totals.get(self.counter, 0) + delta
            self.stamp = now
            self.counter += 1
            self._last_start = False
            return token
    
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
        dec = _linalg.Chol(cov)
        assert dec.n == freehp.size
        self.log(f'{freehp.size}/{flathp.size} free hyperparameters', 2)
        
        # determine starting point for minimization
        initial = self._parse_initial(hyperprior, initial, dec)
        flatinitial = self._flatview(initial)
        x0 = dec.pinv_correlate(flatinitial[~flatfix] - mean)
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
                if altk in hyperprior:
                    raise ValueError(f'duplicate keys {altk!r} and {k!r} in hyperprior')

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
            if dec.n < hyperprior.size:
                flathp = self._flatview(hyperprior)
                cov = gvar.evalcov(flathp) # TODO use evalcov_blocks
                fulldec = _linalg.Chol(cov)
            else:
                fulldec = dec
            iid = numpy.random.randn(fulldec.m)
            flatinitial = numpy.asarray(fulldec.correlate(iid))
            initial = self._unflatview(flatinitial, hyperprior)
        
        else:
            raise KeyError(initial)
        
        self.initial = initial
        return initial
    
    def _parse_data(self, data):
        
        self.data = data
        if isinstance(data, tuple) and len(data) == 1:
            data, = data

        if callable(data):
            self.log('data is callable', 2)
            cachedargs = None
        elif isinstance(data, tuple):
            self.log('data errors provided separately', 2)
            assert len(data) == 2
            cachedargs = data
        elif (gdata := self._copyasarrayorbufferdict(data)).dtype == object:
            self.log('data has errors as gvars', 2)
            data = gvar.gvar(gdata)
            # convert to gvar because non-gvars in the array would upset
            # gvar.mean and gvar.evalcov
            cachedargs = (gvar.mean(data), gvar.evalcov(data))
        else:
            self.log('data has no errors', 2)
            cachedargs = (data,)
        
        return data, cachedargs

    def _prepare_functions(self, *, gpfactory, gpfactorykw, data, cachedargs,
        hpunflat, mlkw, jit, forward):

        timer = self._Timer()
        firstcall = [None]
        
        def make_decomp(p, **kw):
            """ decomposition of the prior covariance and data """

            # start timer
            p = timer.start(p)

            # create GP object
            hp = hpunflat(p)
            gp = gpfactory(hp, **kw)
            assert gp.__class__.__name__ == 'GP'
                # avoid isinstance because it breaks under reloading
                # TODO maybe replace with custom isinstance?
            
            # extract data
            if cachedargs:
                args = cachedargs
            else:
                args = data(hp, **kw)
                if not isinstance(args, tuple):
                    args = (args,)

            # decompose covariance matrix and flatten data
            decomp, r = gp._prior_decomp(*args, covtransf=timer.partial, **mlkw)
            r = r.astype(float) # int data upsets jax
        
            # log number of datapoints
            if firstcall:
                # it is convenient to do here because the data is flattened.
                # works under jit since the first call is tracing
                firstcall.pop()
                xdtype = gp._get_x_dtype()
                nd = '?' if xdtype is None else _array._nd(xdtype)
                self.log(f'{r.size} datapoints, {nd} covariates')

            # split timer and return decomposition
            return timer.partial(decomp), r

        # define wrapper to collect call stats, pass user args, compile
        def wrap(func):
            if jit:
                func = jax.jit(func)
            func = functools.partial(func, **gpfactorykw)
            return self._CountCalls(func)
        if jit:
            self.log('compile functions with jax jit', 2)        
        
        # log derivation method
        modename = 'forward' if forward else 'reverse'
        self.log(f'{modename}-mode autodiff (if used)', 2)

        # TODO time the derivatives separately?

        def prior(p):
            # the prior is a normal with identity covariance matrix because p is
            # transformed to make it so
            return 1/2 * (len(p) * jnp.log(2 * jnp.pi) + p @ p)

        def grad_prior(p):
            return p

        def fisher_prior(p):
            return jnp.eye(len(p))

        @wrap
        def fun(p, **kw):
            """ minus log posterior on the hyperparameters (not normalized) """
            decomp, r = make_decomp(p, **kw)
            cond, _, _, _, _ = decomp.minus_log_normal_density(r, value=True)
            post = cond + prior(p)
            return timer.partial(post)

        def make_gradfwd_fisher_args(p, **kw):
            def make_decomp_tee(p):
                decomp, r = make_decomp(p, **kw)
                return (decomp.matrix(), r), (decomp, r)
            (dK, dr), (decomp, r) = jax.jacfwd(make_decomp_tee, has_aux=True)(p)
            lkw = dict(dK=dK, dr=dr)
            return decomp, r, lkw

        def make_gradrev_args(p, **kw):
            def make_decomp_r(p):
                def make_decomp_K(p):
                    decomp, r = make_decomp(p, **kw)
                    return decomp.matrix(), (decomp, r)
                _, dK_vjp, (decomp, r) = jax.vjp(make_decomp_K, p, has_aux=True)
                return r, (decomp, r, dK_vjp)
            _, dr_vjp, (decomp, r, dK_vjp) = jax.vjp(make_decomp_r, p, has_aux=True)
            unpack = lambda f: lambda x: f(x)[0]
            dK_vjp = unpack(dK_vjp)
            dr_vjp = unpack(dr_vjp)
            lkw = dict(dK_vjp=dK_vjp, dr_vjp=dr_vjp)
            return decomp, r, lkw

        def make_jac_args(p, **kw):
            if forward:
                decomp, r, lkw = make_gradfwd_fisher_args(p, **kw)
                lkw.update(gradfwd=True)
            else:
                decomp, r, lkw = make_gradrev_args(p, **kw)
                lkw.update(gradrev=True)
            return decomp, r, lkw

        @wrap
        def fun_and_jac(p, **kw):
            """ fun and its gradient """
            decomp, r, lkw = make_jac_args(p, **kw)
            cond, gradrev, gradfwd, _, _ = decomp.minus_log_normal_density(r, value=True, **lkw)
            post = cond + prior(p)
            grad_cond = gradfwd if forward else gradrev
            grad_post = grad_cond + grad_prior(p)
            return timer.partial((post, grad_post))
        
        @wrap
        def jac(p, **kw):
            """ gradient of fun """
            decomp, r, lkw = make_jac_args(p, **kw)
            _, gradrev, gradfwd, _, _ = decomp.minus_log_normal_density(r, **lkw)
            grad_cond = gradfwd if forward else gradrev
            grad_post = grad_cond + grad_prior(p)
            return timer.partial(grad_post)

        @wrap
        def fisher(p, **kw):
            """ fisher matrix """
            decomp, r, lkw = make_gradfwd_fisher_args(p, **kw)
            _, _, _, fisher_cond, _ = decomp.minus_log_normal_density(r, fisher=True, **lkw)
            fisher_post = fisher_cond + fisher_prior(p)
            return timer.partial(fisher_post)
        
        # set attributes
        self.gpfactory = gpfactory
        self.gpfactorykw = gpfactorykw

        return timer, {
            'fun': fun,
            'jac': jac,
            'fun&jac': fun_and_jac,
            'fisher': fisher,
        }

    def _prepare_minargs(self, method, functions, hpinitial):
        minargs = dict(fun=functions['fun&jac'], jac=True, x0=hpinitial)
        if self.SEPARATE_JAC:
            minargs.update(fun=functions['fun'], jac=functions['jac'])
        if method == 'nograd':
            minargs.update(fun=functions['fun'], jac=None, method='nelder-mead')
        elif method == 'gradient':
            minargs.update(method='bfgs')
        elif method == 'fisher':
            minargs.update(hess=functions['fisher'], method='dogleg')
            # dogleg requires positive definiteness, fisher is p.s.d.
            # trust-constr has more options, but it seems to be slower than
            # dogleg, so I keep dogleg as default
        else:
            raise KeyError(method)
        self.log(f'method {method!r}', 2)
        return minargs

        # TODO add method with fisher matvec instead of fisher matrix

    def _log_totals(self, total, timer, callback, jit, functions):
        times = {
            'gp&cov': timer.totals[0],
            'decomp': timer.totals[1],
            'likelihood': timer.totals[2],
            'jit': None, # set now and delete later to keep it before 'other'
            'other': total - sum(timer.totals.values()),
        }
        if jit:
            overhead = callback.estimate_firstcall_overhead()
            # TODO this estimation ignores the jit compilation of the function
            # used to compute the precision matrix, to be precise I should
            # manually split the jit into compilation + evaluation or hook into
            # it somehow. Maybe the jit object keeps a compilation wall time
            # stat?
        if jit and overhead is not None:
            times['jit'] = overhead
            times['other'] -= overhead
        else:
            del times['jit']
        self.log('', 4)
        calls = self._CountCalls.fmtcalls('total', functions)
        self.log(f'calls: {calls}')
        self.log(f'total time: {callback.fmttime(total)}')
        self.log(f'partials: {callback.fmttimes(times)}', 2)

    def _check_success(self, result, raises):
        if result.success:
            self.log(f'minimization succeeded: {result.message}')
        else:
            msg = f'minimization failed: {result.message}'
            if raises:
                raise RuntimeError(msg)
            elif self._verbosity == 0:
                warnings.warn(msg)
            else:
                self.log(msg)

    def _posterior_covariance(self, method, covariance, minimizer_result, fisher_func):
        
        if covariance == 'auto':
            if hasattr(minimizer_result, 'hess_inv') or hasattr(minimizer_result, 'hess'):
                covariance = 'minhess'
            else:
                covariance = 'none'

        if covariance == 'fisher':
            self.log('use fisher plus prior precision as precision', 2)
            if method == 'fisher':
                prec = minimizer_result.hess
            else:
                prec = fisher_func(minimizer_result.x)
            cov = _linalg.Chol(prec).ginv()

        elif covariance == 'minhess':
            if hasattr(minimizer_result, 'hess_inv'):
                hessinv = minimizer_result.hess_inv
                if isinstance(hessinv, optimize.LbfgsInvHessProduct):
                    self.log(f'convert LBFGS({hessinv.n_corrs}) hessian inverse to BFGS as covariance', 2)
                    cov = self._invhess_lbfgs_to_bfgs(hessinv)
                    # TODO this still gives a too wide cov when the minimization
                    # terminates due to bad linear search, is it because of
                    # dropped updates? This is currently keeping me from setting
                    # l-bfgs-b as default minimization method.
                elif isinstance(hessinv, numpy.ndarray):
                    self.log('use minimizer estimate of inverse hessian as covariance', 2)
                    cov = hessinv
            elif hasattr(minimizer_result, 'hess'):
                self.log('use minimizer hessian as precision', 2)
                cov = _linalg.Chol(minimizer_result.hess).ginv()
            else:
                raise RuntimeError('the minimizer did not return an estimate of the hessian')

        elif covariance == 'none':
            cov = numpy.full(minimizer_result.x.size, numpy.nan)

        else:
            raise KeyError(covariance)

        return cov

    @staticmethod
    def _invhess_lbfgs_to_bfgs(lbfgs):
        bfgs = optimize.BFGS()
        bfgs.initialize(lbfgs.shape[0], 'inv_hess')
        for i in range(lbfgs.n_corrs):
            bfgs.update(lbfgs.sk[i], lbfgs.yk[i])
        return bfgs.get_matrix()

    class _Callback:
        """ Iteration callback for scipy.optimize.minimize """
        
        def __init__(self, this, functions, timer, unflat):
            self.it = 0
            self.stamp = time.time()
            self.this = this
            self.functions = functions
            self.timer = timer
            self.unflat = unflat
            self.tail_overhead = 0
            self.tail_overhead_iter = 0

        def __call__(self, intermediate_result, arg2=None):
            
            if isinstance(intermediate_result, optimize.OptimizeResult):
                p = intermediate_result.x
            elif isinstance(intermediate_result, numpy.ndarray):
                p = intermediate_result
            else:
                raise TypeError(type(intermediate_result))

            self.it += 1
            now = time.time()
            duration = now - self.stamp
            
            worktime = sum(self.timer.partials.values())
            if worktime:
                overhead = duration - worktime
                assert overhead >= 0, (duration, worktime)
                if self.it == 1:
                    self.first_overhead = overhead
                else:
                    self.tail_overhead_iter += 1
                    self.tail_overhead += overhead

            if self.this._verbosity >= 3:
                calls = self.this._CountCalls.fmtcalls('partial', self.functions)

            if self.this._verbosity >= 4:
                tot = self.fmttime(duration)
                if self.timer.partials:
                    times = {
                        'gp&cov': self.timer.partials[0],
                        'dec': self.timer.partials[1],
                        'like': self.timer.partials[2],
                        'other': duration - sum(self.timer.partials.values()),
                    }
                    times = self.fmttimes(times)
                else:
                    times = 'n/d'
                self.this.log(f'\niteration {self.it}', 4)
                with self.this.loglevel:
                    self.this.log(f'total time: {tot}', 4)
                    self.this.log(f'partial: {times}', 4)
                    self.this.log(f'calls: {calls}', 4)

                if self.this._verbosity >= 5:
                    nicep = self.unflat(p)
                    nicep = self.this._copyasarrayorbufferdict(nicep)
                    with self.this.loglevel:
                        self.this.log(f'parameters = {nicep}', 5)
                    # TODO write a method to format the parameters nicely. => use
                    # gvar.tabulate? => nope, need actual gvars

            elif self.this._verbosity >= 3:
                times = self.fmttime(duration)
                self.this.log(f'iter {self.it}, time: {times}, calls: {calls}', 3)

            self.stamp = now
            self.timer.reset()

        pattern = re.compile(
            r'((\d+) days, )?(\d{1,2}):(\d\d):(\d\d(\.\d{6})?)')

        @classmethod
        def fmttime(cls, seconds):
            td = datetime.timedelta(seconds=seconds)
            m = cls.pattern.fullmatch(str(td))
            _, day, hour, minute, second, _ = m.groups()
            hour = int(hour)
            minute = int(minute)
            second = float(second)
            if day:
                return f'{day.lstrip("0")}d{hour:02d}h'
            elif hour:
                return f'{hour}h{minute:02d}m'
            elif minute:
                return f'{minute}m{second:02.0f}s'
            elif second >= 0.0995:
                return f'{second:#.2g}'.rstrip('.') + 's'
            elif second >= 0.0000995:
                return f'{second * 1e3:#.2g}'.rstrip('.') + 'ms'
            else:
                return f'{second * 1e6:.0f}μs'

        @classmethod
        def fmttimes(cls, times):
            """ times = dict label -> seconds """
            return ', '.join(f'{k} {cls.fmttime(v)}' for k, v in times.items())

        def estimate_firstcall_overhead(self):
            if self.tail_overhead_iter and hasattr(self, 'first_overhead'):
                typical_overhead = self.tail_overhead / self.tail_overhead_iter
                return self.first_overhead - typical_overhead

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
