# lsqfitgp/bayestree/_bart.py
#
# Copyright (c) 2023, 2024, Giacomo Petrillo
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

import functools

import numpy
from jax import numpy as jnp
import jax
import gvar

from .. import copula
from .. import _kernels
from .. import _fit
from .. import _array
from .. import _GP
from .. import _fastraniter

# TODO I added a lot of functionality to bcf. The easiest way to port it over is
# adding the option in bcf to drop the second bart model and its associated
# hypers, and then write bart as a simple convenience wrapper-subclass over bcf.
# (also the option include_pi='none'.)

class bart:
    
    def __init__(self,
        x_train,
        y_train,
        *,
        weights=None,
        fitkw={},
        kernelkw={},
        marginalize_mean=True,
    ):
        """
        Nonparametric Bayesian regression with a GP version of BART.

        Evaluate a Gaussian process regression with a kernel which accurately
        approximates the infinite trees limit of BART. The hyperparameters are
        optimized to their marginal MAP.

        Parameters
        ----------
        x_train : (n, p) array or dataframe
            Observed covariates.
        y_train : (n,) array
            Observed outcomes.
        weights : (n,) array
            Weights used to rescale the error variance (as 1 / weight).
        fitkw : dict
            Additional arguments passed to `~lsqfitgp.empbayes_fit`, overrides
            the defaults.
        kernelkw : dict
            Additional arguments passed to `~lsqfitgp.BART`, overrides the
            defaults.
        marginalize_mean : bool
            If True (default), marginalize the intercept of the model.
        
        Notes
        -----
        The regression model is:

        .. math::
            y_i &= \\mu + \\lambda f(\\mathbf x_i) + \\varepsilon_i, \\\\
            \\varepsilon_i &\\overset{\\mathrm{i.i.d.}}{\\sim}
                N(0, \\sigma^2 / w_i), \\\\
            \\mu &\\sim N(
                (\\max(\\mathbf y) + \\min(\\mathbf y)) / 2,
                (\\max(\\mathbf y) - \\min(\\mathbf y))^2 / 4
            ), \\\\
            \\log \\sigma^2 &\\sim N(
                \\log(\\overline{w(y - \\bar y)^2}),
                4
            ), \\\\
            \\log \\lambda &\\sim N(
                \\log ((\\max(\\mathbf y) - \\min(\\mathbf y)) / 4),
                4
            ), \\\\
            f &\\sim \\mathrm{GP}(
                0,
                \\mathrm{BART}(\\alpha,\\beta)
            ), \\\\
            \\alpha &\\sim \\mathrm{B}(2, 1), \\\\
            \\beta &\\sim \\mathrm{IG}(1, 1).

        To make the inference, :math:`(f, \\boldsymbol\\varepsilon, \\mu)` are
        marginalized analytically, and the marginal posterior mode of
        :math:`(\\sigma, \\lambda, \\alpha, \\beta)` is found by numerical
        minimization, after transforming them to express their prior as a
        Gaussian copula. Their marginal posterior covariance matrix is estimated
        with an approximation of the hessian inverse. See
        `~lsqfitgp.empbayes_fit` and use the parameter ``fitkw`` to customize
        this procedure.

        The tree splitting grid of the BART kernel is set using quantiles of the
        observed covariates. This corresponds to settings ``usequants=True``,
        ``numcut=inf`` in the R packages BayesTree and BART. Use the
        ``kernelkw`` parameter to customize the grid.

        Attributes
        ----------
        mean : gvar
            The prior mean :math:`\\mu`.
        sigma : float or gvar
            The error term standard deviation :math:`\\sigma`. If there are
            weights, the sdev for each unit is obtained dividing ``sigma`` by
            sqrt(weight).
        alpha : gvar
            The numerator of the tree spawn probability :math:`\\alpha` (named
            ``base`` in BayesTree and BART).
        beta : gvar
            The depth exponent of the tree spawn probability :math:`\\beta`
            (named ``power`` in BayesTree and BART).
        meansdev : gvar
            The prior standard deviation :math:`\\lambda` of the latent
            regression function.
        fit : empbayes_fit
            The hyperparameters fit object.

        Methods
        -------
        gp :
            Create a GP object.
        data :
            Creates the dictionary to be passed to `GP.pred` to represent
            ``y_train``.
        pred :
            Evaluate the regression function at given locations.

        See also
        --------
        lsqfitgp.BART
        
        """

        # convert covariates to StructuredArray
        x_train = self._to_structured(x_train)
    
        # convert outcomes to 1d array
        if hasattr(y_train, 'to_numpy'):
            y_train = y_train.to_numpy()
            y_train = y_train.squeeze() # for dataframes
        y_train = jnp.asarray(y_train)
        assert y_train.shape == x_train.shape

        # check weights
        self._no_weights = weights is None
        if self._no_weights:
            weights = jnp.ones_like(y_train)
        assert weights.shape == y_train.shape
    
        # prior mean and variance
        ymin = jnp.min(y_train)
        ymax = jnp.max(y_train)
        mu_mu = (ymax + ymin) / 2
        k_sigma_mu = (ymax - ymin) / 2
        
        # splitting points and indices
        splits = _kernels.BART.splits_from_coord(x_train)
        i_train = self._toindices(x_train, splits)

        # prior on hyperparams
        sigma2_priormean = numpy.mean((y_train - y_train.mean()) ** 2 * weights)
        hyperprior = copula.makedict({
            'alpha': copula.beta(2, 1), # base of tree gen prob
            'beta': copula.invgamma(1, 1), # exponent of tree gen prob
            'log(k)': gvar.gvar(numpy.log(2), 2), # denominator of prior sdev
            'log(sigma2)': gvar.gvar(numpy.log(sigma2_priormean), 2),
                # i.i.d. error variance, scaled with weights
            'mean': gvar.gvar(mu_mu, k_sigma_mu), # mean of the GP
        })
        if marginalize_mean:
            hyperprior.pop('mean')

        # GP factory
        def makegp(hp, *, i_train, weights, splits, **_):
            kw = dict(
                alpha=hp['alpha'], beta=hp['beta'],
                maxd=10, reset=[2, 4, 6, 8],
            )
            kw.update(kernelkw)
            kernel = _kernels.BART(splits=splits, indices=True, **kw)
            kernel *= (k_sigma_mu / hp['k']) ** 2
            
            gp = (_GP
                .GP(kernel, checkpos=False, checksym=False, solver='chol')
                .addx(i_train, 'trainmean')
                .addcov(jnp.diag(hp['sigma2'] / weights), 'trainnoise')
            )
            pieces = {'trainmean': 1, 'trainnoise': 1}
            if 'mean' not in hp:
                gp = gp.addcov(k_sigma_mu ** 2, 'mean')
                pieces.update({'mean': 1})
            return gp.addtransf(pieces, 'train')
            
        # data factory
        def info(hp, *, mu_mu, **_):
            return {'train': y_train - hp.get('mean', mu_mu)}

        # fit hyperparameters
        gpkw = dict(
            i_train=i_train,
            weights=weights,
            splits=splits,
            mu_mu=mu_mu,
        )
        options = dict(
            verbosity=3,
            raises=False,
            minkw=dict(method='l-bfgs-b', options=dict(maxls=4, maxiter=100)),
            mlkw=dict(epsrel=0),
            forward=True,
            gpfactorykw=gpkw,
        )
        options.update(fitkw)
        fit = _fit.empbayes_fit(hyperprior, makegp, info, **options)
        
        # extract hyperparameters from minimization result
        self.sigma = gvar.sqrt(fit.p['sigma2'])
        self.alpha = fit.p['alpha']
        self.beta = fit.p['beta']
        self.meansdev = k_sigma_mu / fit.p['k']
        self.mean = fit.p.get('mean', mu_mu)

        # set public attributes
        self.fit = fit

        # set private attributes
        self._ystd = y_train.std()

    def _gethp(self, hp, rng):
        if not isinstance(hp, str):
            return hp
        elif hp == 'map':
            return self.fit.pmean
        elif hp == 'sample':
            return _fastraniter.sample(self.fit.pmean, self.fit.pcov, rng=rng)
        else:
            raise KeyError(hp)

    def gp(self, *, hp='map', x_test=None, weights=None, rng=None):
        """
        Create a Gaussian process with the fitted hyperparameters.

        Parameters
        ----------
        hp : str or dict
            The hyperparameters to use. If ``'map'``, use the marginal maximum a
            posteriori. If ``'sample'``, sample hyperparameters from the
            posterior. If a dict, use the given hyperparameters.
        x_test : array or dataframe, optional
            Additional covariates for "test points".
        weights : array, optional
            Weights for the error variance on the test points.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'``.

        Returns
        -------
        gp : GP
            A centered Gaussian process object. To add the mean, use the
            ``mean`` attribute of the `bart` object. The keys of the GP are
            'Xmean', 'Xnoise', and 'X', where the "X" stands either for 'train'
            or 'test', and X = Xmean + Xnoise.
        """

        hp = self._gethp(hp, rng)
        return self._gp(hp, x_test, weights, self.fit.gpfactorykw)

    def _gp(self, hp, x_test, weights, gpfactorykw):

        # create GP object
        gp = self.fit.gpfactory(hp, **gpfactorykw)

        # add test points
        if x_test is not None:

            # convert covariates to indices
            x_test = self._to_structured(x_test)
            i_test = self._toindices(x_test, gpfactorykw['splits'])
            assert i_test.dtype == gpfactorykw['i_train'].dtype

            # check weights
            if weights is not None:
                weights = jnp.asarray(weights)
                assert weights.shape == i_test.shape
            else:
                weights = jnp.ones(i_test.shape)

            # add test points
            gp = (gp
                .addx(i_test, 'testmean')
                .addcov(jnp.diag(hp['sigma2'] / weights), 'testnoise')
            )
            pieces = {'testmean': 1, 'testnoise': 1}
            if 'mean' not in hp:
                pieces.update({'mean': 1})
            gp = gp.addtransf(pieces, 'test')

        return gp

    def data(self, *, hp='map', rng=None):
        """
        Get the data to be passed to `GP.pred` on a GP object returned by `gp`.

        Parameters
        ----------
        hp : str or dict
            The hyperparameters to use. If ``'map'``, use the marginal maximum a
            posteriori. If ``'sample'``, sample hyperparameters from the
            posterior. If a dict, use the given hyperparameters.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'``.

        Returns
        -------
        data : dict
            A dictionary representing ``y_train`` in the format required by the
            `GP.pred` method.
        """

        hp = self._gethp(hp, rng)
        return self.fit.data(hp, **self.fit.gpfactorykw)

    def pred(self, *, hp='map', error=False, format='matrices', x_test=None,
        weights=None, rng=None):
        """
        Predict the outcome at given locations.

        Parameters
        ----------
        hp : str or dict
            The hyperparameters to use. If ``'map'``, use the marginal maximum a
            posteriori. If ``'sample'``, sample hyperparameters from the
            posterior. If a dict, use the given hyperparameters.
        error : bool
            If ``False`` (default), make a prediction for the latent mean. If
            ``True``, add the error term.     
        format : {'matrices', 'gvar'}
            If 'matrices' (default), return the mean and covariance matrix
            separately. If 'gvar', return an array of gvars.
        x_test : array or dataframe, optional
            Covariates for the locations where the prediction is computed. If
            not specified, predict at the data covariates.
        weights : array, optional
            Weights for the error variance on the test points.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'``.

        Returns
        -------
        If ``format`` is 'matrices' (default):

        mean, cov : arrays
            The mean and covariance matrix of the Normal posterior distribution
            over the regression function at the specified locations.

        If ``format`` is 'gvar':

        out : array of `GVar`
            The same distribution represented as an array of `GVar` objects.
        """

        # TODO it is a bit confusing that if x_test=None and error=True, the
        # prediction returns y_train exactly, instead of hypothetical new
        # observations at the same covariates.
        
        hp = self._gethp(hp, rng)
        if x_test is not None:
            x_test = self._to_structured(x_test)
        mean, cov = self._pred(hp, x_test, weights, self.fit.gpfactorykw, bool(error))

        if format == 'gvar':
            return gvar.gvar(mean, cov, fast=True)
        elif format == 'matrices':
            return mean, cov
        else:
            raise KeyError(format)

    @functools.cached_property
    def _pred(self):
        
        @functools.partial(jax.jit, static_argnums=(4,))
        def _pred(hp, x_test, weights, gpfactorykw, error):
            gp = self._gp(hp, x_test, weights, gpfactorykw)
            data = self.fit.data(hp, **gpfactorykw)
            if x_test is None:
                label = 'train'
            else:
                label = 'test'
            if not error:
                label += 'mean'
            outmean, outcov = gp.predfromdata(data, label, raw=True)
            return outmean + hp.get('mean', gpfactorykw['mu_mu']), outcov

        return _pred

    @classmethod
    def _to_structured(cls, x):

        # convert to StructuredArray
        if hasattr(x, 'columns'):
            x = _array.StructuredArray.from_dataframe(x)
        elif x.dtype.names is None:
            x = _array.unstructured_to_structured(x)
        else:
            x = _array.StructuredArray(x)

        # check
        assert x.ndim == 1
        def check_numerical(path, dtype):
            if not numpy.issubdtype(dtype, numpy.number):
                raise TypeError(f'covariate `{path}` is not numerical')
        cls._walk_dtype(x.dtype, check_numerical)

        return x

    @classmethod
    def _walk_dtype(cls, dtype, task, path=None):
        if dtype.names is None:
            task(path, dtype)
        else:
            for name in dtype.names:
                subpath = name if path is None else path + ':' + name
                cls._walk_dtype(dtype[name], task, subpath)

    @staticmethod
    def _toindices(x, splits):
        ix = _kernels.BART.indices_from_coord(x, splits)
        return _array.unstructured_to_structured(ix, names=x.dtype.names)

    def __repr__(self):
        out = f"""BART fit:
alpha = {self.alpha} (0 -> intercept only, 1 -> any)
beta = {self.beta} (0 -> any, ∞ -> no interactions)
mean = {self.mean}
latent sdev = {self.meansdev} (large -> conservative extrapolation)
data total sdev = {self._ystd:.3g}"""

        if self._no_weights:
            out += f"""
error sdev = {self.sigma}"""
        else:
            weights = numpy.array(self.fit.gpfactorykw['weights'])
            avgsigma = numpy.sqrt(numpy.mean(self.sigma ** 2 / weights))
            out += f"""
error sdev (avg weighted) = {avgsigma}
error sdev (unweighted) = {self.sigma}"""

        return out
