# lsqfitgp/bayestree/_bcf.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

# TODO adapt docstrings

class bcf:
    
    def __init__(self, *,
        y,
        z,
        x_control,
        x_moderate=None,
        pihat,
        include_pi='control',
        weights=None,
        fitkw={},
        kernelkw={},
        marginalize_mean=True,
    ):
        """
        Nonparametric Bayesian regression with a GP version of BCF.

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
        x_control = self._to_structured(x_control)
        if x_moderate is not None:
            x_moderate = self._to_structured(x_moderate)
    
        # convert outcomes, treatment, propensity score, weights to 1d arrays
        y = self._to_vector(y)
        z = self._to_vector(z)
        pihat = self._to_vector(pihat)
        if weights is not None:
            weights = self._to_vector(weights)

        # check shapes match
        assert y.shape == z.shape == x_control.shape == x_moderate.shape == weights.shape == pihat.shape

        # add propensity score to covariates
        if include_pi not in ('control', 'moderate', 'both'):
            raise KeyError(f'invalid value include_pi={include_pi!r}')
        if include_pi == 'control' or include_pi == 'both':
            x_control = recfunctions.append_fields(x_control, '__bayestree__pihat', pihat)
        if include_pi == 'moderate' or include_pi == 'both':
            x_moderate = recfunctions.append_fields(x_moderate, '__bayestree__pihat', pihat)
    
        # splitting points
        if x_moderate is None:
            x_all = x_control
        else:
            x_all = numpy.concatenate([x_control, x_moderate])
        splits = _kernels.BART.splits_from_coord(x_all)

        # indices w.r.t. splitting grid
        i_control = self._toindices(x_control, splits)
        if x_moderate is None:
            i_moderate = self._toindices(x_moderate, splits)
        else:
            i_moderate = None

        # data-dependent prior parameters
        ymin = jnp.min(y)
        ymax = jnp.max(y)
        ystd = jnp.std(y)
        mu_mu = (ymax + ymin) / 2
        k_sigma_mu = (ymax - ymin) / 2
        squares = (y - y.mean()) ** 2
        if weights is not None:
            squares *= weights
        sigma2_priormean = numpy.mean(squares)

        # define transformations for Gaussian copula
        copula.beta('__bayestree__B', 2, 1)
        copula.invgamma('__bayestree__IG', 1, 1)
        copula.uniform('__bayestree__U', 0, 1)
        copula.halfcauchy('__bayestree__HC', 1 / stats.halfcauchy.ppf(0.5))
        copula.halfnorm('__bayestree__HN', 1 / stats.halfnorm.ppf(0.5))

        # prior on hyperparams
        hyperprior = {
            '__bayestree__B(alpha)': gvar.gvar(numpy.zeros(2), numpy.ones(2)),
                # base of tree gen prob
            '__bayestree__IG(beta)': gvar.gvar(numpy.zeros(2), numpy.ones(2)),
                # exponent of tree gen prob
            '__bayestree__U(z0)': gvar.gvar(0, 1),
                # treatment coding
            '__bayestree__HC(scale_control)': gvar.gvar(0, 1),
            '__bayestree__HN(scale_moderate)': gvar.gvar(0, 1),
            'log(sigma2)': gvar.gvar(numpy.log(sigma2_priormean), 2),
                # i.i.d. error variance, scaled with weights
            'mean': gvar.gvar(mu_mu, k_sigma_mu),
                # mean of the GP
        }
        if marginalize_mean:
            hyperprior.pop('mean')

        # GP factory
        def makegp(hp, *, z, i_control, i_moderate, weights, splits, ystd, **_):
            
            kw = dict(maxd=10, reset=[2, 4, 6, 8], gamma=0.95, splits=splits)
            kw.update(kernelkw, indices=True)
                # TODO maybe I should pass kernelkw as argument

            kernel_control = _kernels.BART(alpha=hp['alpha'][0], beta=hp['beta'][0], dim='control', **kw)
            kernel_control *= (hp['scale_control'] * 2 * ystd) ** 2
            if 'mean' not in hp:
                kernel_control += k_sigma_mu ** 2 * _kernels.Constant()
            
            kernel_moderate = _kernels.BART(alpha=hp['alpha'][1], beta=hp['beta'][1], dim='moderate', **kw)
            kernel_moderate *= (hp['scale_moderate'] * ystd) ** 2

            gp = _GP.GP(checkpos=False, checksym=False, solver='chol')
            gp.addproc(kernel_control, 'control')
            gp.addproc(kernel_moderate, 'moderate')
            gp.addproclintransf(
                lambda mu, tau: lambda x: mu(x) + tau(x) * (x['z'] - hp['z0']),
                ['control', 'moderate'],
            )
            
            x = self._join_points(z, i_control, i_moderate)
            gp.addx(x, 'trainmean')
            errcov = self._error_cov(hp, weights, x)
            gp.addcov(errcov, 'trainnoise')
            gp.addtransf({'trainmean': 1, 'trainnoise': 1}, 'train')
            
            return gp

        # data factory
        def info(hp, *, y, mu_mu, **_):
            return {'train': y - hp.get('mean', mu_mu)}

        # fit hyperparameters
        gpkw = dict(
            z=z,
            i_control=i_control,
            i_moderate=i_moderate,
            weights=weights,
            splits=splits,
            ystd=ystd,
            mu_mu=mu_mu,
            y=y,
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
        self.z0 = fit.p['z0']
        self.sigma = gvar.sqrt(fit.p['sigma2'])
        self.alpha = fit.p['alpha']
        self.beta = fit.p['beta']
        self.scale = numpy.array([
            fit.p['scale_control'] * 2 * ystd,
            fit.p['scale_moderate'] * ystd,
        ])
        self.mean = fit.p.get('mean', mu_mu)

        # save fit object
        self.fit = fit

    def _join_points(self, z, i_control, i_moderate):
        """ join covariates into a single StructuredArray """
        return _array.StructuredArray.from_dict(dict(
            z=z,
            control=i_control,
            moderate=i_control if i_moderate is None else i_moderate,
        ))

    def _error_cov(self, hp, weights, x):
        """ fill error covariance matrix """
        if weights is None:
            error_var = jnp.broadcast_to(hp['sigma2'], len(x))
        else:
            error_var = hp['sigma2'] / weights
        return jnp.diag(error_var)

    def _gethp(self, hp, rng):
        if not isinstance(hp, str):
            return hp
        elif hp == 'map':
            return self.fit.pmean
        elif hp == 'sample':
            return _fastraniter.sample(self.fit.pmean, self.fit.pcov, rng=rng)
        else:
            raise KeyError(hp)

    def gp(self, *, hp='map', z=None, x_control=None, x_moderate=None, weights=None, rng=None):
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
        return self._gp(hp, z, x_control, x_moderate, weights, self.fit.gpfactorykw)

    def _gp(self, hp, z, x_control, x_moderate, weights, gpfactorykw):

        # create GP object
        gp = self.fit.gpfactory(hp, **gpfactorykw)

        # add test points
        if z is not None:

            # check presence/absence of arguments is coherent
            self._check_coherent_covariates(z, x_control, x_moderate)

            # convert covariates to indices
            x_control = self._to_structured(x_control)
            i_control = self._toindices(x_control, gpfactorykw['splits'])
            assert i_control.dtype == gpfactorykw['i_control'].dtype
            if test_moderate:
                x_moderate = self._to_structured(x_moderate)
                i_moderate = self._toindices(x_moderate, gpfactorykw['splits'])
                assert i_moderate.dtype == gpfactorykw['i_moderate'].dtype
            else:
                i_moderate = None

            # check weights
            if weights is not None:
                weights = jnp.asarray(weights)
                assert weights.shape == i_control.shape

            # add test points
            x = self._join_points(z, i_control, i_moderate)
            gp.addx(x, 'testmean')
            errcov = self._error_cov(hp, weights, x)
            gp.addcov(errcov, 'testnoise')
            gp.addtransf({'testmean': 1, 'testnoise': 1}, 'test')

        return gp

    def _check_coherent_covariates(self, z, x_control, x_moderate):
        if z is None:
            assert x_control is None and x_moderate is None
        else:
            assert x_control is not None
            train_moderate = self.fit.gpfactorykw['i_moderate']
            if x_moderate is None:
                assert train_moderate is None
            else:
                assert train_moderate is not None

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

    def pred(self, *, hp='map', error=False, format='matrices', z=None,
        x_control=None, x_moderate=None, weights=None, rng=None):
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
            separately. If 'gvar', return an array of `GVar`s.
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
        
        # get hyperparameters
        hp = self._gethp(hp, rng)
        
        # check and process covariates
        self._check_coherent_covariates(z, x_control, x_moderate)
        if z is not None:
            x_control = self._to_structured(x_control)
            if x_moderate is not None:
                x_moderate = self._to_structured(x_moderate)
        
        # GP regression
        mean, cov = self._pred(hp, z, x_control, x_moderate, weights, self.fit.gpfactorykw, bool(error))

        # pack output
        if format == 'gvar':
            return gvar.gvar(mean, cov, fast=True)
        elif format == 'matrices':
            return mean, cov
        else:
            raise KeyError(format)

    @functools.cached_property
    def _pred(self):
        
        @functools.partial(jax.jit, static_argnums=(6,))
        def _pred(hp, z, x_control, x_moderate, weights, gpfactorykw, error):
            gp = self._gp(hp, z, x_control, x_moderate, weights, gpfactorykw)
            data = self.fit.data(hp, **gpfactorykw)
            if z is None:
                label = 'train'
            else:
                label = 'test'
            if not error:
                label += 'mean'
            outmean, outcov = gp.predfromdata(data, label, raw=True)
            return outmean + hp.get('mean', gpfactorykw['mu_mu']), outcov

        # TODO make everything pure and jit this per class instead of per
        # instance

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
        assert x.size > len(x.dtype)
        def check_numerical(path, dtype):
            if not numpy.issubdtype(dtype, numpy.number):
                raise TypeError(f'covariate `{path}` is not numerical')
        cls._walk_dtype(x.dtype, check_numerical)

        return x

    @classmethod
    def _to_vector(cls, x):
        if hasattr(x, 'to_numpy'):
            x = x.to_numpy()
            x = x.squeeze() # for dataframes
        return jnp.asarray(x)

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
        out = f"""BCF fit:
[control param, moderate param]
alpha = {self.alpha} (0 -> intercept only, 1 -> any)
beta = {self.beta} (0 -> any, ∞ -> no interactions)
z0 = {self.z0} (control model applies at z = z0)
mean = {self.mean}
latent sdev = {self.scale} (large -> conservative extrapolation)
data total sdev = {self.fit.gpfactorykw['ystd']:.3g}"""

        weights = self.fit.gpfactorykw['weights']
        if weights is None:
            out += f"""
error sdev = {self.sigma}"""
        else:
            weights = numpy.array(weights) # to avoid jax taking over the ops
            avgsigma = numpy.sqrt(numpy.mean(self.sigma ** 2 / weights))
            out += f"""
error sdev (avg weighted) = {avgsigma}
error sdev (unweighted) = {self.sigma}"""

        return out
