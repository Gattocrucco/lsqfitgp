# lsqfitgp/bayestree.py
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

import warnings

import numpy
from jax import numpy as jnp
import gvar

from . import copula
from . import _kernels
from . import _fit
from . import _array
from . import _GP

class bart:
    
    def __init__(self,
        x_train,
        y_train,
        *,
        x_test=None,
        fitkw={},
        kernelkw={}):
        """
        GP version of BART.

        Evaluate a Gaussian process regression with a kernel which accurately
        approximates the infinite trees limit of BART. The hyperparameters are
        optimized to their marginal MAP.

        Parameters
        ----------
        x_train : (n, p) or dataframe
            Observed covariates.
        y_train : (n,)
            Observed outcomes.
        x_test : (n*, p) or dataframe, optional
            Covariates of outcomes to be imputed.
        fitkw : dict
            Additional arguments passed to `empbayes_fit`, overrides the
            defaults.
        kernelkw : dict
            Additional arguments passed to `BART`, overrides the
            defaults.
        
        Attributes
        ----------
        yhat_train_mean : (n,) array
            The posterior mean of the latent regression function at the observed
            covariates.
        yhat_train_var : (n,) array
            The posterior variance of the latent regression function at the
            observed covariates.
        yhat_test_mean : (n*,) array
            The posterior mean of the latent regression function at the
            covariates of imputed outcomes.
        yhat_test_var : (n*,) array
            The posterior variance of the latent regression function at the
            covariates of imputed outcomes.
        sigma : gvar
            The error term standard deviation marginal MAP.
        alpha : gvar
            The numerator of the tree spawn probability (named ``base`` in
            BayesTree and BART).
        beta : gvar
            The depth exponent of the tree spawn probability (named ``power`` in
            BayesTree and BART).
        meansdev : gvar
            The prior standard deviation of the latent regression function.
        fit : empbayes_fit
            The hyperparameters fit object.
        gp : GP
            The centered Gaussian process object constructed at the
            hyperparameters MAP. Its keys are 'trainmean' and 'testmean'.
        mu : scalar
            The prior mean.

        Notes
        -----
        The tree splitting grid is set using quantiles of the observed
        covariates. This corresponds to settings ``usequants=True``,
        ``numcut=inf`` in the R packages BayesTree and BART.

        See also
        --------
        BART
        
        """        
        # convert covariate matrices to StructuredArray
        x_train = self._to_structured(x_train)
        if x_test is None:
            x_test = _array.StructuredArray(numpy.empty_like(x_train, shape=0))
        else:
            x_test = self._to_structured(x_test)
        assert x_train.dtype == x_test.dtype
        assert x_train.ndim == x_test.ndim == 1
        assert x_train.size > len(x_train.dtype)
    
        # make sure data is a 1d array
        if hasattr(y_train, 'to_numpy'):
            y_train = y_train.to_numpy()
            y_train = y_train.squeeze() # for dataframes
        y_train = jnp.asarray(y_train)
        assert y_train.shape == x_train.shape
    
        # prior mean and variance
        ymin = jnp.min(y_train)
        ymax = jnp.max(y_train)
        mu_mu = (ymax + ymin) / 2
        k_sigma_mu = (ymax - ymin) / 2
        
        # splitting points and indices
        splits = _kernels.BART.splits_from_coord(x_train)
        def toindices(x):
            ix = _kernels.BART.indices_from_coord(x, splits)
            return _array.unstructured_to_structured(ix, names=x.dtype.names)
        i_train = toindices(x_train)
        i_test = toindices(x_test)

        # prior on hyperparams
        hyperprior = {
            '__bayestree__B(alpha)': copula.beta('__bayestree__B', 2, 1),
            '__bayestree__IG(beta)': copula.invgamma('__bayestree__IG', 1, 1),
            'log(k)': gvar.gvar(numpy.log(2), 2),
            'log(sigma2)': gvar.gvar(numpy.log(1), 2),
        }

        # GP factory
        def makegp(hp, *, i_train, i_test, splits):
            kw = dict(alpha=hp['alpha'], beta=hp['beta'], maxd=10, reset=[2,4,6,8], gamma=0.95)
            kw.update(kernelkw)
            kernel = _kernels.BART(splits=splits, indices=True, **kw)
            kernel *= (k_sigma_mu / hp['k']) ** 2
            
            gp = _GP.GP(kernel, checkpos=False, checksym=False, solver='chol')
            gp.addx(i_train, 'trainmean')
            gp.addcov(hp['sigma2'] * jnp.eye(i_train.size), 'noise')
            gp.addtransf({'trainmean': 1, 'noise': 1}, 'data')
            gp.addx(i_test, 'testmean')
            
            return gp

        # fit hyperparameters
        info = {'data': y_train - mu_mu}
        options = dict(
            verbosity=3,
            raises=False,
            jit=True,
            minkw=dict(method='l-bfgs-b', options=dict(maxls=4, maxiter=100)),
            mlkw=dict(epsrel=0),
            forward=True,
        )
        options.update(fitkw)
        gpkw=dict(i_train=i_train, i_test=i_test, splits=splits)
        fit = _fit.empbayes_fit(hyperprior, makegp, info, gpfactorykw=gpkw, **options)
        
        # extract hyperparameters from minimization result
        self.sigma = gvar.sqrt(fit.p['sigma2'])
        self.alpha = fit.p['alpha']
        self.beta = fit.p['beta']
        self.meansdev = k_sigma_mu / fit.p['k']

        # set attributes
        gp = makegp(fit.pmean, **gpkw)
        yhat_train_mean, yhat_train_cov = gp.predfromdata(info, 'trainmean', raw=True)
        self.yhat_train_mean = mu_mu + yhat_train_mean
        self.yhat_train_var = jnp.diag(yhat_train_cov)
        yhat_test_mean, yhat_test_cov = gp.predfromdata(info, 'testmean', raw=True)
        self.yhat_test_mean = mu_mu + yhat_test_mean
        self.yhat_test_var = jnp.diag(yhat_test_cov)
        self.fit = fit
        self.gp = gp
        self.mu = mu_mu

    @staticmethod
    def _to_structured(x):
        if hasattr(x, 'columns'):
            return _array.StructuredArray.from_dataframe(x)
        elif x.dtype.names is None:
            return _array.unstructured_to_structured(x)
        else:
            return _array.StructuredArray(x)
