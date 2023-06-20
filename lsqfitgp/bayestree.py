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

import numpy
from jax import numpy as jnp
import gvar

from . import copula
from . import _kernels
from . import _fit
from . import _array
from . import _GP
from . import _fastraniter

class bart:
    
    def __init__(self,
        x_train,
        y_train,
        *,
        weights=None,
        fitkw={},
        kernelkw={}):
        """
        GP version of BART.

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
        
        Attributes
        ----------
        mu : scalar
            The prior mean.
        sigma : gvar
            The error term standard deviation. If there are weights, the sdev
            for each unit is obtained dividing ``sigma`` by the weight.
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
        info : dict
            The dictionary to be passed to `GP.predfromdata` to represent
            ``y_train``.

        Methods
        -------
        gp :
            Create a GP object.

        Notes
        -----
        The tree splitting grid is set using quantiles of the observed
        covariates. This corresponds to settings ``usequants=True``,
        ``numcut=inf`` in the R packages BayesTree and BART.

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
        if weights is None:
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
        hyperprior = {
            '__bayestree__B(alpha)': copula.beta('__bayestree__B', 2, 1),
                # base of tree gen prob
            '__bayestree__IG(beta)': copula.invgamma('__bayestree__IG', 1, 1),
                # exponent of tree gen prob
            'log(k)': gvar.gvar(numpy.log(2), 2),
                # denominator of prior sdev
            'log(sigma2)': gvar.gvar(numpy.log(sigma2_priormean), 2),
                # i.i.d. error variance, scaled with weights
        }

        # GP factory
        def makegp(hp, *, i_train, weights, splits):
            kw = dict(
                alpha=hp['alpha'], beta=hp['beta'],
                maxd=10, reset=[2, 4, 6, 8], gamma=0.95,
            )
            kw.update(kernelkw)
            kernel = _kernels.BART(splits=splits, indices=True, **kw)
            kernel *= (k_sigma_mu / hp['k']) ** 2
            
            gp = _GP.GP(kernel, checkpos=False, checksym=False, solver='chol')
            gp.addx(i_train, 'trainmean')
            gp.addcov(jnp.diag(hp['sigma2'] / weights), 'trainnoise')
            gp.addtransf({'trainmean': 1, 'trainnoise': 1}, 'train')
            
            return gp

        # fit hyperparameters
        info = {'train': y_train - mu_mu}
        gpkw = dict(i_train=i_train, weights=weights, splits=splits)
        options = dict(
            verbosity=3,
            raises=False,
            jit=True,
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

        # set public attributes
        self.fit = fit
        self.mu = mu_mu.item()
        self.info = info

        # set private attributes
        self._gpkw = gpkw
        self._splits = splits
        self._ystd = y_train.std()

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

        Return
        ------
        gp : GP
            A centered Gaussian process object. To add the mean, use the ``mu``
            attribute of the `bart` object. The keys of the GP are '*mean',
            '*noise', '*' where the "*" stands either for 'train' or 'test'.
        """

        # determine hyperparameters
        if hp == 'map':
            hp = self.fit.pmean
        elif hp == 'sample':
            hp = _fastraniter.sample(self.fit.pmean, self.fit.pcov, rng=rng)

        # create GP object
        gp = self.fit.gpfactory(hp, **self._gpkw)

        # add test points
        if x_test is not None:

            # convert covariates to indices
            x_test = self._to_structured(x_test)
            i_test = self._toindices(x_test, self._splits)
            assert i_test.dtype == self._gpkw['i_train'].dtype

            # check weights
            if weights is not None:
                weights = jnp.asarray(weights)
                assert weights.shape == i_test.shape
            else:
                weights = jnp.ones(i_test.shape)

            # add test points
            gp.addx(i_test, 'testmean')
            gp.addcov(jnp.diag(hp['sigma2'] / weights), 'testnoise')
            gp.addtransf({'testmean': 1, 'testnoise': 1}, 'test')

        return gp

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
        return """BART fit:
alpha = {self.alpha} (0 -> intercept only, 1 -> any)
beta = {self.beta} (0 -> any, ∞ -> no interactions)
latent sdev = {self.meansdev} (large -> conservative extrapolation)
data total sdev = {self._ystd:.3g}
error sdev (weight scale) = {self.sigma}"""
