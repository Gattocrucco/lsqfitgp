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
from scipy import stats
from jax import numpy as jnp
import jax
import gvar

from .. import copula
from .. import _kernels
from .. import _fit
from .. import _array
from .. import _GP
from .. import _fastraniter

# TODO add methods or options to do causal inference stuff, e.g., impute missing
# outcomes, or ate, att, cate, catt, sate, satt. Remember that the effect may
# also depend on aux. See bartCause, possibly copy its naming.

def _recursive_cast(dtype, default, mapping):
    if dtype in mapping:
        return mapping[dtype]
    elif dtype.names is not None:
        return numpy.dtype([
            (name, _recursive_cast(dtype[name], default, mapping))
            for name in dtype.names
        ])
    elif dtype.subdtype is not None:
        # note: has names => does not have subdtype
        return numpy.dtype((_recursive_cast(dtype.base, default, mapping), dtype.shape))
    elif default is None:
        return dtype
    else:
        return default

def cast(dtype, default, mapping={}):
    """
    Recursively cast a numpy data type.

    Parameters
    ----------
    dtype : dtype
        The data type to cast.
    default : dtype or None
        The leaf fields of `dtype` are casted to `default`, which can be
        structured, unless they appear in `mapping`. If None, dtypes not in
        `mapping` are left unchanged.
    mapping : dict
        A dictionary from dtypes to dtypes, indicating specific casting rules.
        The dtypes can be structured, a match of a structured dtype takes
        precedence over matches in its leaves, and the converted dtype is not
        further searched for matches.

    Returns
    -------
    casted_dtype : dtype
        The casted version of `dtype`. May not have the same structure if
        `mapping` contains structured dtypes.
    """
    mapping = {numpy.dtype(k): numpy.dtype(v) for k, v in mapping.items()}
    default = None if default is None else numpy.dtype(default)
    return _recursive_cast(numpy.dtype(dtype), default, mapping)

class bcf:
    
    def __init__(self, *,
        y,
        z,
        x_mu,
        x_tau=None,
        pihat,
        include_pi='mu',
        weights=None,
        fitkw={},
        kernelkw_mu={},
        kernelkw_tau={},
        marginalize_mean=True,
        gpaux=None,
        x_aux=None,
        otherhp={},
    ):
        r"""
        Nonparametric Bayesian regression with a GP version of BCF.

        BCF (Bayesian Causal Forests) is a regression method for observational
        causal inference studies introduced in [1]_ based on a pair of BART
        models.

        This class evaluates a Gaussian process regression with a kernel which
        accurately approximates BCF in the infinite trees limit of each BART
        model. The hyperparameters are optimized to their marginal MAP.

        The model is (loosely, see notes below) :math:`y = \mu(x) + z\tau(x)`,
        so :math:`\tau(x)` is the expected causal effect of :math:`z` on
        :math:`y` at location :math:`x`.

        Parameters
        ----------
        y : (n,) array, series or dataframe
            Outcome.
        z : (n,) array, series or dataframe
            Binary treatment status: 0 control group, 1 treatment group.
        x_mu : (n, p) array, series or dataframe
            Covariates for the :math:`\mu` model.
        x_tau : (n, q) array, series or dataframe, optional
            Covariates for the :math:`\tau` model. If not specified, use `x_mu`.
        pihat : (n,) array, series or dataframe
            Estimated propensity score, i.e., P(Z=1|X).
        include_pi : {'mu', 'tau', 'both'}, optional
            Whether to include the propensity score in the :math:`\mu` model,
            the :math:`\tau` model, or both. Default is ``'mu'``.
        weights : (n,) array, series or dataframe
            Weights used to rescale the error variance (as 1 / weight).
        fitkw : dict
            Additional arguments passed to `~lsqfitgp.empbayes_fit`, overrides
            the defaults.
        kernelkw_mu, kernelkw_tau : dict
            Additional arguments passed to `~lsqfitgp.BART` for each model,
            overrides the defaults.
        marginalize_mean : bool
            If True (default), marginalize the intercept of the model.
        gpaux : callable, optional
            If specified, this function is called with a pair ``(hp, gp)``,
            where ``hp`` is a dictionary of hyperparameters, and ``gp`` is a
            `~lsqfitgp.GP` object under construction, and is expected to define
            a new process named ``'aux'`` with `~lsqfitgp.GP.defproc` or
            similar. The process is added to the regression model. The input to
            the process is a structured array with fields:

            'train' : bool
                Indicates wether the data is training set (the one passed on
                initialization) or test set (the one passed to `pred` or `gp`).
            'i' : int
                Index of the flattened array.
            'z' : int
                Treatment status.
            'mu', 'tau' : structured
                The values in `x_mu` and `x_tau`, converted to indices according
                to the BART grids. Where `pihat` has been added, there are two
                subfields: ``'x'`` which contains the covariates, and
                ``'pihat'``, the latter expressed in indices as well.
            'pihat' : float
                The `pihat` argument. Contrary to the subfield included under
                ``'mu'`` and/or ``'tau'``, this field contains the original
                values.
            'aux' : structured
                The values in `x_aux`, if specified.
        
        x_aux : (n, k) array, series or dataframe, optional
            Additional covariates for the ``'aux'`` process.
        otherhp : dictionary of gvar
            A dictionary with the prior of additional hyperpameters, intended to
            be used by ``gpaux``.
        
        Notes
        -----
        The regression model is:

        .. math::
            y_i &= m + {} \\
                &\phantom{{}={}} +
                    \lambda_\mu\,\mathrm{std}(\mathbf y)
                    \mu(\mathbf x^\mu_i, \hat\pi_i?) + {} \\
                &\phantom{{}={}} +
                    \lambda_\tau\,\mathrm{std}(\mathbf y)
                    \tau(\mathbf x^\tau_i, \hat\pi_i?) (z_i - z_0) + {} \\
                &\phantom{{}={}} +
                    \mathrm{aux}(i, z_i, \mathbf x^\mu_i, \mathbf x^\tau_i,
                        \hat\pi_i, \mathbf x^\text{aux}_i) + {} \\
                &\phantom{{}={}} +
                    \varepsilon_i, \\
            \varepsilon_i &\sim
                N(0, \sigma^2 / w_i), \\
            m &\sim N(
                (\max(\mathbf y) + \min(\mathbf y)) / 2,
                (\max(\mathbf y) - \min(\mathbf y))^2 / 4
            ), \\
            \log \sigma^2 &\sim N(
                \log(\overline{w(y - \bar y)^2}),
                4
            ), \\
            \lambda_\mu
                &\sim \mathrm{HalfCauchy}(2), \\
            \lambda_\tau
                &\sim \mathrm{HalfNormal}(1.48), \\
            \mu &\sim \mathrm{GP}(0,
                \mathrm{BART}(\alpha_\mu, \beta_\mu) ), \\
            \tau &\sim \mathrm{GP}(0,
                \mathrm{BART}(\alpha_\tau, \beta_\tau) ), \\
            \mathrm{aux} & \sim \mathrm{GP}(0, \text{<user defined>}), \\
            \alpha_\mu, \alpha_\tau &\sim \mathrm{B}(2, 1), \\
            \beta_\mu, \beta_\tau &\sim \mathrm{IG}(1, 1), \\
            z_0 &\sim U(0, 1),

        To make the inference, :math:`(\mu, \tau, \boldsymbol\varepsilon, m,
        \mathrm{aux})` are marginalized analytically, and the marginal posterior
        mode of
        :math:`(\sigma, \lambda_*, \alpha_*, \beta_*, z_0)` is found by
        numerical minimization, after transforming them to express their prior
        as a Gaussian copula. Their marginal posterior covariance matrix is
        estimated with an approximation of the hessian inverse. See
        `~lsqfitgp.empbayes_fit` and use the parameter ``fitkw`` to customize
        this procedure.

        The tree splitting grid of the BART kernel is set using quantiles of the
        observed covariates. This corresponds to settings ``usequants=True``,
        ``numcut=inf`` in the R packages BayesTree and BART. Use the parameters
        `kernelkw_mu` and `kernelkw_tau` to customize the grids.

        The difference between the regression model evaluated at :math:`Z=1` vs.
        :math:`Z=0` can be interpreted as the causal effect :math:`Z \rightarrow
        Y` if the unconfoundedness assumption is made:

        .. math::
            \{Y(Z=0), Y(Z=1)\} \perp\!\!\!\perp Z \mid X.

        In practical terms, this holds when:

            1) :math:`X` are pre-treatment variables, i.e., they represent
               quantities causally upstream of :math:`Z`.

            2) :math:`X` are sufficient to adjust for all common causes of
               :math:`Z` and :math:`Y`, such that the only remaining difference
               is the causal effect and not just a correlation.

        Here :math:`X` consists in `x_tau`, `x_mu` and `x_aux`. However these
        arrays may also be used to pass "technical" values used to set up the
        model, that do not satisfy the uncounfoundedness assumption, if you know
        what you are doing.

        Attributes
        ----------
        m : float or gvar
            The prior mean :math:`m`.
        sigma : gvar
            The error term standard deviation :math:`\sigma`. If there are
            weights, the sdev for each unit is obtained dividing ``sigma`` by
            sqrt(weight).
        alpha_mu, alpha_tau : gvar
            The numerator of the tree spawn probability :math:`\alpha_*` (named
            ``base`` in R bcf).
        beta_mu, beta_tau : gvar
            The depth exponent of the tree spawn probability :math:`\beta_*`
            (named ``power`` in R bcf).
        sdev_mu, sdev_tau : gvar
            The prior standard deviation :math:`\lambda_* \mathrm{std}(\mathbf
            y)`.
        z_0 : gvar
            The treatment coding parameter.
        fit : empbayes_fit
            The hyperparameters fit object.

        Methods
        -------
        gp :
            Create a GP object.
        data :
            Creates the dictionary to be passed to `GP.pred` to represent ``y``.
        pred :
            Evaluate the regression function at given locations.

        See also
        --------
        lsqfitgp.BART
        
        References
        ----------
        .. [1] P. Richard Hahn, Jared S. Murray, Carlos M. Carvalho "Bayesian
            Regression Tree Models for Causal Inference: Regularization,
            Confounding, and Heterogeneous Effects (with Discussion)," Bayesian
            Analysis 15(3), 965-1056, September 2020,
            https://doi.org/10.1214/19-BA1195
        """

        # convert covariates to StructuredArray
        x_mu = self._to_structured(x_mu)
        if x_tau is not None:
            x_tau = self._to_structured(x_tau)
            assert x_tau.shape == x_mu.shape
        if x_aux is not None:
            x_aux = self._to_structured(x_aux)
            assert x_aux.shape == x_mu.shape
    
        # convert outcomes, treatment, propensity score, weights to 1d arrays
        y = self._to_vector(y)
        z = self._to_vector(z)
        pihat = self._to_vector(pihat)
        assert y.shape == z.shape == pihat.shape == x_mu.shape
        if weights is not None:
            weights = self._to_vector(weights)
            assert weights.shape == x_mu.shape

        # check include_pi
        if include_pi not in ('mu', 'tau', 'both'):
            raise KeyError(f'invalid value include_pi={include_pi!r}')
        self._include_pi = include_pi

        # add pihat to covariates
        x_mu, x_tau = self._append_pihat(x_mu, x_tau, pihat)
    
        # grid and indices
        splits_mu = _kernels.BART.splits_from_coord(x_mu)
        i_mu = self._toindices(x_mu, splits_mu)
        if x_tau is None:
            splits_tau = splits_mu
            i_tau = None
        else:
            splits_tau = _kernels.BART.splits_from_coord(x_tau)
            i_tau = self._toindices(x_tau, splits_tau)

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

        # prior on hyperparams
        hyperprior = copula.makedict({
            'm': gvar.gvar(mu_mu, k_sigma_mu),
            'log(sigma^2)': gvar.gvar(numpy.log(sigma2_priormean), 2),
            'lambda_mu': copula.halfcauchy(2),
            'lambda_tau': copula.halfnorm(1.48),
            'alpha_mu': copula.beta(2, 1),
            'alpha_tau': copula.beta(2, 1),
            'beta_mu': copula.invgamma(1, 1),
            'beta_tau': copula.invgamma(1, 1),
            'z_0': copula.uniform(0, 1),
        })
        if marginalize_mean:
            hyperprior.pop('m')

        # add user hyperparams
        otherhp = gvar.BufferDict(otherhp)
        for key in otherhp.all_keys():
            if hyperprior.has_dictkey(key):
                raise ValueError(f'otherhp[{key!r}] overrides another hyperparameter')
        hyperprior.update(otherhp)

        # GP factory
        def makegp(hp, *, z, i_mu, i_tau, pihat, x_aux, weights,
            splits_mu, splits_tau, ystd, k_sigma_mu, **_):
            
            # TODO maybe I should pass kernelkw_* as arguments, but they may not
            # be jittable. I need jitkw in empbayes_fit for that.

            kw_overridable = dict(
                maxd=10,
                reset=[2, 4, 6, 8],
                gamma=0.95,
                intercept=False,
            )
            kw_not_overridable = dict(indices=True)

            gp = _GP.GP(checkpos=False, checksym=False, solver='chol')

            kernelkw = dict(mu=kernelkw_mu, tau=kernelkw_tau)
                # eval() won't work because they are not captured elsewhere
            
            for name in 'mu', 'tau':
                kw = dict(
                    alpha=hp[f'alpha_{name}'],
                    beta=hp[f'beta_{name}'],
                    dim=name,
                    splits=eval(f'splits_{name}'),
                    **kw_overridable,
                )
                kw.update(kernelkw[name])
                kernel = _kernels.BART(**kw, **kw_not_overridable)
                kernel *= (hp[f'lambda_{name}'] * ystd) ** 2
                
                gp.defproc(name, kernel)

            if 'm' in hp:
                kernel_mean = 0 * _kernels.Constant()
            else:
                kernel_mean = k_sigma_mu ** 2 * _kernels.Constant()
            gp.defproc('m', kernel_mean)

            if gpaux is None:
                gp.defproc('aux', 0 * _kernels.Constant())
            else:
                gpaux(hp, gp)

            gp.defproclintransf(
                gp.DefaultProcess,
                lambda m, mu, tau, aux: lambda x:
                m(x) + mu(x) + tau(x) * (x['z'] - hp['z_0']) + aux(x),
                ['m', 'mu', 'tau', 'aux'],
            )
            
            x = self._join_points(True, z, i_mu, i_tau, pihat, x_aux)
            gp.addx(x, 'trainmean')
            errcov = self._error_cov(hp, weights, x)
            gp.addcov(errcov, 'trainnoise')
            gp.addtransf({'trainmean': 1, 'trainnoise': 1}, 'train')
            
            return gp

        # data factory
        def info(hp, *, y, mu_mu, **_):
            return {'train': y - hp.get('m', mu_mu)}

        # fit hyperparameters
        gpkw = dict(
            y=y,
            z=z,
            i_mu=i_mu,
            i_tau=i_tau,
            pihat=pihat,
            x_aux=x_aux,
            weights=weights,
            splits_mu=splits_mu,
            splits_tau=splits_tau,
            ystd=ystd,
            mu_mu=mu_mu,
            k_sigma_mu=k_sigma_mu,
        )
        options = dict(
            verbosity=3,
            minkw=dict(method='l-bfgs-b', options=dict(maxls=4, maxiter=100)),
            mlkw=dict(epsrel=0),
            forward=True,
            gpfactorykw=gpkw,
        )
        options.update(fitkw)
        fit = _fit.empbayes_fit(hyperprior, makegp, info, **options)
        
        # extract hyperparameters from minimization result
        self.m = fit.p.get('m', mu_mu)
        self.sigma = gvar.sqrt(fit.p['sigma^2'])
        self.sdev_mu = fit.p['lambda_mu'] * ystd
        self.sdev_tau = fit.p['lambda_tau'] * ystd
        self.alpha_mu = fit.p['alpha_mu']
        self.alpha_tau = fit.p['alpha_tau']
        self.beta_mu = fit.p['beta_mu']
        self.beta_tau = fit.p['beta_tau']
        self.z_0 = fit.p['z_0']

        # save fit object
        self.fit = fit

    def _append_pihat(self, x_mu, x_tau, pihat):
        ip = self._include_pi
        if ip == 'mu' or ip == 'both':
            x_mu = _array.StructuredArray.from_dict(dict(
                x=x_mu,
                pihat=pihat,
            ))
        if x_tau is not None and (ip == 'tau' or ip == 'both'):
            x_tau = _array.StructuredArray.from_dict(dict(
                x=x_tau,
                pihat=pihat,
            ))
        return x_mu, x_tau

    def _join_points(self, train, z, i_mu, i_tau, pihat, x_aux):
        """ join covariates into a single StructuredArray """
        columns = dict(
            train=jnp.broadcast_to(bool(train), z.shape),
            i=jnp.arange(z.size).reshape(z.shape),
            z=z,
            mu=i_mu,
            tau=i_mu if i_tau is None else i_tau,
            pihat=pihat,
        )
        if x_aux is not None:
            columns.update(aux=x_aux)
        return _array.StructuredArray.from_dict(columns)

    @staticmethod
    def _error_cov(hp, weights, x):
        """ fill error covariance matrix """
        if weights is None:
            error_var = jnp.broadcast_to(hp['sigma^2'], len(x))
        else:
            error_var = hp['sigma^2'] / weights
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

    def gp(self, *, hp='map', z=None, x_mu=None, x_tau=None, pihat=None,
        x_aux=None, weights=None, rng=None):
        """
        Create a Gaussian process with the fitted hyperparameters.

        Parameters
        ----------
        hp : str or dict
            The hyperparameters to use. If ``'map'`` (default), use the marginal
            maximum a posteriori. If ``'sample'``, sample hyperparameters from
            the posterior. If a dict, use the given hyperparameters.
        z : (m,) array, series or dataframe, optional
            Treatment status at test points. If specified, also `x_mu`, `pihat`,
            `x_tau` and `x_aux` (the latter two if and only also specified at
            initialization) must be specified.
        x_mu : (m, p) array, series or dataframe, optional
            Control model covariates at test points.
        x_tau : (m, q) array, series or dataframe, optional
            Moderating model covariates at test points.
        pihat : (m,) array, series or dataframe, optional
            Estimated propensity score at test points.
        x_aux : (m, k) array, series or dataframe, optional
            Additional covariates for the ``'aux'`` process.
        weights : (m,) array, series or dataframe, optional
            Weights for the error variance on the test points.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'``.

        Returns
        -------
        gp : GP
            A centered Gaussian process object. To add the mean, use the `m`
            attribute of the `bcf` object. The keys of the GP are ``'Xmean'``,
            ``'Xnoise'``, and ``'X'``, where the "X" stands either for 'train'
            or 'test', and X = Xmean + Xnoise.
        """

        hp = self._gethp(hp, rng)
        return self._gp(hp, z, x_mu, x_tau, pihat, x_aux, weights, self.fit.gpfactorykw)

    def _gp(self, hp, z, x_mu, x_tau, pihat, x_aux, weights, gpfactorykw):
        """
        Internal function to create the GP object. This function must work
        both if the arguments are user-provided and need to be checked and
        converted to standard format, or if they are traced jax values.
        """

        # create GP object
        gp = self.fit.gpfactory(hp, **gpfactorykw)

        # add test points
        if z is not None:

            # check presence/absence of arguments is coherent
            self._check_coherent_covariates(z, x_mu, x_tau, pihat, x_aux)

            # check treatment and propensity score
            z = self._to_vector(z)
            pihat = self._to_vector(pihat)
            assert pihat.shape == z.shape

            # check weights
            if weights is not None:
                weights = self._to_vector(weights)
                assert weights.shape == z.shape

            # add propensity score to covariates
            x_mu = self._to_structured(x_mu)
            assert x_mu.shape == z.shape
            if x_tau is not None:
                x_tau = self._to_structured(x_tau)
                assert x_tau.shape == z.shape
            x_mu, x_tau = self._append_pihat(x_mu, x_tau, pihat)

            # convert covariates to indices
            i_mu = self._toindices(x_mu, gpfactorykw['splits_mu'])
            assert i_mu.dtype == gpfactorykw['i_mu'].dtype
            if x_tau is not None:
                i_tau = self._toindices(x_tau, gpfactorykw['splits_tau'])
                assert i_tau.dtype == gpfactorykw['i_tau'].dtype
            else:
                i_tau = None

            # check auxiliary points
            if x_aux is not None:
                x_aux = self._to_structured(x_aux)

            # add test points
            x = self._join_points(False, z, i_mu, i_tau, pihat, x_aux)
            gp.addx(x, 'testmean')
            errcov = self._error_cov(hp, weights, x)
            gp.addcov(errcov, 'testnoise')
            gp.addtransf({'testmean': 1, 'testnoise': 1}, 'test')

        return gp

    def _check_coherent_covariates(self, z, x_mu, x_tau, pihat, x_aux):
        if z is None:
            assert x_mu is None
            assert x_tau is None
            assert pihat is None
            assert x_aux is None
        else:
            assert x_mu is not None
            assert pihat is not None
            train_tau = self.fit.gpfactorykw['i_tau']
            if x_tau is None:
                assert train_tau is None
            else:
                assert train_tau is not None
            train_aux = self.fit.gpfactorykw['x_aux']
            if x_aux is None:
                assert train_aux is None
            else:
                assert train_aux is not None

    def data(self, *, hp='map', rng=None):
        """
        Get the data to be passed to `GP.pred` on a GP object returned by `gp`.

        Parameters
        ----------
        hp : str or dict
            The hyperparameters to use. If ``'map'`` (default), use the marginal
            maximum a posteriori. If ``'sample'``, sample hyperparameters from
            the posterior. If a dict, use the given hyperparameters.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'``.

        Returns
        -------
        data : dict
            A dictionary representing ``y`` in the format required by the
            `GP.pred` method.
        """

        hp = self._gethp(hp, rng)
        return self.fit.data(hp, **self.fit.gpfactorykw)

    def pred(self, *, hp='map', error=False, format='matrices', z=None,
        x_mu=None, x_tau=None, pihat=None, x_aux=None, weights=None, rng=None):
        r"""
        Predict the outcome at given locations.

        Parameters
        ----------
        hp : str or dict
            The hyperparameters to use. If ``'map'`` (default), use the marginal
            maximum a posteriori. If ``'sample'``, sample hyperparameters from
            the posterior. If a dict, use the given hyperparameters.
        error : bool
            If ``False`` (default), make a prediction for the latent mean. If
            ``True``, add the error term.     
        format : {'matrices', 'gvar'}
            If ``'matrices'`` (default), return the mean and covariance matrix
            separately. If ``'gvar'``, return an array of `~gvar.GVar`.
        z : (m,) array, series or dataframe, optional
            Treatment status at test points. If specified, also `x_mu`, `pihat`,
            `x_tau` and `x_aux` (the latter two if and only also specified at
            initialization) must be specified.
        x_mu : (m, p) array, series or dataframe, optional
            :math:`\mu` model covariates at test points.
        x_tau : (m, q) array, series or dataframe, optional
            :math:`\tau` model covariates at test points.
        pihat : (m,) array, series or dataframe, optional
            Estimated propensity score at test points.
        x_aux : (m, k) array, series or dataframe, optional
            Additional covariates for the ``'aux'`` process at test points.
        weights : (m,) array, series or dataframe, optional
            Weights for the error variance on the test points.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'``.

        Returns
        -------
        If `format` is ``'matrices'`` (default):

        mean, cov : arrays
            The mean and covariance matrix of the Normal posterior distribution
            over the regression function at the specified locations.

        If `format` is ``'gvar'``:

        out : array of gvar
            The same distribution represented as an array of `~gvar.GVar`
            objects.
        """
        
        # get hyperparameters
        hp = self._gethp(hp, rng)
        
        # check presence of covariates is coherent
        self._check_coherent_covariates(z, x_mu, x_tau, pihat, x_aux)

        # convert all inputs to arrays compatible with jax to pass them to the
        # compiled implementation
        if z is not None:
            z = self._to_vector(z)
            pihat = self._to_vector(pihat)
            x_mu = self._to_structured(x_mu)
            if x_tau is not None:
                x_tau = self._to_structured(x_tau)
            if x_aux is not None:
                x_aux = self._to_structured(x_aux)
        if weights is not None:
            weights = self._to_vector(weights)
        
        # GP regression
        mean, cov = self._pred(hp, z, x_mu, x_tau, pihat, x_aux, weights, self.fit.gpfactorykw, bool(error))

        # pack output
        if format == 'gvar':
            return gvar.gvar(mean, cov, fast=True)
        elif format == 'matrices':
            return mean, cov
        else:
            raise KeyError(format)

    @functools.cached_property
    def _pred(self):
        
        @functools.partial(jax.jit, static_argnums=(8,))
        def _pred(hp, z, x_mu, x_tau, pihat, x_aux, weights, gpfactorykw, error):
            gp = self._gp(hp, z, x_mu, x_tau, pihat, x_aux, weights, gpfactorykw)
            data = self.fit.data(hp, **gpfactorykw)
            if z is None:
                label = 'train'
            else:
                label = 'test'
            if not error:
                label += 'mean'
            outmean, outcov = gp.predfromdata(data, label, raw=True)
            return outmean + hp.get('m', gpfactorykw['mu_mu']), outcov

        # TODO make everything pure and jit this per class instead of per
        # instance

        return _pred

    @classmethod
    def _to_structured(cls, x, *, check_numerical=True):

        # convert to StructuredArray
        if hasattr(x, 'columns'):
            x = _array.StructuredArray.from_dataframe(x)
        elif hasattr(x, 'to_numpy'):
            x = _array.StructuredArray.from_dict({
                'f0' if x.name is None else x.name: x.to_numpy()
            })
        elif x.dtype.names is None:
            x = _array.unstructured_to_structured(x)
        else:
            x = _array.StructuredArray(x)

        # check fields are numerical, for BART
        if check_numerical:
            assert x.ndim == 1
            assert x.size > len(x.dtype)
            def check_numerical(path, dtype):
                if not numpy.issubdtype(dtype, numpy.number):
                    raise TypeError(f'covariate `{path}` is not numerical')
            cls._walk_dtype(x.dtype, check_numerical)

        return x

    @staticmethod
    def _to_vector(x):
        if hasattr(x, 'columns'): # dataframe
            x = x.to_numpy().squeeze(axis=1)
        elif hasattr(x, 'to_numpy'): # series (dataframe column)
            x = x.to_numpy()
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError(f'array is not 1d vector, ndim={x.ndim}')
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
        dtype = cast(x.dtype, ix.dtype)
        return _array.unstructured_to_structured(ix, dtype=dtype)

    def __repr__(self):
        if hasattr(self.m, 'sdev'):
            m = str(self.m)
        else:
            m = f'{self.m:.3g}'

        out = f"""BCF fit:
alpha_mu/tau = {self.alpha_mu} {self.alpha_tau} (0 -> intercept only, 1 -> any)
beta_mu/tau = {self.beta_mu} {self.beta_tau} (0 -> any, ∞ -> no interactions)
z_0 = {self.z_0} (mu model applies at z = z_0)
m = {m}
lambda_mu/tau std(y) = {self.sdev_mu} {self.sdev_tau} (large -> conservative extrapolation)
std(y) = {self.fit.gpfactorykw['ystd']:.3g}"""

        weights = self.fit.gpfactorykw['weights']
        if weights is None:
            out += f"""
sigma = {self.sigma}"""
        
        else:
            weights = numpy.array(weights) # to avoid jax taking over the ops
            avgsigma = numpy.sqrt(numpy.mean(self.sigma ** 2 / weights))
            out += f"""
sqrt(mean(sigma^2/w))  = {avgsigma}
sigma = {self.sigma}"""

        return out
