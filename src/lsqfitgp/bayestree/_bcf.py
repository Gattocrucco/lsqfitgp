# lsqfitgp/bayestree/_bcf.py
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
import warnings

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
from .. import _jaxext
from .. import _gvarext
from .. import _utils

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

    # TODO
    # - move this to generic utils
    # - make unit tests

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
        transf='standardize',
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
            `~lsqfitgp.GP` object under construction, and is expected to return
            a modified ``gp`` with a new process named ``'aux'`` defined with
            `~lsqfitgp.GP.defproc` or similar. The process is added to the
            regression model. The input to the process is a structured array
            with fields:

            'train' : bool
                Indicates whether the data is training set (the one passed on
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
            A dictionary with the prior of arbitrary additional hyperpameters,
            intended to be used by ``gpaux`` or ``transf``.
        transf : (list of) str or pair of callable
            Data transformation. Either a string indicating a pre-defined
            transformation, or a pair ``(from_data, to_data)``, two functions
            with signatures ``from_data(hp, y) -> eta`` and ``to_data(hp, eta)
            -> y``, where ``eta`` is the value to which the model is fit, and
            ``hp`` is the dictionary of hyperparameters. The functions must be
            ufuncs and one the inverse of the other w.r.t. the second parameter.
            ``from_data`` must be derivable with `jax` w.r.t. ``y``.

            If a list of such specifications is provided, the transformations
            are applied in order, with the first one being the outermost, i.e.,
            the one applied first to the data.

            If a transformation uses additional hyperparameters, either
            predefined automatically or passed by the user through `otherhp`,
            they are inferred with the rest of the hyperparameters.

            The pre-defined transformations are:

            'standardize' (default)
                eta = (y - mean(train_y)) / sdev(train_y)
            'yeojohnson'
                The Yeo-Johnson transformation [2]_ to reduce skewness. The
                :math:`\lambda` parameter is bounded in :math:`(0, 2)`
                for implementation convenience, this restriction may be lifted
                in future versions.
        
        Notes
        -----
        The regression model is:

        .. math::
            \eta_i = g(y_i; \ldots) &= m + {} \\
                &\phantom{{}={}} +
                    \lambda_\mu
                    \mu(\mathbf x^\mu_i, \hat\pi_i?) + {} \\
                &\phantom{{}={}} +
                    \lambda_\tau
                    \tau(\mathbf x^\tau_i, \hat\pi_i?) (z_i - z_0) + {} \\
                &\phantom{{}={}} +
                    \mathrm{aux}(i, z_i, \mathbf x^\mu_i, \mathbf x^\tau_i,
                        \hat\pi_i, \mathbf x^\text{aux}_i) + {} \\
                &\phantom{{}={}} +
                    \varepsilon_i, \\
            \varepsilon_i &\sim
                N(0, \sigma^2 / w_i), \\
            m &\sim N(0, 1), \\
            \log \sigma^2 &\sim N(\log\bar w, 4), \\
            \lambda_\mu
                &\sim \mathrm{HalfCauchy}(2), \\
            \lambda_\tau
                &\sim \mathrm{HalfNormal}(1.48), \\
            \mu &\sim \mathrm{GP}(0,
                \mathrm{BART}(\alpha_\mu, \beta_\mu) ), \\
            \tau &\sim \mathrm{GP}(0,
                \mathrm{BART}(\alpha_\tau, \beta_\tau) ), \\
            \mathrm{aux} & \sim \mathrm{GP}(0, \text{<user defined>}), \\
            \alpha_\mu, \alpha_\tau &\sim \mathrm{Beta}(2, 1), \\
            \beta_\mu, \beta_\tau &\sim \mathrm{InvGamma}(1, 1), \\
            z_0 &\sim U(0, 1),

        To make the inference, :math:`(\mu, \tau, \boldsymbol\varepsilon, m,
        \mathrm{aux})` are marginalized analytically, and the marginal posterior
        mode of
        :math:`(\sigma, \lambda_*, \alpha_*, \beta_*, z_0, \ldots)` is found by
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
        lambda_mu, lambda_tau : gvar
            The prior standard deviation :math:`\lambda_*`.
        z_0 : gvar
            The treatment coding parameter.
        fit : empbayes_fit
            The hyperparameters fit object.

        Methods
        -------
        gp :
            Create a GP object.
        data :
            Creates the dictionary to be passed to `GP.pred` to represent data.
        pred :
            Evaluate the regression function at given locations.
        from_data :
            Convert :math:`y` to :math:`\eta`.
        to_data :
            Convert :math:`\eta` to :math:`y`.

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
        .. [2] Yeo, In-Kwon; Johnson, Richard A. (2000). "A New Family of Power
            Transformations to Improve Normality or Symmetry". Biometrika. 87
            (4): 954–959. https://doi.org/10.1093/biomet/87.4.954
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

        # get functions for data transformation
        from_data, to_data, transfloss, transfhp = self._get_transf(
            transf=transf, weights=weights, y=y)

        # scale of error variance
        logsigma2_loc = 0 if weights is None else numpy.log(jnp.mean(weights))

        # prior on hyperparams
        hyperprior = copula.makedict({
            'm': gvar.gvar(0, 1),
            'sigma^2': copula.lognorm(logsigma2_loc, 2),
            'lambda_mu': copula.halfcauchy(2),
            'lambda_tau': copula.halfnorm(1.48),
            'alpha_mu': copula.beta(2, 1),
            'alpha_tau': copula.beta(2, 1),
            'beta_mu': copula.invgamma(1, 1),
            'beta_tau': copula.invgamma(1, 1),
            'z_0': copula.uniform(0, 1),
        })

        # remove explicit mean parameter if it's baked into the Gaussian process
        if marginalize_mean:
            hyperprior.pop('m')

        # add data transformation and user hyperparameters
        def update_hyperparams(new, newname, raises):
            new = gvar.BufferDict(new)
            for key in new.all_keys():
                if hyperprior.has_dictkey(key):
                    message = f'{newname} hyperparameter {key!r} overrides existing one'
                    if raises:
                        raise ValueError(message)
                    else:
                        warnings.warn(message)
            hyperprior.update(new)
        update_hyperparams(transfhp, 'data transformation', True)
            # the hypers handed by _get_transf are not allowed to override
        update_hyperparams(otherhp, 'user', False)

        # GP factory
        def gpfactory(hp, *, z, i_mu, i_tau, pihat, x_aux, weights,
            splits_mu, splits_tau, **_):
            
            # TODO maybe I should pass kernelkw_* as arguments, but they may not
            # be jittable. I need jitkw in empbayes_fit for that.

            kw_overridable = dict(
                maxd=10,
                reset=[2, 4, 6, 8],
                intercept=False,
            )
            kw_not_overridable = dict(indices=True)

            gp = _GP.GP(checkpos=False, checksym=False, solver='chol')

            for name, kernelkw in dict(mu=kernelkw_mu, tau=kernelkw_tau).items():
                kw = dict(
                    alpha=hp[f'alpha_{name}'],
                    beta=hp[f'beta_{name}'],
                    dim=name,
                    splits=eval(f'splits_{name}'),
                    **kw_overridable,
                )
                kw.update(kernelkw)
                kernel = _kernels.BART(**kw, **kw_not_overridable)
                kernel *= hp[f'lambda_{name}'] ** 2
                
                gp = gp.defproc(name, kernel)

            if 'm' in hp:
                kernel_mean = 0 * _kernels.Constant()
            else:
                kernel_mean = _kernels.Constant()
            gp = gp.defproc('m', kernel_mean)

            if gpaux is None:
                gp = gp.defproc('aux', 0 * _kernels.Constant())
            else:
                gp = gpaux(hp, gp)

            gp = gp.deflintransf(
                gp.DefaultProcess,
                lambda m, mu, tau, aux: lambda x:
                m(x) + mu(x) + tau(x) * (x['z'] - hp['z_0']) + aux(x),
                ['m', 'mu', 'tau', 'aux'],
            )
            
            x = self._join_points(True, z, i_mu, i_tau, pihat, x_aux)
            gp = gp.addx(x, 'trainmean')
            errcov = self._error_cov(hp, weights, x)
            return (gp
                .addcov(errcov, 'trainnoise')
                .addtransf({'trainmean': 1, 'trainnoise': 1}, 'train')
            )

        # data factory
        def data(hp, *, y, **_):
            return {'train': from_data(hp, y) - hp.get('m', 0)}

        # fit hyperparameters
        options = dict(
            verbosity=3,
            minkw=dict(
                method='l-bfgs-b',
                options=dict(
                    maxls=4,
                    maxiter=100,
                ),
            ),
            mlkw=dict(
                epsrel=0,
            ),
            forward=True,
            gpfactorykw=dict(
                y=y,
                z=z,
                i_mu=i_mu,
                i_tau=i_tau,
                pihat=pihat,
                x_aux=x_aux,
                weights=weights,
                splits_mu=splits_mu,
                splits_tau=splits_tau,
            ),
            additional_loss=transfloss,
        )
        options.update(fitkw)
        fit = _fit.empbayes_fit(hyperprior, gpfactory, data, **options)
        
        # extract hyperparameters from minimization result
        self.m = fit.p.get('m', 0)
        self.sigma = gvar.sqrt(fit.p['sigma^2'])
        self.lambda_mu = fit.p['lambda_mu']
        self.lambda_tau = fit.p['lambda_tau']
        self.alpha_mu = fit.p['alpha_mu']
        self.alpha_tau = fit.p['alpha_tau']
        self.beta_mu = fit.p['beta_mu']
        self.beta_tau = fit.p['beta_tau']
        self.z_0 = fit.p['z_0']

        # save other attributes
        self.fit = fit
        self._from_data = from_data
        self._to_data = to_data

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

    @staticmethod
    def _join_points(train, z, i_mu, i_tau, pihat, x_aux):
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
            attribute of the `bcf` object. The keys of the GP are ``'@mean'``,
            ``'@noise'``, and ``'@'``, where the "@" stands either for 'train'
            or 'test', and @ = @mean + @noise.

            This Gaussian process is defined on the transformed data ``eta``.
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
            gp = gp.addx(x, 'testmean')
            errcov = self._error_cov(hp, weights, x)
            gp = (gp
                .addcov(errcov, 'testnoise')
                .addtransf({'testmean': 1, 'testnoise': 1}, 'test')
            )

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
            A dictionary representing ``eta`` in the format required by the
            `GP.pred` method.
        """

        hp = self._gethp(hp, rng)
        return self.fit.data(hp, **self.fit.gpfactorykw)

    def pred(self, *, hp='map', error=False, z=None, x_mu=None, x_tau=None,
        pihat=None, x_aux=None, weights=None, transformed=True, samples=None,
        gvars=False, rng=None):
        r"""
        Predict the transformed outcome at given locations.

        Parameters
        ----------
        hp : str or dict
            The hyperparameters to use. If ``'map'`` (default), use the marginal
            maximum a posteriori. If ``'sample'``, sample hyperparameters from
            the posterior. If a dict, use the given hyperparameters.
        error : bool, default False
            If ``False``, make a prediction for the latent mean. If ``True``,
            add the error term.
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
        transformed : bool, default True
            If ``True``, return the prediction on the transformed outcome
            :math:`\eta`, else the observable outcome :math:`y`.
        samples : int, optional
            If specified, indicates the number of samples to take from the
            posterior. If not, return the mean and covariance matrix of the
            posterior.
        gvars : bool, default False
            If ``True``, return the mean and covariance matrix of the posterior
            as an array of `GVar` variables.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'`` or ``samples``
            is not `None`.

        Returns
        -------
        If ``samples`` is `None` and ``gvars`` is `False` (default):

        mean, cov : (m,) and (m, m) arrays
            The mean and covariance matrix of the Normal posterior distribution
            over the regression function or :math:`\eta` at the specified
            locations.

        If ``samples`` is `None` and ``gvars`` is `True`:

        out : (m,) array of gvars
            The same distribution represented as an array of `~gvar.GVar`
            objects.

        If ``samples`` is an integer:

        sample : (samples, m) array
            Posterior samples over either the regression function, :math:`\eta`,
            or :math:`y`.
        """

        # check consistency of output choice
        if samples is None:
            if not transformed:
                raise ValueError('Posterior is required in analytical form '
                                 '(samples=None) and in data space '
                                 '(transformed=False), this is not possible as '
                                 'the transformation model space -> data space '
                                 'is arbitrary. Either sample the posterior or '
                                 'get the result in model space.')
        else:
            if not transformed and not error:
                raise ValueError('Posterior is required in data space '
                                 '(transformed=False) and without error term '
                                 '(error=False), this is not possible as the '
                                 'transformation model space -> data space '
                                 'applies after adding the error.')
            assert not gvars, 'can not represent posterior samples as gvars'

        # TODO allow exceptions to these rules when there are no transformations
        # or the only transformation is 'standardize'.
        
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

        # return Normal posterior moments
        if samples is None:
            if gvars:
                return gvar.gvar(mean, cov, fast=True)
            else:
                return mean, cov

        # sample from posterior
        sample = jnp.stack(list(_fastraniter.raniter(mean, cov, n=samples, rng=rng)))
            # TODO when I add vectorized sampling, use it here
        if not transformed:
            sample = self._to_data(hp, sample)
        return sample

        # TODO the default should be something in data space, so with samples.
        # If I handle the analyitical posterior through standardize, I could
        # also make it without samples by default. Although I guess for
        # whatever calculations samples are more convenient (just do the
        # calculation on the samples.)

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
            return outmean + hp.get('m', 0), outcov

        # TODO make everything pure and jit this per class instead of per
        # instance

        return _pred

    def from_data(self, y, *, hp='map', rng=None):
        """
        Transforms outcomes :math:`y` to the regression variable :math:`\\eta`.

        Parameters
        ----------
        y : (n,) array
            Outcomes.
        hp : str or dict
            The hyperparameters to use. If ``'map'`` (default), use the marginal
            maximum a posteriori. If ``'sample'``, sample hyperparameters from
            the posterior. If a dict, use the given hyperparameters.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'``.

        Returns
        -------
        eta : (n,) array
            Transformed outcomes.
        """

        hp = self._gethp(hp, rng)
        return self._from_data(hp, y)

    def to_data(self, eta, *, hp='map', rng=None):
        """
        Convert the regression variable :math:`\\eta` to outcomes :math:`y`.

        Parameters
        ----------
        eta : (n,) array
            Transformed outcomes.
        hp : str or dict
            The hyperparameters to use. If ``'map'`` (default), use the marginal
            maximum a posteriori. If ``'sample'``, sample hyperparameters from
            the posterior. If a dict, use the given hyperparameters.
        rng : numpy.random.Generator, optional
            Random number generator, used if ``hp == 'sample'``.

        Returns
        -------
        y : (n,) array
            Outcomes.
        """

        hp = self._gethp(hp, rng)
        return self._to_data(hp, eta)

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

        with _gvarext.gvar_format():
            if hasattr(self.m, 'sdev'):
                m = str(self.m)
            else:
                m = f'{self.m:.3g}'

            n = self.fit.gpfactorykw['y'].size
            p_mu = _array._nd(self.fit.gpfactorykw['i_mu']['x'].dtype)
            x_tau = self.fit.gpfactorykw['i_tau']
            x_aux = self.fit.gpfactorykw['x_aux']

            out = f"""\
Data:
    n = {n}"""

            if x_tau is None:
                out += f"""
    p = {p_mu}"""
            else:
                p_tau = _array._nd(x_tau['x'].dtype)
                out += f"""
    p_mu/tau = {p_mu}, {p_tau}"""

            if x_aux is not None:
                p_aux = _array._nd(x_aux['x'].dtype)
                out += f"""
    p_aux = {p_aux}"""

            out += f"""
Hyperparameter posterior:
    m = {m}
    z_0 = {self.z_0}
    alpha_mu/tau = {self.alpha_mu} {self.alpha_tau}
    beta_mu/tau = {self.beta_mu} {self.beta_tau}
    lambda_mu/tau = {self.lambda_mu} {self.lambda_tau}"""

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

        out += """
Meaning of hyperparameters:
    mu(x) = reference outcome level
    tau(x) = effect of the treatment
    z_0 in (0, 1): reference treatment level
        z_0 -> 0: mu is the model of the untreated
        z_0 -> 1: mu is the model of the treated
    alpha in (0, 1)
        alpha -> 0: constant function
        alpha -> 1: no constraints on the function
    beta in (0, ∞)
        beta -> 0: no constraints on the function
        beta -> ∞: no interactions, f(x) = f1(x1) + f2(x2) + ...
    lambda in (0, ∞): standard deviation of function
        lambda small: confident extrapolation
        lambda large: conservative extrapolation
    sigma in (0, ∞): standard deviation of i.i.d. error"""
        
        return _utils.top_bottom_rule('BCF', out)

        # TODO print user parameters, applying transformations. Copy the dict and use .pop() to remove the predefined params as they are printed.

    def _get_transf(self, *, transf, y, weights):

        from_datas = []
        to_datas = []
        hypers = {}

        if transf is None:
            transf = []
        elif isinstance(transf, list):
            name = lambda n: f'transf{i}_{n}'
        else:
            name = lambda n: n
            transf = [transf]
        
        for i, tr in enumerate(transf):

            hyper = {}
        
            if not isinstance(tr, str):
                
                from_data, to_data = tr
                
            elif tr == 'standardize':

                if i > 0:
                    warnings.warn('standardization applied after other '
                        'transformations: standardization always uses the '
                        'initial data mean and standard deviation, so it may '
                        'not work as intended')

                    # It's not possible to overcome this limitation if one wants
                    # to stick to transformations that act on one point at a
                    # time to make them generalizable out of sample.

                if weights is None:
                    loc = jnp.mean(y)
                    scale = jnp.std(y)
                else:
                    loc = jnp.average(y, weights=weights)
                    scale = jnp.sqrt(jnp.average((y - loc) ** 2, weights=weights))
                
                def from_data(hp, y):
                    return (y - loc) / scale
                def to_data(hp, eta):
                    return loc + scale * eta

            elif tr == 'yeojohnson':
                    
                def from_data(hp, y):
                    return yeojohnson(y, hp[name('lambda_yj')])
                def to_data(hp, eta):
                    return yeojohnson_inverse(eta, hp[name('lambda_yj')])
                hyper[name('lambda_yj')] = 2 * copula.beta(2, 2)

            else:
                raise KeyError(tr)
            
            from_datas.append(from_data)
            to_datas.append(to_data)
            hypers.update(hyper)

        if transf:
            def from_data(hp, y):
                for fd in from_datas:
                    y = fd(hp, y)
                return y
            def to_data(hp, eta):
                for td in reversed(to_datas):
                    eta = td(hp, eta)
                return eta
        else:
            from_data = lambda hp, y: y
            to_data = lambda hp, eta: eta

        from_data_grad = _jaxext.elementwise_grad(from_data, 1)
        def loss(hp):
            return -jnp.sum(jnp.log(from_data_grad(hp, y)))

        hypers = copula.makedict(hypers)

        return from_data, to_data, loss, hypers

def yeojohnson(x, lmbda):
    """ Yeo-Johnson transformation with lamda != 0, 2 """
    return jnp.where(
        x >= 0,
        (jnp.power(x + 1, lmbda) - 1) / lmbda,
        -((jnp.power(-x + 1, 2 - lmbda) - 1) / (2 - lmbda))
    )

    # TODO
    # - rewrite the cases with expm1, log1p, etc. to make them accurate
    # - split the cases into lambda 0/2
    # - make custom_jvps for the singular points to define derivatives w.r.t.
    #   lambda even though it does not appear in the expression
    # - add unit tests that check gradients with finite differences

def yeojohnson_inverse(y, lmbda):
    return jnp.where(
        y >= 0,
        jnp.power(y * lmbda + 1, 1 / lmbda) - 1,
        -jnp.power(-(2 - lmbda) * y + 1, 1 / (2 - lmbda)) + 1
    )
