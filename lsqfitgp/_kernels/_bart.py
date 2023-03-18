# lsqfitgp/_kernels/_bart.py
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

import jax
from jax import numpy as jnp
from jax import lax
from jax.scipy import special as jspecial
from numpy.lib import recfunctions

from .. import _patch_jax
from .. import _array
from .._Kernel import kernel

def _bart_maxdim(splits=None, **_):
    splits = BART._check_splits(splits)
    return splits[0].size

@kernel(maxdim=_bart_maxdim, derivable=False, batchbytes=10e6)
# TODO maybe batching should be done automatically by GP instead of by the
# kernels? But before doing that I need to support batching non-traceable
# functions.
def _BARTBase(x, y,
    alpha=0.95,
    beta=2,
    maxd=2,
    gamma=1,
    splits=None,
    pnt=None,
    intercept=True,
    weights=None,
    reset=None,
    altimpl=False,
    indices=False):
    """
    BART kernel.
    
    Parameters
    ----------
    alpha, beta : scalar
        The parameters of the branching probability.
    maxd : int
        The maximum depth of the trees.
    splits : pair of arrays
        The first is an int (p,) array containing the number of splitting
        points along each dimension, the second has shape (n, p) and contains
        the sorted splitting points in each column, filled with high values
        after the length.
    gamma : scalar or str
        Interpolation coefficient in [0, 1] between a lower and a upper
        bound on the infinite maxd limit, or a string 'auto' indicating to
        use a formula which depends on alpha, beta, maxd and the number of
        covariates, empirically calibrated on maxd from 1 to 3. Default 1
        (upper bound).
    pnt : (maxd + 1,) array, optional
        Nontermination probabilities at depths 0...maxd. If specified,
        `alpha`, `beta` and `maxd` are ignored.
    intercept : bool
        The correlation is in [1 - alpha, 1] (or [1 - pnt[0], 1] when using
        pnt). If intercept=False, it is rescaled to [0, 1]. Default True.
    weights : (p,) array
        Unnormalized selection probabilities for the covariate axes. If not
        specified, all axes have the same probability to be selected for
        splitting.
    reset : int or sequence of int, optional
        List of depths at which the recursion is reset, in the sense that the
        function value at a reset depth is evaluated on the initial inputs for
        all recursion paths, instead of the modified input handed down by the
        recursion. Default none.
    
    Methods
    -------
    splits_from_coord
    indices_from_coord
    correlation
    
    Notes
    -----
    This is the covariance function of the latent mean prior of BART (Bayesian
    Additive Regression Trees) [1]_ in the limit of an infinite number of
    trees, and with an upper bound :math:`D` on the depth of the trees. This
    prior is the distribution of the function
    
    .. math::
        f(\\mathbf x) = \\lim_{m\\to\\infty}
        \\sum_{j=1}^m g(\\mathbf x; T_j, M_j),
    
    where each :math:`g(\\mathbf x; T_j, M_j)` is a decision tree evaluated at
    :math:`\\mathbf x`, with structure :math:`T_j` and leaf values :math:`M_j`.
    The trees are i.i.d., with the following distribution for :math:`T_j`: for
    a node at depth :math:`d`, with :math:`d = 0` for the root, the probability
    of not being a leaf, conditional on its existence and its ancestors only, is
    
    .. math::
        P_d = \\alpha (1+d)^{-\\beta}, \\quad
        \\alpha \\in [0, 1], \\quad \\beta \\ge 0.
    
    For a non-leaf node, conditional on existence and ancestors, the splitting
    variable has uniform distribution amongst the variables with any splitting
    points not used by ancestors, and the splitting point has uniform
    distribution amongst the available ones. The splitting points are fixed,
    tipically from the data.
    
    The distribution of leaves :math:`M_j` is i.i.d. Normal with variance
    :math:`1/m`, such that :math:`f(x)` has variance 1. In the limit
    :math:`m\\to\\infty`, the distribution of :math:`f(x)` becomes a Gaussian
    process.
    
    Since the trees are independent, the covariance function can be computed
    for a single tree. Consider two coordinates :math:`x` and :math:`y`, with
    :math:`x \\le y`. Let :math:`n^-`, :math:`n^0` and :math:`n^+` be the
    number of splitting points respectively before :math:`x`, between
    :math:`x`, :math:`y` and after :math:`y`. Next, define :math:`\\mathbf
    n^-`, :math:`\\mathbf n^0` and :math:`\\mathbf n^+` as the vectors of such
    quantities for each dimension, with a total of :math:`p` dimensions, and
    :math:`\\mathbf n = \\mathbf n^- + \\mathbf n^0 + \\mathbf n^+`. Then the
    covariance function can be written recursively as
    
    .. math::
        \\newcommand{\\nvecs}{\\mathbf n^-, \\mathbf n^0, \\mathbf n^+}
        k(\\mathbf x, \\mathbf y) &= k_0(\\nvecs), \\\\
        k_D(\\nvecs) &= 1 - (1 - \\gamma) P_D,
            \\quad \\mathbf n^0 \\ne \\mathbf 0, \\\\
        k_d(\\mathbf 0, \\mathbf 0, \\mathbf 0) &= 1, \\\\
        k_d(\\nvecs) &= 1 - P_d \\Bigg(1 - \\frac1{W(\\mathbf n)}
            \\sum_{\\substack{i=1 \\\\ n_i\\ne 0}}^p
                \\frac{w_i}{n_i} \\Bigg( \\\\
                &\\qquad \\sum_{k=0}^{n^-_i - 1}
                k_{d+1}(\\mathbf n^-_{n^-_i=k}, \\mathbf n^0, \\mathbf n^+)
                + {} \\\\
                &\\qquad \\sum_{k=0}^{n^+_i - 1}
                k_{d+1}(\\mathbf n^-, \\mathbf n^0, \\mathbf n^+_{n^+_i=k})
            \\Bigg)
        \\Bigg), \\quad d < D, \\\\
        W(\\mathbf n) &= \\sum_{\\substack{i=1 \\\\ n_i\\ne 0}}^p w_i.
        
    The introduction of a maximum depth :math:`D` is necessary for
    computational feasibility. As :math:`D` increases, the result converges to
    the one without depth limit. For :math:`D \\le 2` (the default value), the
    covariance is implemented in closed form and takes :math:`O(p)` to compute.
    For :math:`D > 2`, the computational complexity grows exponentially as
    :math:`O(p(\\bar np)^{D-2})`, where :math:`\\bar n` is the average number of splitting
    points along a dimension.
    
    In the maximum allowed depth is 1, i.e., either :math:`D = 1` or
    :math:`\\beta\\to\\infty`, the kernel assumes the simple form
    
    .. math::
        k(\\mathbf x, \\mathbf y) &= 1 - P_0 \\left(
            1 - Q + \\frac Q{W(\\mathbf n)}
            \\sum_{\\substack{i=1 \\\\ n_i\\ne 0}}^p w_i
            \\frac{n^0_i}{n_i} \\right), \\\\
        Q &= \\begin{cases}
            1 - (1 - \\gamma) P_1 & \\mathbf n^0 \\ne \\mathbf 0, \\\\
            1 & \\mathbf n^0 = \\mathbf 0,
        \\end{cases}
    
    which is separable along dimensions, i.e., it has no interactions.
    
    References
    ----------
    .. [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch "BART:
        Bayesian additive regression trees," The Annals of Applied Statistics,
        Ann. Appl. Stat. 4(1), 266-298, (March 2010).
    """

    splits = BART._check_splits(splits)
    if not x.dtype.names:
        x = x[..., None]
    if not y.dtype.names:
        y = y[..., None]
    if indices:
        ix = BART._check_x(x)
        iy = BART._check_x(y)
        assert jnp.issubdtype(ix.dtype, jnp.integer)
        assert jnp.issubdtype(iy.dtype, jnp.integer)
        with _patch_jax.skipifabstract():
            assert jnp.all((0 <= ix) & (ix <= splits[0]))
            assert jnp.all((0 <= iy) & (iy <= splits[0]))
    else:
        ix = BART.indices_from_coord(x, splits)
        iy = BART.indices_from_coord(y, splits)
    if altimpl:
        arg1 = splits[0]
        arg2 = ix
        arg3 = iy
    else:
        l = jnp.minimum(ix, iy)
        r = jnp.maximum(ix, iy)
        arg1 = l
        arg2 = r - l
        arg3 = splits[0] - r
    return BART.correlation(
        arg1, arg2, arg3,
        pnt=pnt, alpha=alpha, beta=beta, gamma=gamma, maxd=maxd,
        intercept=intercept, weights=weights, reset=reset, altinput=altimpl,
    )
    
    # TODO
    # - interpolate (is linear interp pos def? => I think it can be seen as
    #   the covariance of the interpolation) => it is slow for high p
    # - approximate as stationary w.r.t. indices (is it pos def?)
    # - allow index input
    # - default gamma='auto'? => wait for better gamma
    # - make gamma='auto' depend on maxd and reset with a dictionary, error
    #   if not specified
    # - use as default, for the time being, maxd=4, reset=2, gamma=0.95?

class BART(_BARTBase):
    
    __doc__ = _BARTBase.__doc__
    
    @classmethod
    def splits_from_coord(cls, x):
        """
        Generate splitting points from data.
        
        Parameters
        ----------
        x : array of numbers
            The data. Can be passed in two formats: 1) a structured array where
            each leaf field represents a dimension, 2) a normal array where the
            last axis runs over dimensions. In the structured case, each
            index in any shaped field is a different dimension.
        
        Returns
        -------
        length : int (p,) array
            The number of splitting points along each of `p` dimensions.
        splits : (n, p) array
            Each column contains the sorted splitting points along a dimension.
            The splitting points are the midpoints between consecutive values
            appearing in `x` for that dimension. Column ``splits[:, i]``
            contains splitting points only up to ``length[i]``, while afterward
            it is filled with a very large value.
        
        """
        x = cls._check_x(x)
        x = x.reshape(-1, x.shape[-1]) if x.size else x.reshape(1, x.shape[-1])
        if jnp.issubdtype(x.dtype, jnp.inexact):
            info = jnp.finfo
        else:
            info = jnp.iinfo
        fill = info(x.dtype).max
        def loop(_, xi):
            u = jnp.unique(xi, size=xi.size, fill_value=fill)
            m = jnp.where(u[1:] < fill, (u[1:] + u[:-1]) / 2, fill)
            l = jnp.searchsorted(m, fill)
            return _, (l, m)
        _, (length, midpoints) = lax.scan(loop, None, x.T)
        return length, midpoints.T
        
        # TODO options like BayesTree, i.e., use an evenly spaced range
        # instead of quantilizing, and set a maximum number of splits. Use the
        # same parameter names as BayesTree::bart, but change the defaults.
    
    @classmethod
    def indices_from_coord(cls, x, splits):
        """
        Convert coordinates to indices w.r.t. splitting points.
        
        Parameters
        ----------
        x : array of numbers
            The coordinates. Can be passed in two formats: 1) a structured
            array where each leaf field represents a dimension, 2) a normal
            array where the last axis runs over dimensions. In the structured
            case, each index in any shaped field is a different dimension.
        splits : pair of arrays
            The first is an int (p,) array containing the number of splitting
            points along each dimension, the second has shape (n, p) and
            contains the sorted splitting points in each column, filled with
            high values after the length.
        
        Returns
        -------
        ix : int array
            An array with the same shape as `x`, unless `x` is a structured
            array, in which case the last axis of `ix` is the flattened version
            of the structured type. `ix` contains indices mapping `x` to
            positions between splitting points along each coordinate, with the
            following convention: index 0 means before the first split, index
            i > 0 means between split i - 1 and split i.
        
        """
        x = cls._check_x(x)
        splits = cls._check_splits(splits)
        assert x.shape[-1] == splits[0].size
        return cls._searchsorted_vectorized(splits[1], x)
    
    @classmethod
    def correlation(cls,
        splitsbefore_or_totalsplits,
        splitsbetween_or_index1,
        splitsafter_or_index2,
        *,
        alpha=0.95,
        beta=2,
        gamma=1,
        maxd=2,
        debug=False,
        pnt=None,
        intercept=True,
        weights=None,
        reset=None,
        altinput=False):
        """
        Compute the BART prior correlation between two points.

        Apart from arguments `maxd`, `debug` and `reset`, this method is fully
        vectorized.
    
        Parameters
        ----------
        splitsbefore_or_totalsplits : int (p,) array
            The number of splitting points less than the two points, separately
            along each coordinate, or the total number of splits if `altinput`.
        splitsbetween_or_index1 : int (p,) array
            The number of splitting points between the two points, separately
            along each coordinate, or the index in the splitting bins of the
            first point if `altinput`, where 0 means to the left of the leftmost
            splitting point.
        splitsafter_or_index2 : int (p,) array
            The number of splitting points greater than the two points,
            separately along each coordinate, or the index in the splitting bins
            of the second point if `altinput`.
        debug : bool
            If True, disable shortcuts in the tree recursion. Default False.
        altinput : bool
            If True, take as input the indices in the splitting bins of the
            points instead of the counts of splitting points separating them,
            and use a different implementation optimized for that case. Default
            False.
        Other parameters :
            See :class:`BART`.

        Returns
        -------
        corr : scalar
            The prior correlation.
        """
        
        # check splitting points are integers
        splitsbefore_or_totalsplits = jnp.asarray(splitsbefore_or_totalsplits)
        splitsbetween_or_index1 = jnp.asarray(splitsbetween_or_index1)
        splitsafter_or_index2 = jnp.asarray(splitsafter_or_index2)
        assert jnp.issubdtype(splitsbefore_or_totalsplits.dtype, jnp.integer)
        assert jnp.issubdtype(splitsbetween_or_index1.dtype, jnp.integer)
        assert jnp.issubdtype(splitsafter_or_index2.dtype, jnp.integer)
        
        # check splitting points are nonnegative
        with _patch_jax.skipifabstract():
            assert jnp.all(splitsbefore_or_totalsplits >= 0)
            assert jnp.all(splitsbetween_or_index1 >= 0)
            assert jnp.all(splitsafter_or_index2 >= 0)
        
        # get splitting probabilities
        if pnt is None:
            assert maxd == int(maxd) and maxd >= 0, maxd
            alpha = jnp.asarray(alpha)
            beta = jnp.asarray(beta)
            with _patch_jax.skipifabstract():
                assert jnp.all((0 <= alpha) & (alpha <= 1))
                assert jnp.all(beta >= 0)
            d = jnp.arange(maxd + 1)
            alpha = alpha[..., None]
            beta = beta[..., None]
            pnt = alpha / (1 + d) ** beta
        else:
            pnt = jnp.asarray(pnt)
        
        # get covariate weights
        if weights is None:
            weights = jnp.ones(splitsbefore_or_totalsplits.shape[-1], pnt.dtype)
        else:
            weights = jnp.asarray(weights)
        
        # get interpolation coefficients
        if isinstance(gamma, str):
            if gamma == 'auto':
                assert reset is None and 1 <= pnt.shape[-1] - 1 <= 3
                p = weights.shape[-1]
                gamma = cls._gamma(p, pnt)
            else:
                raise KeyError(gamma)
        else:
            gamma = jnp.asarray(gamma)
        
        # check values are in range
        with _patch_jax.skipifabstract():
            assert jnp.all((0 <= gamma) & (gamma <= 1))
            assert jnp.all((0 <= pnt) & (pnt <= 1))
            assert jnp.all(weights >= 0)

        # set first splitting probability to 1 to remove flat baseline (keep
        # last!)
        if not intercept:
            pnt = pnt.at[..., 0].set(1)
        
        # expand and check recursion reset depths
        if reset is None:
            reset = []
        if not hasattr(reset, '__len__'):
            reset = [reset]
        reset = [0] + list(reset) + [pnt.shape[-1] - 1]
        for i, j in zip(reset, reset[1:]):
            assert int(j) == j and i <= j, (i, j)
        
        # call recursive function for each recursion slice
        corr = gamma
        for t, b in zip(reversed(reset[:-1]), reversed(reset)):
            probs = pnt[..., t:b + 1]
            if t > 0:
                probs = probs.at[..., 0].set(1)
            corr = cls._correlation_vectorized(
                splitsbefore_or_totalsplits,
                splitsbetween_or_index1,
                splitsafter_or_index2,
                probs, corr, weights,
                debug, altinput,
            )
        return corr
    
    # TODO public method to compute pnt
        
    @staticmethod
    def _gamma(p, pnt):
        # gamma(alpha, beta, maxd) =
        #   = (gamma_0 - gamma_d maxd) (1 - alpha^s 2^(-t beta)) =
        #   = (gamma_0 - gamma_d maxd) (1 - P0^s-t P1^t)

        gamma_0 = 0.611 + 0.021 * jnp.exp(-1.3 * (p - 1))
        gamma_d = -0.0034 + 0.084 * jnp.exp(-2.02 * (p - 1))
        s = 2.03 - 0.69 * jnp.exp(-0.72 * (p - 1))
        t = 4.01 - 1.49 * jnp.exp(-0.77 * (p - 1))

        maxd = pnt.shape[-1] - 1
        floor = jnp.clip(gamma_0 - gamma_d * maxd, 0, 1)

        P0 = pnt[..., 0]
        P1 = jnp.minimum(P0, pnt[..., 1])
        corner = jnp.where(P0, 1 - P0 ** (s - t) * P1 ** t, 1)
        
        return floor * corner
        
        # TODO make this public?

    @staticmethod
    def _check_x(x):
        x = _array.asarray(x)
        if x.dtype.names:
            x = recfunctions.structured_to_unstructured(x)
        return x

    @staticmethod
    def _check_splits(splits):
        l, s = splits
        l = jnp.asarray(l)
        s = jnp.asarray(s)
        assert l.ndim == 1 and 1 <= s.ndim <= 2
        if s.ndim == 1:
            s = s[:, None]
        assert l.size == s.shape[1]
        with _patch_jax.skipifabstract():
            assert jnp.all((0 <= l) & (l <= s.shape[0]))
            assert jnp.all(jnp.sort(s, axis=0) == s)
        return l, s
    
    @staticmethod
    @functools.partial(jax.jit, static_argnames=('side',))
    def _searchsorted_vectorized(A, V, **kw):
        """
        A : (n, p)
        V : (..., p)
        out : (..., p)
        """
        def loop(_, av):
            return _, jnp.searchsorted(*av, **kw)
        _, out = lax.scan(loop, None, (A.T, V.T))
        return out.T

    @classmethod
    @functools.partial(jax.jit, static_argnums=(0, 7))
    def _correlation(cls, nminus, n0, nplus, pnt, gamma, w, debug):

        assert nminus.shape == n0.shape == nplus.shape == w.shape
        assert nminus.ndim == 1 and nminus.size >= 0
        assert pnt.ndim == 1 and pnt.size > 0
        # TODO repeat this shape checks in BART.correlation such that the
        # error messages are user-legible

        # optimization to avoid looping over ignored axes
        nminus = jnp.where(w, nminus, 0)
        n0 = jnp.where(w, n0, 0)
        nplus = jnp.where(w, nplus, 0)
        
        float_type = _patch_jax.float_type(pnt, gamma, w)
        
        if nminus.size == 0:
            return jnp.array(1, float_type)
        
        anyn0 = jnp.any(jnp.logical_and(n0, w))

        if pnt.size == 1:
            return jnp.where(anyn0, 1 - (1 - gamma) * pnt[0], 1)
    
        nout = nminus + nplus
        n = nout + n0
        Wn = jnp.sum(jnp.where(n, w, 0)) # <-- @

        if pnt.size == 2 and not debug:
            Q = 1 - (1 - gamma) * pnt[1]
            sump = Q * jnp.sum(jnp.where(n, w * nout / n, 0)) # <-- @
            return jnp.where(anyn0, 1 - pnt[0] * (1 - sump / Wn), 1)
    
        if pnt.size == 3 and not debug:
            Q = 1 - (1 - gamma) * pnt[2]
            s = w * nout / n
            S = jnp.sum(jnp.where(n, s, 0)) # <-- @
            t = w * n0 / n
            psin = jspecial.digamma(n.astype(float_type))
            def terms(nminus, nplus):
                nminus0 = nminus + n0
                Wnmod = Wn - jnp.where(nminus0, 0, w)
                frac = jnp.where(nminus0, w * nminus / nminus0, 0)
                terms1 = (S - s + frac) / Wnmod
                psi1nminus0 = jspecial.digamma((1 + nminus0).astype(float_type))
                terms2 = ((nplus - 1) * (S + t) - w * n0 * (psin - psi1nminus0)) / Wn
                return jnp.where(nplus, terms1 + terms2, 0)
            tplus = terms(nminus, nplus)
            tminus = terms(nplus, nminus)
            tall = jnp.where(n, w * (tplus + tminus) / n, 0)
            sump = (1 - pnt[1]) * S + pnt[1] * Q * jnp.sum(tall) # <-- @
            return jnp.where(anyn0, 1 - pnt[0] * (1 - sump / Wn), 1)
        
            # TODO the pnt.size == 3 calculation is probably less accurate than
            # the recursive one, see comparison limits > 30 ULP in test_bart.py
    
        p = len(nminus)

        val = (0., nminus, n0, nplus)
        def loop(i, val):
            sump, nminus, n0, nplus = val

            nminusi = nminus[i]
            n0i = n0[i]
            nplusi = nplus[i]
            ni = nminusi + n0i + nplusi
        
            val = (0., nminus, n0, nplus, i, nminusi)
            def loop(k, val):
                sumn, nminus, n0, nplus, i, nminusi = val
            
                # here I use the fact that .at[].set won't set the value if the
                # index is out of bounds
                nminus = nminus.at[jnp.where(k < nminusi, i, i + p)].set(k)
                nplus = nplus.at[jnp.where(k >= nminusi, i, i + p)].set(k - nminusi)
            
                sumn += cls._correlation(nminus, n0, nplus, pnt[1:], gamma, w, debug)
            
                nminus = nminus.at[i].set(nminusi)
                nplus = nplus.at[i].set(nplusi)
            
                return sumn, nminus, n0, nplus, i, nminusi
        
            # if ni == 0 I skip recursion by passing 0 as iteration end
            end = jnp.where(ni, nminusi + nplusi, 0)
            start = jnp.zeros_like(end)
            sumn, nminus, n0, nplus, _, _ = lax.fori_loop(start, end, loop, val)

            sump += jnp.where(ni, w[i] * sumn / ni, 0)

            return sump, nminus, n0, nplus

        # skip summation if all(n0 == 0)
        end = jnp.where(anyn0, p, 0)
        sump, _, _, _ = lax.fori_loop(0, end, loop, val)

        return jnp.where(anyn0, 1 - pnt[0] * (1 - sump / Wn), 1)

    @classmethod
    @functools.partial(jax.jit, static_argnums=(0, 7))
    def _correlation2(cls, n, ix, iy, pnt, gamma, w, debug):

        assert n.shape == ix.shape == iy.shape == w.shape
        assert n.ndim == 1 and n.size >= 0
        assert pnt.ndim == 1 and pnt.size > 0
        # TODO repeat this shape checks in BART.correlation such that the
        # error messages are user-legible

        # optimization to avoid looping over ignored axes
        n = jnp.where(w, n, 0)
        ix = jnp.where(w, ix, 0)
        iy = jnp.where(w, iy, 0)

        # convert to alternative format
        nminus = jnp.minimum(ix, iy)
        nplus = n - jnp.maximum(ix, iy)
        n0 = jnp.abs(ix - iy)
        del n, ix, iy
        
        float_type = _patch_jax.float_type(pnt, gamma, w)
        
        if nminus.size == 0:
            return jnp.array(1, float_type)
        
        anyn0 = jnp.any(jnp.logical_and(n0, w))

        if pnt.size == 1:
            return jnp.where(anyn0, 1 - (1 - gamma) * pnt[0], 1)
    
        nout = nminus + nplus
        n = nout + n0
        Wn = jnp.sum(jnp.where(n, w, 0)) # <-- @

        if pnt.size == 2 and not debug:
            Q = 1 - (1 - gamma) * pnt[1]
            sump = Q * jnp.sum(jnp.where(n, w * nout / n, 0)) # <-- @
            return jnp.where(anyn0, 1 - pnt[0] * (1 - sump / Wn), 1)
    
        if pnt.size == 3 and not debug:
            Q = 1 - (1 - gamma) * pnt[2]
            s = w * nout / n
            S = jnp.sum(jnp.where(n, s, 0)) # <-- @
            t = w * n0 / n
            psin = jspecial.digamma(n.astype(float_type))
            def terms(nminus, nplus):
                nminus0 = nminus + n0
                Wnmod = Wn - jnp.where(nminus0, 0, w)
                frac = jnp.where(nminus0, w * nminus / nminus0, 0)
                terms1 = (S - s + frac) / Wnmod
                psi1nminus0 = jspecial.digamma((1 + nminus0).astype(float_type))
                terms2 = ((nplus - 1) * (S + t) - w * n0 * (psin - psi1nminus0)) / Wn
                return jnp.where(nplus, terms1 + terms2, 0)
            tplus = terms(nminus, nplus)
            tminus = terms(nplus, nminus)
            tall = jnp.where(n, w * (tplus + tminus) / n, 0)
            sump = (1 - pnt[1]) * S + pnt[1] * Q * jnp.sum(tall) # <-- @
            return jnp.where(anyn0, 1 - pnt[0] * (1 - sump / Wn), 1)
        
            # TODO the pnt.size == 3 calculation is probably less accurate than
            # the recursive one, see comparison limits > 30 ULP in test_bart.py
    
        p = len(nminus)

        val = (0., nminus, n0, nplus)
        def loop(i, val):
            sump, nminus, n0, nplus = val

            nminusi = nminus[i]
            n0i = n0[i]
            nplusi = nplus[i]
            ni = nminusi + n0i + nplusi
        
            val = (0., nminus, n0, nplus, i, nminusi)
            def loop(k, val):
                sumn, nminus, n0, nplus, i, nminusi = val
            
                # here I use the fact that .at[].set won't set the value if the
                # index is out of bounds
                nminus = nminus.at[jnp.where(k < nminusi, i, i + p)].set(k)
                nplus = nplus.at[jnp.where(k >= nminusi, i, i + p)].set(k - nminusi)
            
                sumn += cls._correlation(nminus, n0, nplus, pnt[1:], gamma, w, debug)
            
                nminus = nminus.at[i].set(nminusi)
                nplus = nplus.at[i].set(nplusi)
            
                return sumn, nminus, n0, nplus, i, nminusi
        
            # if ni == 0 I skip recursion by passing 0 as iteration end
            end = jnp.where(ni, nminusi + nplusi, 0)
            start = jnp.zeros_like(end)
            sumn, nminus, n0, nplus, _, _ = lax.fori_loop(start, end, loop, val)

            sump += jnp.where(ni, w[i] * sumn / ni, 0)

            return sump, nminus, n0, nplus

        # skip summation if all(n0 == 0)
        end = jnp.where(anyn0, p, 0)
        sump, _, _, _ = lax.fori_loop(0, end, loop, val)

        return jnp.where(anyn0, 1 - pnt[0] * (1 - sump / Wn), 1)

    @classmethod
    @functools.partial(jnp.vectorize, excluded=(0, 7, 8), signature='(p),(p),(p),(d),(),(p)->()')
    def _correlation_vectorized(cls, nminus_or_n, n0_or_ix, nplus_or_iy, pnt, gamma, w, debug, altinput):
        func = cls._correlation2 if altinput else cls._correlation
        return func(nminus_or_n, n0_or_ix, nplus_or_iy, pnt, gamma, w, bool(debug))