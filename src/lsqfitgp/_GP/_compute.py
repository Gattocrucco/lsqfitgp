# lsqfitgp/_GP/_compute.py
#
# Copyright (c) 2020, 2022, 2023, 2025, Giacomo Petrillo
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
import math

from jax import numpy as jnp
import numpy
import gvar

from .. import _linalg
from .. import _jaxext

from . import _base

class GPCompute(_base.GPBase):

    def __init__(self, *, solver, solverkw):
        self._decompcache = {} # tuple of keys -> Decomposition
        decomp = self._getdecomp(solver)
        self._decompclass = lambda K, **kwargs: decomp(K, **kwargs, **solverkw)

    def _clone(self):
        newself = super()._clone()
        newself._decompcache = self._decompcache.copy()
        newself._decompclass = self._decompclass
        return newself

    def _solver(self, keys, ycov=None, *, covtransf=None, **kw):
        """
        Return a decomposition of the covariance matrix of the keys in ``keys``
        plus the matrix ycov. Keyword arguments are passed to the decomposition.
        """
        
        # TODO cache ignores **kw.

        keys = tuple(keys)
        
        # Check if decomposition is in cache.
        if ycov is None:
            cache = self._decompcache.get(keys)
            if cache is not None:
                return cache
            # TODO use frozenset(keys) instead of tuple(keys) to make cache
            # work when order changes, but I have to permute the decomposition
            # to make that work. Needs an ad-hoc class in _linalg. Make the
            # decompcache a dict subclass that accepts tuples of keys but uses
            # internally frozenset.
        
        # Compute decomposition. # woodbury, currently un-implemented
        # if isinstance(ycov, _linalg.Decomposition):
        #     ancestors = []
        #     transfs = []
        #     for key in keys:
        #         elem = self._elements[key]
        #         nest = False
        #         if isinstance(elem, self._LinTransf):
        #             size = sum(self._elements[k].size for k in elem.keys)
        #             if size < elem.size:
        #                 nest = True
        #                 ancestors += list(elem.keys)
        #                 transfs.append(jnp.concatenate(elem.matrices(self), 1))
        #         if not nest:
        #             ancestors.append(key)
        #             transfs.append(jnp.eye(elem.size))
        #     transf = jlinalg.block_diag(*transfs)
        #     cov = self._assemblecovblocks(ancestors)
        #     if covtransf:
        #         ycov, transf, cov = covtransf((ycov, transf, cov))
        #     covdec = self._decompclass(cov, **kw)
        #     # TODO obtain covdec from _solver recursively, to use cache?
        #     decomp = _linalg.Woodbury2(ycov, transf, covdec, self._decompclass, sign=1, **kw)
        # else:
        Kxx = self._assemblecovblocks(keys)
        if ycov is not None:
            Kxx = Kxx + ycov
        if covtransf:
            Kxx = covtransf(Kxx)
        decomp = self._decompclass(Kxx, **kw)
        
        # Cache decomposition.
        if ycov is None:
            self._decompcache[keys] = decomp
        
        return decomp
        
    def _flatgiven(self, given, givencov):
        
        if not hasattr(given, 'keys'):
            raise TypeError('`given` must be dict')
        gcblack = givencov is None or isinstance(givencov, _linalg.Decomposition)
        if not gcblack and not hasattr(givencov, 'keys'):
            raise TypeError('`givenconv` must be None, dict or Decomposition')
        
        ylist = []
        keylist = []
        for key, l in given.items():
            if key not in self._elements:
                raise KeyError(key)
            
            if not isinstance(l, jnp.ndarray):
                # use numpy since there could be gvars
                l = numpy.asarray(l)
            shape = self._elements[key].shape
            if l.shape != shape:
                msg = 'given[{!r}] has shape {!r} different from shape {!r}'
                raise ValueError(msg.format(key, l.shape, shape))
            if l.dtype != object and not jnp.issubdtype(l.dtype, jnp.number):
                msg = 'given[{!r}] has non-numerical dtype {!r}'
                raise TypeError(msg.format(key, l.dtype))
            
            ylist.append(l.reshape(-1))
            keylist.append(key)
        
        # TODO error checking on the unpacking of givencov
        
        if gcblack:
            covblocks = givencov
        else:
            covblocks = [
                [
                    jnp.asarray(givencov[keylist[i], keylist[j]]).reshape(ylist[i].shape + ylist[j].shape)
                    for j in range(len(keylist))
                ]
                for i in range(len(keylist))
            ]
            
        return ylist, keylist, covblocks
    
    def pred(self, given, key=None, givencov=None, *, fromdata=None, raw=False, keepcorr=None):
        """
        
        Compute the posterior.
        
        The posterior can be computed either for all points or for a subset,
        and either directly from data or from a posterior obtained with a fit.
        The latter case is for when the Gaussian process was used in a fit with
        other parameters.
        
        The output is a collection of gvars, either an array or a dictionary
        of arrays. They are properly correlated with gvars returned by
        `prior` and with the input data/fit.
        
        The input is a dictionary of arrays, ``given``, with keys corresponding
        to the keys in the GP as added by `addx` or `addtransf`.
        
        Parameters
        ----------
        given : dictionary of arrays
            The data or fit result for some/all of the points in the GP.
            The arrays can contain either gvars or normal numbers, the latter
            being equivalent to zero-uncertainty gvars.
        key : None, key or list of keys, optional
            If None, compute the posterior for all points in the GP (also those
            used in ``given``). Otherwise only those specified by key.
        givencov : dictionary of arrays, optional
            Covariance matrix of ``given``. If not specified, the covariance
            is extracted from ``given`` with ``gvar.evalcov(given)``.
        fromdata : bool
            Mandatory. Specify if the contents of ``given`` are data or already
            a posterior.
        raw : bool, optional
            If True, instead of returning a collection of gvars, return
            the mean and the covariance. When the mean is a dictionary, the
            covariance is a dictionary whose keys are pairs of keys of the
            mean (the same format used by `gvar.evalcov`). Default False.
        keepcorr : bool, optional
            If True (default), the returned gvars are correlated with the
            prior and the data/fit. If False, they have the correct covariance
            between themselves, but are independent from all other preexisting
            gvars.
        
        Returns
        -------
        If raw=False (default):
        
        posterior : array or dictionary of arrays
            A collections of gvars representing the posterior.
        
        If raw=True:
        
        pmean : array or dictionary of arrays
            The mean of the posterior. Equivalent to ``gvar.mean(posterior)``.
        pcov : 2D array or dictionary of 2D arrays
            The covariance matrix of the posterior. If ``pmean`` is a
            dictionary, the keys of ``pcov`` are pairs of keys of ``pmean``.
            Equivalent to ``gvar.evalcov(posterior)``.
        
        """

        # TODO GP.pred(..., raw=True, onlyvariance=True) computes only the
        # variance (requires actually implementing diagquad at least in Chol and
        # Diag).

        
        if fromdata is None:
            raise ValueError('you must specify if `given` is data or fit result')
        fromdata = bool(fromdata)
        raw = bool(raw)
        if keepcorr is None:
            keepcorr = not raw
        if keepcorr and raw:
            raise ValueError('both keepcorr=True and raw=True')
        
        strip = False
        if key is None:
            outkeys = list(self._elements)
        elif isinstance(key, list):
            outkeys = key
        else:
            outkeys = [key]
            strip = True
        outslices = self._slices(outkeys)
                
        ylist, inkeys, ycovblocks = self._flatgiven(given, givencov)
        y = self._concatenate(ylist)
        if y.dtype == object:
            if ycovblocks is not None:
                raise ValueError('given may contain gvars but a separate covariance matrix has been provided')
        
        self._checkpos_keys(inkeys + outkeys)
        
        Kxxs = self._assemblecovblocks(inkeys, outkeys)
        
        # if isinstance(ycovblocks, _linalg.Decomposition): # woodbury, currently un-implemented
        #     ycov = ycovblocks
        # elif ...
        if ycovblocks is not None:
            ycov = jnp.block(ycovblocks)
        elif (fromdata or raw or not keepcorr) and y.dtype == object:
            ycov = gvar.evalcov(gvar.gvar(y))
            # TODO use evalcov_blocks
            # TODO I think this ignores the case in which we are using gvars
            # and they are correlated with the GP. I guess the correct thing
            # would be to sum the data gvars to the prior ones and use the
            # resulting covariance matrix, and write a note about possible
            # different results in this case when switching raw or keepcorr.
        else:
            ycov = None
        self._check_ycov(ycov)
        
        if raw or not keepcorr or self._checkfinite:
            if y.dtype == object:
                ymean = gvar.mean(y)
            else:
                ymean = y
            self._check_ymean(ymean)
        
        if raw or not keepcorr:
            
            Kxsxs = self._assemblecovblocks(outkeys)
            
            if fromdata:
                solver = self._solver(inkeys, ycov)
            else:
                solver = self._solver(inkeys)

            mean = solver.pinv_bilinear(Kxxs, ymean)
            cov = Kxsxs - solver.ginv_quad(Kxxs)
            
            if not fromdata:
                # cov = Kxsxs - Kxsx Kxx^-1 (Kxx - ycov) Kxx^-1 Kxxs =
                #     = Kxsxs - Kxsx Kxx^-1 Kxxs + Kxsx Kxx^-1 ycov Kxx^-1 Kxxs
                if ycov is not None:
                    # if isinstance(ycov, _linalg.Decomposition): # for woodbury, currently un-implemented
                    #     ycov = ycov.matrix()
                    A = solver.ginv_linear(Kxxs)
                    # TODO do I need K⁺ here or is K⁻ fine?
                    cov += A.T @ ycov @ A
            
        else: # (keepcorr and not raw)        
            yplist = [numpy.reshape(self._prior(key), -1) for key in inkeys]
            ysplist = [numpy.reshape(self._prior(key), -1) for key in outkeys]
            yp = self._concatenate(yplist)
            ysp = self._concatenate(ysplist)
            
            if y.dtype != object and ycov is not None:
                # if isinstance(ycov, _linalg.Decomposition): # for woodbury, currently un-implemented
                #     ycov = ycov.matrix()
                y = gvar.gvar(y, ycov)
            else:
                y = numpy.asarray(y) # because y - yp fails if y is a jax array
            mat = ycov if fromdata else None
            flatout = ysp + self._solver(inkeys, mat).pinv_bilinear_robj(Kxxs, y - yp)
        
        if raw and not strip:
            meandict = {
                key: mean[slic].reshape(self._elements[key].shape)
                for key, slic in zip(outkeys, outslices)
            }
            
            covdict = {
                (row, col):
                cov[rowslice, colslice].reshape(self._elements[row].shape + self._elements[col].shape)
                for row, rowslice in zip(outkeys, outslices)
                for col, colslice in zip(outkeys, outslices)
            }
            
            return meandict, covdict
            
        elif raw:
            outkey, = outkeys
            mean = mean.reshape(self._elements[outkey].shape)
            cov = cov.reshape(2 * self._elements[outkey].shape)
            return mean, cov
        
        elif not keepcorr:
                        
            ##### temporary fix for gplepage/gvar#49 #####
            cov = numpy.array(cov)
            ##############################################
            
            flatout = gvar.gvar(mean, cov, fast=True)
        
        if not strip:
            return {
                key: flatout[slic].reshape(self._elements[key].shape)
                for key, slic in zip(outkeys, outslices)
            }
        else:
            outkey, = outkeys
            return flatout.reshape(self._elements[outkey].shape)
        
    def predfromfit(self, *args, **kw):
        """
        Like `pred` with ``fromdata=False``.
        """
        return self.pred(*args, fromdata=False, **kw)
    
    def predfromdata(self, *args, **kw):
        """
        Like `pred` with ``fromdata=True``.
        """
        return self.pred(*args, fromdata=True, **kw)
    
    def _prior_decomp(self, given, givencov=None, **kw):
        """ Internal implementation of marginal_likelihood. Keyword arguments
        are passed to _solver. """
        ylist, inkeys, ycovblocks = self._flatgiven(given, givencov)
        y = self._concatenate(ylist)
        
        self._checkpos_keys(inkeys)
        
        # Get mean.
        if y.dtype == object:
            ymean = gvar.mean(y)
        else:
            ymean = y
        self._check_ymean(ymean)
        
        # Get covariance matrix.
        # if isinstance(ycovblocks, _linalg.Decomposition): # for woodbury, currently un-implemented
        #     ycov = ycovblocks
        # elif ...
        if ycovblocks is not None:
            ycov = jnp.block(ycovblocks)
            if y.dtype == object:
                warnings.warn(f'covariance matrix may have been specified both explicitly and with gvars; the explicit one will be used')
        elif y.dtype == object:
            gvary = gvar.gvar(y)
            ycov = gvar.evalcov(gvary)
        else:
            ycov = None
        self._check_ycov(ycov)
        
        decomp = self._solver(inkeys, ycov, **kw)
        return decomp, ymean
    
    def _check_ymean(self, ymean):
        with _jaxext.skipifabstract():
            if self._checkfinite and not jnp.all(jnp.isfinite(ymean)):
                raise ValueError('mean of `given` is not finite')
    
    def _check_ycov(self, ycov):
        if ycov is None or isinstance(ycov, _linalg.Decomposition):
            return
        with _jaxext.skipifabstract():
            if self._checkfinite and not jnp.all(jnp.isfinite(ycov)):
                raise ValueError('covariance matrix of `given` is not finite')
            if self._checksym and not jnp.allclose(ycov, ycov.T):
                raise ValueError('covariance matrix of `given` is not symmetric')

    def marginal_likelihood(self, given, givencov=None, **kw):
        """
        
        Compute the logarithm of the probability of the data.
        
        The probability is computed under the Gaussian prior and Gaussian error
        model. It is also called marginal likelihood. If :math:`y` is the data
        and :math:`g` is the Gaussian process, this is
        
        .. math::
            \\log \\int p(y|g) p(g) \\mathrm{d} g.
        
        Unlike `pred`, you can't compute this with a fit result instead of
        data. If you used the Gaussian process as latent variable in a fit,
        use the whole fit to compute the marginal likelihood. E.g. `lsqfit`
        always computes the logGBF (it's the same thing).
        
        The input is an array or dictionary of arrays, ``given``. The contents
        of ``given`` represent the input data.
                
        Parameters
        ----------
        given : dictionary of arrays
            The data for some/all of the points in the GP. The arrays can
            contain either gvars or normal numbers, the latter being
            equivalent to zero-uncertainty gvars.
        givencov : dictionary of arrays, optional
            Covariance matrix of ``given``. If not specified, the covariance
            is extracted from ``given`` with ``gvar.evalcov(given)``.
        **kw :
            Additional keyword arguments are passed to the matrix decomposition.
        
        Returns
        -------
        logp : scalar
            The logarithm of the marginal likelihood.
        """
        decomp, ymean = self._prior_decomp(given, givencov, **kw)
        mll, _, _, _, _ = decomp.minus_log_normal_density(ymean, value=True)
        return -mll
    
    @staticmethod
    def _getdecomp(solver):
        return {
            'chol': _linalg.Chol,
        }[solver]
    
    @classmethod
    def decompose(cls, posdefmatrix, solver='chol', **kw):
        """
        Decompose a nonnegative definite matrix.
        
        The decomposition can be used to calculate linear algebra expressions
        where the (pseudo)inverse of the matrix appears.
        
        Parameters
        ----------
        posdefmatrix : array
            A nonnegative definite nonempty symmetric square matrix. If the
            array is not square, it must have a shape of the kind (k, n, m,
            ..., k, n, m, ...) and is reshaped to (k * n * m * ..., k * n * m *
            ...).
        solver : str
            Algorithm used to decompose the matrix.

            'chol'
                Cholesky decomposition after regularizing the matrix with a
                Gershgorin estimate of the maximum eigenvalue.
        **kw :
            Additional options.

            epsrel, epsabs : positive float or 'auto'
                Specify the threshold for considering small the eigenvalues:
                
                    eps = epsrel * maximum_eigenvalue + epsabs
                
                epsrel='auto' sets epsrel = matrix_size * float_epsilon, 
                while epsabs='auto' sets epsabs = float_epsilon. Default is
                epsrel='auto', epsabs=0.
        
        Returns
        -------
        decomp : Decomposition
            An object representing the decomposition of the matrix. The
            available methods and properties are (K being the matrix):
        
            matrix():
                Return K.
            ginv():
                Compute K⁻.
            ginv_linear(X):
                Compute K⁻X.
            pinv_bilinear(A, r)
                Compute A'K⁺r.
            pinv_bilinear_robj(A, r)
                Compute A'K⁺r, and r can be an array of arbitrary objects.
            ginv_quad(A)
                Compute A'K⁻A.
            ginv_diagquad(A)
                Compute diag(A'K⁻A).
            correlate(x)
                Compute Zx such that K = ZZ', Z can be rectangular.
            back_correlate(X)
                Compute Z'X.
            pinv_correlate(x):
                Compute Z⁺x.
            minus_log_normal_density(r, ...)
                Compute a Normal density and its derivatives.
            eps
                The threshold below which eigenvalues are not calculable.
            n
                Number of rows/columns of K.
            m
                Number of columns of Z.
        
        Notes
        -----
        The decomposition operations are JAX-traceable, but they are not meant
        to be differentiated. The method `minus_log_normal_density` provides
        required derivatives with a custom implementation, given the derivatives
        of the inputs.
        
        """
        m = jnp.asarray(posdefmatrix)
        assert m.size > 0
        assert m.ndim % 2 == 0
        half = m.ndim // 2
        head = m.shape[:half]
        tail = m.shape[half:]
        assert head == tail
        n = math.prod(head)
        m = m.reshape(n, n)
        decompcls = cls._getdecomp(solver)
        return decompcls(m, **kw)
        
        # TODO extend the interface to use composite decompositions
        # TODO accept a dict for covariance matrix
