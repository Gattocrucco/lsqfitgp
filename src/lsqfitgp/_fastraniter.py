# lsqfitgp/_fastraniter.py
#
# Copyright (c) 2020, 2022, 2023, 2024, Giacomo Petrillo
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

import itertools

import gvar
import numpy

from . import _linalg

# TODO support jax tracing and jax random sampling

# TODO I should have a vectorized sampler, that returns an array of samples
# right away. Copy the new gvar interface (from gvar 12.1).

def _toslice(s):
    if isinstance(s, slice):
        return s
    if isinstance(s, int):
        return slice(s, s + 1)
    raise TypeError(f'cannot convert {s!r} to slice')

def raniter(mean, cov, n=None, eps=None, rng=None):
    """
    
    Take random samples from a multivariate Gaussian.
    
    This generator mimics the interface of `gvar.raniter`, but takes as input
    the mean and covariance separately instead of a collection of gvars.
    
    Parameters
    ----------
    mean : scalar, array, or dictionary of scalars/arrays
        The mean of the Gaussian distribution.
    cov : scalar, array, or dictionary of scalars/arrays
        The covariance matrix. If ``mean`` is a dictionary, ``cov`` must be a
        dictionary with pair of keys from ``mean`` as keys.
    n : int, optional
        The maximum number of iterations. Default unlimited.
    eps : float, optional
        Used to correct the eigenvalues of the covariance matrix to handle
        non-positivity due to roundoff, relative to the largest eigenvalue.
        Default is number of variables times floating point epsilon.
    rng : seed or random generator, optional
        ``rng`` is passed through `numpy.random.default_rng` to produce a random
        number generator.
    
    Yields
    ------
    samp : scalar, array, or dictionary of scalars/arrays
        The random sample in the same format of ``mean``.
    
    Examples
    --------
    
    >>> mean = {'a': np.arange(3)}
    >>> cov = {('a', 'a'): np.eye(3)}
    >>> for sample in lgp.raniter(mean, cov, 3):
    >>>     print(sample)
    
    """

    # convert mean and cov to 1d and 2d arrays
    if hasattr(mean, 'keys'): # a dict or gvar.BufferDict
        if not hasattr(mean, 'buf'):
            mean = gvar.BufferDict(mean)
        flatmean = mean.buf
        squarecov = numpy.empty((len(flatmean), len(flatmean)))
        for k1 in mean:
            slic1 = _toslice(mean.slice(k1))
            for k2 in mean:
                slic2 = _toslice(mean.slice(k2))
                sqshape = (slic1.stop - slic1.start, slic2.stop - slic2.start)
                squarecov[slic1, slic2] = cov[k1, k2].reshape(sqshape)
    else: # an array or scalar
        mean = numpy.array(mean, copy=False)
        cov = numpy.array(cov, copy=False)
        flatmean = mean.reshape(-1)
        squarecov = cov.reshape(len(flatmean), len(flatmean))

    # decompose the covariance matrix
    try:
        covdec = _linalg.Chol(squarecov, epsrel='auto' if eps is None else eps)
    except numpy.linalg.LinAlgError:
        raise numpy.linalg.LinAlgError('covariance matrix not positive definite with eps={}'.format(eps))
        # TODO when I implement a pseudoinverse or something that does not fail
        # like diagonalization, issue a warning and use the other decomposition

    # get random number generator
    rng = numpy.random.default_rng(rng)
    
    # take samples
    iterable = itertools.count() if n is None else range(n)
    for _ in iterable:
        iidsamp = rng.standard_normal(covdec.m)
        samp = flatmean + covdec.correlate(iidsamp)

        # pack the samples with the original shape
        if hasattr(mean, 'keys'):
            samp = gvar.BufferDict(mean, buf=samp)
        else:
            samp = samp.reshape(mean.shape) if mean.shape else samp.item

        yield samp

def sample(*args, **kw):
    """
    Shortcut for ``next(raniter(..., n=1))``.
    """
    return next(raniter(*args, n=1, **kw))
