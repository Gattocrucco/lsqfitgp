.. currentmodule:: lsqfitgp

.. _optim:

Optimization
============

There are three main computational steps when doing a gaussian process fit with
:mod:`lsqfitgp`:

  * Compute the prior covariance matrix using the kernel. This is :math:`O((n +
    m)^2)` where `n` is the number of datapoints and `m` the number of
    additional points where the posterior is computed.
    
  * Solve a linear system of size `n`. This is :math:`O(n^3)`.
  
  * Take random samples from the posterior. This is :math:`O(m^3 + s)` where
    `s` is the number of samples taken.

Since usually :math:`m \gg n` because the plot is done on a finely spaced grid,
the typical bottleneck is taking the samples, i.e. calling
:func:`gvar.raniter`. This problem can be bypassed by plotting only the
standard deviation band instead of taking samples, but it is less informative.

I'm working with :mod:`gvar`'s author to make :func:`gvar.raniter` more
efficient; in the meantime, you can use this quick replacement which is a bit
faster, although not as robust::

    import gvar
    from scipy import linalg
    import numpy as np
    import itertools
    
    def fastraniter(g, n=None):
    
        # convert g to a 1-dimensional array flatg
        if hasattr(g, 'buf'): # a gvar.BufferDict
            flatg = g.buf
        elif hasattr(g, 'keys'): # a dict
            g = gvar.BufferDict(g)
            flatg = g.buf
        else: # an array or scalar
            g = np.array(g, copy=False)
            flatg = g.reshape(-1)
        
        # compute the covariance matrix
        cov = gvar.evalcov(flatg)
        
        # compute the cholesky decomposition
        maxeigv_bound = np.max(np.sum(np.abs(cov), axis=-1))
        eps = np.finfo(float).eps
        cov[np.diag_indices(len(cov))] += len(cov) * eps * maxeigv_bound
        L = linalg.cholesky(cov, lower=True)
        
        # take samples
        iterable = itertools.count() if n is None else range(n)
        for _ in iterable:
            iidsamp = np.random.randn(len(cov))
            samp = L @ iidsamp
            
            # pack the samples with the original shape
            if hasattr(g, 'keys'):
                samp = gvar.BufferDict(g, buf=samp)
            else:
                samp = samp.reshape(g.shape) if g.shape else samp.item
            
            yield samp

Let's benchmark it::

    # make 100 correlated variables
    n = 100
    a, b = np.random.randn(2, n, n)
    cov = a.T @ a
    mean = np.random.randn(n)
    x = b @ gvar.gvar(mean, cov)
    
    import time
    
    start = time.time()
    for _ in range(10):
        next(gvar.raniter(x))
    end = time.time()
    print('gvar.raniter: {:.3g} s'.format(end - start))
    
    start = time.time()
    for _ in range(10):
        next(fastraniter(x))
    end = time.time()
    print('fastraniter: {:.3g} s'.format(end - start))

Output::

   gvar.raniter: 0.64 s
   fastraniter: 0.244 s

If this is not enough, you need to bypass :mod:`gvar` altogether using the
`raw` option of :meth:`GP.predfromdata`, which will make it return separately
the posterior mean and covariance matrix. Here's a version of `fastraniter` for
taking samples from the mean and covariance directly::

    def fastraniter_nogvar(mean, cov, n=None):
    
        # convert mean and cov to 1d and 2d arrays
        if hasattr(mean, 'keys'): # a dict or gvar.BufferDict
            if not hasattr(mean, 'buf'):
                mean = gvar.BufferDict(mean)
            flatmean = mean.buf
            squarecov = np.empty((len(flatmean), len(flatmean)))
            for k1 in mean:
                slic1 = mean.slice(k1)
                for k2 in mean:
                    slic2 = mean.slice(k2)
                    squarecov[slic1, slic2] = cov[k1, k2]
        else: # an array or scalar
            mean = np.array(mean, copy=False)
            cov = np.array(cov, copy=False)
            flatmean = mean.reshape(-1)
            squarecov = cov.reshape(len(flatmean), len(flatmean))
        
        # compute the cholesky decomposition
        maxeigv_bound = np.max(np.sum(np.abs(squarecov), axis=-1))
        eps = np.finfo(float).eps
        squarecov[np.diag_indices(len(squarecov))] += len(squarecov) * eps * maxeigv_bound
        L = linalg.cholesky(squarecov, lower=True)
        
        # take samples
        iterable = itertools.count() if n is None else range(n)
        for _ in iterable:
            iidsamp = np.random.randn(len(squarecov))
            samp = L @ iidsamp
            
            # pack the samples with the original shape
            if hasattr(mean, 'keys'):
                samp = gvar.BufferDict(mean, buf=samp)
            else:
                samp = samp.reshape(mean.shape) if mean.shape else samp.item
            
            yield samp

Benchmark with the mean and covariance of the 100 variables from above::

    mean = b @ mean
    cov = b @ cov @ b.T
    
    start = time.time()
    for _ in range(10):
        next(fastraniter_nogvar(mean, cov))
    end = time.time()
    print('fastraniter_nogvar: {:.3g} s'.format(end - start))

Output::

   fastraniter_nogvar: 0.0137 s

So it is 40x faster than :func:`gvar.raniter`. In general the :class:`GP`
methods have options for doing everything without :mod:`gvar`, but don't try to
use all of them mindlessly before profiling the code to know where the
bottleneck actually is. Python has the module :mod:`profile` for that, and in
an IPython shell you can use ``%run -p``.

Once you have solved eventual :mod:`gvar`-related issues, if you have at least
some hundreds of datapoints the next bottleneck is probably in
:meth:`GP.predfromdata`. Making it faster is quick: select a solver different
from the default one when initializing the :class:`GP` object, like
``GP(kernel, solver='gersh')``. This applies also when using
:func:`empbayes_fit`. And don't forget to disable the positivity check:
``GP(kernel, solver='gersh', checkpos=False)``.

Finally, if you have written a custom kernel, it may become a bottleneck. For
example the letter counting kernel in :ref:`customs` was very slow. A quick
way to get a 2x improvement is disabling the symmetry check in :class:`GP`:
``GP(kernel, checksym=False)``.