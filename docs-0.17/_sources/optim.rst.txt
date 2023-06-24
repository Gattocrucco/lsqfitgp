.. lsqfitgp/docs/optim.rst
..
.. Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
..
.. This file is part of lsqfitgp.
..
.. lsqfitgp is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, either version 3 of the License, or
.. (at your option) any later version.
..
.. lsqfitgp is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
..
.. You should have received a copy of the GNU General Public License
.. along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.

.. currentmodule:: lsqfitgp

.. _optim:

.. TODO: explain and test f32/f64

Optimization
============

Evaluating a single Gaussian process posterior
----------------------------------------------

There are three main computational steps when doing a Gaussian process fit with
:class:`GP`:

  * Compute the prior covariance matrix using the kernel. This is :math:`O((n +
    m)^2)` where :math:`n` is the number of datapoints and :math:`m` the number
    of additional points where the posterior is computed.
    
  * Decompose the prior covariance matrix. This is :math:`O(n^3)`.
  
  * Take random samples from the posterior. This is :math:`O(m^3)`.

Since usually :math:`m \gg n` because the plot is done on a finely spaced grid,
the typical bottleneck is taking the samples, i.e., calling :func:`gvar.sample`
(or :func:`gvar.raniter`). This problem can be bypassed by plotting only the
standard deviation band instead of taking samples, but it is less informative.
To make :func:`gvar.sample` faster, use its ``eps`` option: ``gvar.sample(x,
eps=1e-12)``. This forces it to use a Cholesky decomposition instead of a
diagonalization.

In general the :class:`GP` methods have options for doing everything without
:mod:`gvar`, but don't try to use all of them mindlessly before profiling the
code to know where the bottleneck actually is. Python has the module
:mod:`cProfile` for that, and in an IPython shell you can use ``%run -p``. If
you opt out of gvars, you can use :func:`lsqfitgp.raniter` to draw samples from
an explicit mean vector and covariance matrix instead of :func:`gvar.raniter`.

Once you have solved eventual :mod:`gvar`-related issues, if you have at least
some hundreds of datapoints the next bottleneck is probably in
:meth:`GP.predfromdata`. Making it faster is quick: select a solver different
from the default one when initializing the :class:`GP` object, like ``GP(kernel,
solver='chol')``. And don't forget to disable the :math:`O((n+m)^2)` positivity
check: ``GP(kernel, solver='chol', checkpos=False)``.

If you have written a custom kernel, it may become a bottleneck. For example the
letter counting kernel in :ref:`customs` was very slow. A quick way to get a 2x
improvement is computing only half of the covariance matrix: ``GP(kernel,
checksym=False, halfmatrix=True)``. Note however that in some cases this may
cause a large perfomance hit, so by default the full covariance matrix is
computed even if ``checksym=False`` (but the cross covariance matrices are not
computed twice).

The JAX compiler
----------------

Since :mod:`lsqfitgp` uses JAX as computational backend, which provides a just
in time compiler (JIT), in many cases a piece of code doing stuff with a
Gaussian process can be put into a function and compiled to low-level
instructions with ``jax.jit``, provided all the array operations are
implemented with ``jax.numpy`` instead of ``numpy``, and gvars are avoided.
Example::

    import jax
    from jax import numpy as jnp
    import lsqfitgp as lgp
    
    def doinference(data, **options):
        x = jnp.linspace(0, 10, len(data))
        xplot = jnp.linspace(0, 10, 100)
        gp = lgp.GP(lgp.ExpQuad(), **options)
        gp.addx(x, 'data')
        gp.addx(xplot, 'plot')
        yplot_mean, yplot_cov = gp.predfromdata({'data': data}, 'plot', raw=True)
        # notice we use raw=True to return mean and covariance separately
        # instead of implicitly tracked into gvars
        yplot_sdev = jnp.sqrt(jnp.diag(yplot_cov))
        return yplot_mean, yplot_sdev
    
    doinference_compiled = jax.jit(doinference, static_argnames=['solver', 'checkpos', 'checksym'])
    # static_argnames indicates the function parameters that are not numerical
    # and should not be dealt with by the compiler, I've put some I will use
    # later
    
    import timeit
    
    def benchmark(func, *args, **kwargs):
        timer = timeit.Timer('func(*args, **kwargs)', globals=locals())
        n, _ = timer.autorange()
        times = timer.repeat(5, n)
        time = min(times) / n
        print(f'{func.__name__} took {time * 1e3:7.3f} ms on average')
    
    data = jnp.zeros(10)
    benchmark(doinference, data)
    benchmark(doinference_compiled, data)

And the winner is::

   doinference took   5.651 ms on average
   doinference took   0.074 ms on average

The compiled version is 70 times faster. The difference is so stark because we
used only 10 datapoints, so most of the time is spent in routing overhead
instead of actual computations. Repeating with 1000 datapoints, the advantage
should be limited::

    data = jnp.zeros(1000)
    benchmark(doinference, data)
    benchmark(doinference_compiled, data)

Indeed::

   doinference took 356.814 ms on average
   doinference took 187.924 ms on average

With many datapoints we said that changing the :class:`GP` options is the
important tweak. Let's check::

    kw = dict(solver='chol', checkpos=False, checksym=False)
    benchmark(doinference, data, **kw)
    benchmark(doinference_compiled, data, **kw)

Result::

   doinference took  54.725 ms on average
   doinference took  16.106 ms on average

As expected.

Fitting hyperparameters
-----------------------

The function :class:`empbayes_fit` finds the "optimal" hyperparameters by
feeding the GP-factory you give to it into a minimization routine that tries to
change the hyperparameters one step at a time and each time recreates the GP
object and does some computations to check how a "good fit" it is for the given
data.

From the point of view of computational efficiency this means that, apart from
taking posterior samples, the other techniques explained in the previous
sections also apply here. In particular, when the number of datapoints `n`
starts to be in the hundreds, to speed up the fit do::

   GP(kernel, solver='chol', checkpos=False)

when you create the :class:`GP` object in the factory function.
:class:`empbayes_fit` applies the jit for you if passed the ``jit=True``
option, so you don't have to deal with it yourself.
