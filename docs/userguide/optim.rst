.. lsqfitgp/docs/optim.rst
..
.. Copyright (c) 2020, 2022, 2023, 2024, Giacomo Petrillo
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

Additionally, by default `GP` checks that the prior covariance matrix is
symmetric and positive semidefinite. This has complexity :math:`O((n + m)^2)`,
and the positivity check is somewhat slow. To disable the check, write ``GP(...,
checkpos=False, checksym=False)``. The check is disabled anyway when using the
jax compiler, addressed in the next section.

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

.. .. Uncomment and adapt when I reintroduce different solvers
.. Once you have solved eventual :mod:`gvar`-related issues, if you have at least
.. some hundreds of datapoints the next bottleneck is probably in
.. :meth:`GP.predfromdata`. Making it faster is quick: select a solver different
.. from the default one when initializing the :class:`GP` object, like ``GP(kernel,
.. solver='chol')``. And don't forget to disable the :math:`O((n+m)^2)` positivity
.. check: ``GP(kernel, solver='chol', checkpos=False)``.

If you have written a custom kernel, it may become a bottleneck. For example the
letter counting kernel in :ref:`customs` was very slow. A quick way to get a 2x
improvement is computing only half of the covariance matrix: ``GP(kernel,
checksym=False, halfmatrix=True)``. Note however that in some cases this may
cause a large perfomance hit (for example in the `BART` kernel), so by default
the full covariance matrix is computed even if ``checksym=False`` (but the cross
covariance matrices are not computed twice).

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
        yplot_mean, yplot_cov = (lgp
            .GP(lgp.ExpQuad(), **options)
            .addx(x, 'data')
            .addx(xplot, 'plot')
            .predfromdata({'data': data}, 'plot', raw=True)
        )
        # we use raw=True to return mean and covariance separately
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

And the winner is:

.. code-block:: text

    doinference took  10.591 ms on average
    doinference took   0.023 ms on average

The compiled version is 400 times faster. The difference is so stark because we
used only 10 datapoints, so most of the time is spent in routing overhead
instead of actual computations. Repeating with 1000 datapoints, the advantage
should be milder::

    data = jnp.zeros(1000)
    benchmark(doinference, data)
    benchmark(doinference_compiled, data)

Indeed, it's 6x faster, lower but still high:

.. code-block:: text

    doinference took  30.618 ms on average
    doinference took   5.346 ms on average

We said that using the :class:`GP` options ``checkpos=False, checksym=False``
makes it faster, and that they are disabled anyway under jit. Let's check::

    benchmark(doinference, data, checkpos=False, checksym=False)
    benchmark(doinference_compiled, data, checkpos=False, checksym=False)

Result:

.. code-block:: text

    doinference took  19.312 ms on average
    doinference took   6.120 ms on average

As expected, the compiled version is not affected, while the original one gains
a lot of speed: now the advantage is just 3x.

Fitting hyperparameters
-----------------------

The function :class:`empbayes_fit` finds the "optimal" hyperparameters by
feeding the GP-factory you give to it into a minimization routine that tries to
change the hyperparameters one step at a time and each time recreates the GP
object and does some computations to check how a "good fit" it is for the given
data.

From the point of view of computational efficiency this means that, apart from
taking posterior samples, the techniques explained in the previous sections also
apply here. However, :class:`empbayes_fit` applies the jit for you by default,
so you don't have to deal with this yourself. If you disable the jit for some
reason, use the options::

   GP(..., checksym=False, checkpos=False)

Another way to improve the performance is by tweaking the minimization method.
The main setting is the ``method`` argument, which picks a sensible preset for
the underlying routine `scipy.optimize.minimize`. Then, additional
configurations can be specified through the ``minkw`` argument; to use it, it
may be useful to look at the full argument list passed to `minimize`, which is
provided after the fit in the attribute ``minargs``.

Floating point 32 vs. 64 bit
----------------------------

`jax` by default uses the `float32` data type for all floating point arrays and
calculations. Upon initialization, `lsqfitgp` configures `jax` to use `float64`
instead, like `numpy`. Although operations with 32 bit floats are about twice as
fast, Gaussian process regression is particularly sensitive to numerical
accuracy. You can reset `jax`'s default with:

.. code-block::

    jax.config.update('jax_enable_x64', False)

to get a speedup, but this will likely give problems when the number of
datapoints is over 1000, and will break `empbayes_fit` unless you make an effort
to tune the minimizer parameters to make it work at `float32` precision.

.. TODO I expect that using many structured fields under jit slows down things
.. compared to a single array field due to unrolling. Check this and if so make
.. a section on it.
