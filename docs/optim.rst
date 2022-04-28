.. lsqfitgp/docs/optim.rst
..
.. Copyright (c) 2020, 2022, Giacomo Petrillo
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

Optimization
============

There are three main computational steps when doing a Gaussian process fit with
:mod:`lsqfitgp`:

  * Compute the prior covariance matrix using the kernel. This is :math:`O((n +
    m)^2)` where `n` is the number of datapoints and `m` the number of
    additional points where the posterior is computed.
    
  * Solve a linear system of size `n`. This is :math:`O(n^3)`.
  
  * Take random samples from the posterior. This is :math:`O(m^3 + s)` where
    `s` is the number of samples taken.

Since usually :math:`m \gg n` because the plot is done on a finely spaced grid,
the typical bottleneck is taking the samples, i.e., calling
:func:`gvar.raniter`. This problem can be bypassed by plotting only the
standard deviation band instead of taking samples, but it is less informative.
To make :func:`gvar.raniter` faster, use its ``eps`` option: ``gvar.raniter(x,
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
from the default one when initializing the :class:`GP` object, like
``GP(kernel, solver='gersh')``. This applies also when using
:class:`empbayes_fit`. And don't forget to disable the positivity check:
``GP(kernel, solver='gersh', checkpos=False)``.

Finally, if you have written a custom kernel, it may become a bottleneck. For
example the letter counting kernel in :ref:`customs` was very slow. A quick
way to get a 2x improvement is disabling the symmetry check in :class:`GP`:
``GP(kernel, checksym=False)``.
