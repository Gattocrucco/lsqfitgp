.. lsqfitgp/docs/reference/bayestree.rst
..
.. Copyright (c) 2023, Giacomo Petrillo
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

.. module:: lsqfitgp.bayestree

BART
====

The `bayestree` submodule contains classes to set up a Gaussian process
regression with the `~lsqfitgp.BART` kernel. See the :ref:`bart <bart>`,
:ref:`barteasy <barteasy>` and :ref:`acic <acic>` examples.

The original version of these models is implemented in the R packages `BART
<https://cran.r-project.org/package=BART>`_, `dbarts
<https://cran.r-project.org/package=dbarts>`_, and `bcf
<https://cran.r-project.org/package=bcf>`_. Pros of the GP version provided
here:

  * No MCMC, the result is computed quite directly.

  * Fast inference on the hyperparameters.

  * Allows a wider region of hyperparameter space.

Cons:

  * Does not scale to large datasets.

  * Worse predictive performance at fixed hyperparameters.

  * Slower at fixed hyperparameters.

  * May use more memory.

Overall, this version should tipycally be a better choice than the R packages if
you have at most 10000 datapoints.

Index
-----
`bart`, `bcf`

Documentation
-------------

.. autoclass:: lsqfitgp.bayestree.bart
    :members:

.. autoclass:: lsqfitgp.bayestree.bcf
    :members:
