.. lsqfitgp/docs/copula.rst
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

.. module:: lsqfitgp.copula

Gaussian copulas
================

The `copula` submodule provides classes to define probability distributions and
parametrize them such that the joint distribution of the parameters is Normal.
This is useful to define priors for the hyperparameters of a Gaussian process,
but can also be used on its own.

To define a variable, use one of the subclasses of `Distr` listed :ref:`below
<families>`. Combine the variables together by using them as parameters to other
variables, or by putting them in a `gvar.BufferDict`. See `Distr` for details.

..  note::
    
    I define "Gaussian copula" to mean a representation of an arbitrary random
    variable as the transformation of a multivariate Normal random variable, as
    explained, e.g., `here
    <https://blogs.sas.com/content/iml/2021/07/05/introduction-copulas.html>`_.
    This is different from another common usage, which is representing a
    bivariate Normal as the transformation of uniform variables, as `"Gaussian
    copula" on wikipedia
    <https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula>`_.

.. autoclass:: Distr
    :members:

.. _families:

Predefined families
-------------------

.. autoclass:: beta
.. autoclass:: dirichlet
.. autoclass:: halfcauchy
.. autoclass:: halfnorm
.. autoclass:: invgamma
.. autoclass:: loggamma
.. autoclass:: uniform

.. TODO generate this file automatically
