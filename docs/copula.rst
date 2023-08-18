.. file generated automatically by lsqfitgp/docs/copula.py

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

.. autoclass:: beta(alpha, beta)
.. autoclass:: dirichlet(alpha, n)
.. autoclass:: gamma(alpha, beta)
.. autoclass:: halfcauchy(gamma)
.. autoclass:: halfnorm(sigma)
.. autoclass:: invgamma(alpha, beta)
.. autoclass:: loggamma(c)
.. autoclass:: uniform(a, b)
