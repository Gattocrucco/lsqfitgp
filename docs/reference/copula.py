# lsqfitgp/docs/reference/copula.py
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

""" Generate the reference page for the copula module """

import inspect
import pathlib

import numpy as np
from lsqfitgp import copula

distrs = []
for name, obj in vars(copula).items():
    if inspect.isclass(obj) and issubclass(obj, copula.Distr) and obj is not copula.Distr:
        distrs.append(name)

out = """\
.. file generated automatically by lsqfitgp/docs/copula.py

.. module:: lsqfitgp.copula

Gaussian copulas
================

The `copula` submodule provides classes to define probability distributions and
reparametrize them such that the joint distribution is Normal. This is useful to
define priors for the hyperparameters of a Gaussian process, but can also be
used on its own.

To define a variable, use one of the subclasses of `Distr` listed under
:ref:`families`. Combine the variables together to define a model by putting
them in a `Copula` object. See `Distr` for examples.

To represent at once concrete values of the variables and their transformed
parametrization, put them in a `gvar.BufferDict` using `makedict`. To apply
the transformation manually, use `~DistrBase.partial_invfcn`.

..  note::
    
    I define "Gaussian copula" to mean a representation of an arbitrary random
    variable as the transformation of a multivariate Normal random variable, as
    explained, e.g., `here
    <https://blogs.sas.com/content/iml/2021/07/05/introduction-copulas.html>`_.
    This is different from another common usage, which is representing a
    bivariate Normal as the transformation of uniform variables, as `"Gaussian
    copula" on wikipedia
    <https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula>`_.

Generic classes
---------------

The class `DistrBase` defines basic functionality shared by `Distr` and
`Copula`, and can not be instantiated. `Distr` represents a probability
distribution on a numerical tensor; it can not be instantied, use its concrete
subclasses :ref:`listed below <families>`. `Copula` represents a collection of
possibly related `Distr` objects and is intended for direct use.

.. autoclass:: DistrBase
    :members:

--------

.. autoclass:: Distr
    :members:

--------

.. autoclass:: Copula
    :members:

Utilities
---------

.. autofunction:: makedict

--------

.. autofunction:: distribution

.. _families:

Predefined families
-------------------

The parametrizations follow Wikipedia, while the class names are as in
`scipy.stats`.

"""

for name in sorted(distrs):
    distr = getattr(copula, name)
    sig = inspect.signature(distr.invfcn)
    params = dict(sig.parameters)
    params.pop(next(iter(params)))
    sig = inspect.Signature(params.values())
    out += f"""\
.. autoclass:: {name}{sig}
"""

# TODO make a table instead of listing the classes, columns: signature in
# lsqfitgp, name in scipy (with ref), wikipedia page name (with link), Notes

outfile = pathlib.Path(__file__).with_suffix('.rst').relative_to(pathlib.Path().absolute())
print(f'writing to {outfile}...')
outfile.write_text(out)
