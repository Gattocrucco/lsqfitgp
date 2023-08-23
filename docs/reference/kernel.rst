.. lsqfitgp/docs/reference/kernel.rst
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

Generic kernel classes
======================

All kernels in :mod:`lsqfitgp` are subclasses of :class:`Kernel`,
:class:`StationaryKernel` or :class:`IsotropicKernel` (the latter two are
themselves subclasses of :class:`Kernel`).

:class:`Kernel` itself is a subclass of :class:`CrossKernel`, which represents
the covariance function between two different processes. In symbols, the kernel
of a process :math:`f` is

.. math::
    k_f(x, y) = \operatorname{Cov}[f(x), f(y)],

while the cross kernel between processes :math:`f` and :math:`g` is

.. math::
    k_{fg}(x, y) = \operatorname{Cov}[f(x), g(y)].

However you will probably never need to deal directly with a cross kernel, they
are generated automatically behind the scenes by :class:`GP`.

The three general classes can be used directly by instantiating them with a
callable which will do the actual computation. However, this can be done in a
simpler and more functional way using the decorators :func:`kernel`,
:func:`stationarykernel` and :func:`isotropickernel`.

.. autoclass:: Kernel
    :inherited-members:

.. autoclass:: StationaryKernel    

.. autoclass:: IsotropicKernel
