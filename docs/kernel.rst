.. lsqfitgp/docs/kernel.rst
..
.. Copyright (c) 2020, Giacomo Petrillo
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

All kernels in :mod:`lsqfitgp` are subclasses of :class:`Kernel` or
:class:`IsotropicKernel` (which is itself a subclass of :class:`Kernel`).

The two general classes can be used directly by instantiating them with a
callable which will do the actual computation. However, this can be done in a
simpler and more functional way using the decorators :func:`kernel` and
:func:`isotropickernel`.

Kernel
------

.. autoclass:: Kernel
    :inherited-members: diff
    
IsotropicKernel
---------------

.. autoclass:: IsotropicKernel
