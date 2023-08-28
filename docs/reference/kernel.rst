.. lsqfitgp/docs/reference/kernel.rst
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

Generic kernel classes
======================

All kernels in `lsqfitgp` are subclasses of `CrossKernel`. The user should
almost exclusively deal with its subclass `Kernel`, as `CrossKernel` objects
are produced behind the scenes by `GP`.

The generic classes can be used standalone. To ease subclassing, use the
:ref:`decorators <kerneldec>`.

Index
-----
`CrossKernel`, `Kernel`, `CrossStationaryKernel`, `StationaryKernel`,
`CrossIsotropicKernel`, `IsotropicKernel`

Classes
-------

.. autoclass:: CrossKernel
    :members:

----

.. autoclass:: Kernel
    :members:

----

.. autoclass:: CrossStationaryKernel
    :members:

----

.. autoclass:: StationaryKernel
    :members:

----

.. autoclass:: CrossIsotropicKernel
    :members:

----

.. autoclass:: IsotropicKernel
    :members:
