.. lsqfitgp/docs/kerneldec.rst
..
.. Copyright (c) Giacomo Petrillo 2020
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

.. _kerneldec:

Kernel decorators
=================

These decorators convert a callable to a subclass of :class:`Kernel`. The
decorators can be used both with and without keyword arguments. The keyword
arguments will be passed as initialization arguments to the superclass, and
will be overwritten by keyword arguments given at initialization of the
subclass.

.. autofunction:: kernel

.. autofunction:: isotropickernel
