.. lsqfitgp/docs/installation.rst
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

.. _installation:

Installation
============

First, you have to get a working Python interpreter. There are three main options: the official package from `<https://www.python.org>`_, the Anaconda distribution `<https://www.anaconda.com>`_, and the `Spyder IDE <https://www.spyder-ide.org>`_. The latter is probably the easier one if it's your first time with Python.

Then, install :mod:`lsqfitgp` by running this command in a shell:

.. code-block:: sh

    pip install lsqfitgp

Windows
-------

JAX is still a bit buggy on Windows, a few tests in lsqfitgp's test suite fail due to accuracy problems.
