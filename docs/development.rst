.. lsqfitgp/docs/development.rst
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

.. currentmodule:: lsqfitgp

Development
===========

To work on `lsqfitgp`, clone the repository, create a virtual environment and
install the requirements:

::

    $ git clone git@github.com:Gattocrucco/lsqfitgp.git
    $ cd lsqfitgp
    $ make resetenv
    $ . pyenv/bin/activate
    (pyenv) $


The ``Makefile`` in the root directory contains targets to build the
documentation, run the tests, and prepare a release. Run ``make`` without
arguments to show the available targets:

::

    (pyenv) $ make
    available targets: [...]
    (pyenv) $ make tests # or make examples, or ...

The tests are run on each push and the resulting coverage report is published
online at `gattocrucco.github.io/lsqfitgp/htmlcov
<https://gattocrucco.github.io/lsqfitgp/htmlcov/>`_. To browse it locally after
``make tests`` etc., do ``make covreport`` and open ``htmlcov/index.html`` in
your browser.

Licensed sources
----------------

This software contains code adapted from the following sources:

  * `TOEPLITZ_CHOLESKY
    <http://people.sc.fsu.edu/~jburkardt/py_src/toeplitz_cholesky/toeplitz_cholesky.html>`_
    by John Burkardt (LGPL license)
  * `SuperGauss <https://cran.r-project.org/package=SuperGauss>`_ by
    Yun Ling and Martin Lysy (GPL license)
  * `gvar <https://github.com/gplepage/gvar>`_ by Peter Lepage (GPL license)
  * `scipy <https://github.com/scipy/scipy>`_ (BSD-3 License)
  * `numpy <https://github.com/numpy/numpy>`_ (BSD-3 License)
