.. lsqfitgp/docs/development/setup.rst
..
.. Copyright (c) 2023, 2026, Giacomo Petrillo
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

Setup
=====

Initial setup
-------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ (for
example, with `Homebrew <https://brew.sh>`_ do :literal:`brew install uv`), then
clone the repository and set it up:

.. code-block:: sh

    $ git clone git@github.com:Gattocrucco/lsqfitgp.git
    $ cd lsqfitgp
    $ make setup

:literal:`make setup` installs the pre-commit hooks and checks that the Python
environment works. The environment, with the versions of the dependencies pinned
in :literal:`uv.lock`, is created in :literal:`.venv` by uv the first time it is
needed. To run commands that involve the Python installation, do
:literal:`uv run <command>`, e.g., :literal:`uv run ipython`. Alternatively, do
:literal:`source .venv/bin/activate` to activate the virtual environment in the
current shell.

The version of the package is derived from the git tags, so the repository must
be cloned with its history.

Pre-defined commands
--------------------

The :literal:`Makefile` in the root directory contains targets to run the tests
and the examples, build the documentation, update the dependencies, and prepare
a release. Run :literal:`make` without arguments to list the targets. All
targets that simply consist in invoking a tool with the right command line
arguments use the :literal:`ARGS` variable to add extra arguments, for example:

.. code-block:: sh

    $ make tests ARGS='-k test_pigs_fly'

will invoke something like

.. code-block:: sh

    $ uv run pytest --foo=1 --bar=128 --etc-etc -k test_pigs_fly

:literal:`make lint` runs all the linters through pre-commit, and
:literal:`make ipython` starts an IPython shell with the common imports.

Tests
-----

The typical workflow to debug new changes is to first run all tests with

.. code-block:: sh

    $ make tests

The tests run in parallel with pytest-xdist; set :literal:`NPROC` to change the
number of workers, :literal:`NPROC=0` disables parallelization. If some tests
fail, use :literal:`pytest` directly to run and debug only the relevant tests,
e.g., with

.. code-block:: sh

    $ uv run pytest --lf --sw --pdb

where :code:`--lf` selects only the tests that failed, :code:`--sw` stops on the
first failed test, starting again from it on the next run, and :code:`--pdb`
opens the python debugger at the point where the test failed. Another useful
option is :code:`-k <pattern>`, which selects only tests whose name matches
<pattern>.

:literal:`make tests-old` runs the tests with the oldest supported versions of
Python and of the dependencies, see `Dependencies`_.

The examples and the code in the documentation are run with :literal:`make
examples` and :literal:`make docscode`, which also produce the figures shown in
the manual. Then :literal:`make docs` builds the documentation in
:literal:`docs/_build/html`.

All these commands save coverage information. :literal:`make covreport` merges
it into an html report in :literal:`htmlcov/index.html`, and :literal:`make
covcheck` checks the coverage is above some thresholds. The tests are run on
each push, and the coverage report and the documentation for the main branch are
published online at `gattocrucco.github.io/lsqfitgp/htmlcov
<https://gattocrucco.github.io/lsqfitgp/htmlcov/>`_ and
`gattocrucco.github.io/lsqfitgp/docs
<https://gattocrucco.github.io/lsqfitgp/docs/>`_. The documentation of each
release is published in :literal:`docs-<version>` when its tag is pushed.

Dependencies
------------

The dependencies used for development are pinned in :literal:`uv.lock`. To
upgrade them to the latest versions, do :literal:`make update-deps`. This skips
versions published in the last week (see :literal:`COOLDOWN_DAYS` in the
:literal:`Makefile`).

The minimum supported versions of Python and of the dependencies are the lower
bounds in :literal:`pyproject.toml`. :literal:`make tests-old` resolves the
dependencies to these lower bounds, and their other dependencies to the latest
versions available at :literal:`OLD_DATE` (set in the :literal:`Makefile`).
:literal:`make update-oldest-deps` moves :literal:`OLD_DATE` to one year ago,
and raises the lower bounds to the latest versions available at that date, and
the minimum Python version to the oldest one among the last five releases.

To debug tests that fail with old versions of dependencies, it's convenient to
piggyback on the predefined make target using :code:`ARGS`:

.. code-block:: sh

    $ make tests-old ARGS='-n0 -k test_pigs_fly'

For more fine-grained control, it's useful to invoke directly :code:`uv` with
the :code:`--with` option, e.g., the following command will start an IPython
shell equipped with specific versions of python and jax:

.. code-block:: sh

    $ uv run --with='jax<0.7,jaxlib<0.7' --isolated --python=3.11 --dev python -m IPython
