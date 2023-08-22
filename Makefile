# lsqfitgp/Makefile
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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

# Makefile for running tests, prepare and upload a release.

COVERAGE_SUFFIX=

RELEASE_TARGETS = tests examples docscode docs
TARGETS = upload release $(RELEASE_TARGETS) covreport resetenv

.PHONY: all $(TARGETS)

all:
	@echo "available targets: $(TARGETS)"
	@echo "release = $(RELEASE_TARGETS) (in order)"
	@echo
	@echo "Setup instructions:"
	@echo " 1) $$ make resetenv"
	@echo " 2) $$ . pyenv/bin/activate"
	@echo
	@echo "Release instructions:"
	@echo " 1) remove .devN suffix from version in src/lsqfitgp/__init__.py"
	@echo " 2) describe release in docs/development/changelog.md"
	@echo " 3) link versioned docs in docs/index.rst"
	@echo " 4) push and check CI completes"
	@echo " 5) $$ make release"
	@echo " 6) $$ make upload"
	@echo " 7) publish the github release"
	@echo " 8) bump version number and add .dev0 suffix"

upload:
	python3 -m twine upload dist/*

release: $(RELEASE_TARGETS)
	test ! -d dist || rm -r dist
	python3 -m build

PY = MPLBACKEND=agg coverage run
TESTSPY = COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX) $(PY) --context=tests$(COVERAGE_SUFFIX)
EXAMPLESPY = COVERAGE_FILE=.coverage.examples$(COVERAGE_SUFFIX) $(PY) --context=examples$(COVERAGE_SUFFIX)
DOCSPY = COVERAGE_FILE=.coverage.docs$(COVERAGE_SUFFIX) $(PY) --context=docs$(COVERAGE_SUFFIX)

tests:
	$(TESTSPY) -m pytest tests

# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

EXAMPLES = $(wildcard examples/*.py)
EXAMPLES := $(filter-out examples/runexamples.py, $(EXAMPLES)) # runner script
EXAMPLES := $(filter-out examples/pdf7.py, $(EXAMPLES)) # slow
EXAMPLES := $(filter-out examples/pdf8.py, $(EXAMPLES)) # slow
EXAMPLES := $(filter-out examples/pdf9.py, $(EXAMPLES)) # slow	
.PHONY: $(EXAMPLES)
examples: $(EXAMPLES)
	$(EXAMPLESPY) examples/runexamples.py $(EXAMPLES)

docscode:
	$(DOCSPY) docs/runcode.py docs/*.rst docs/*/*.rst

docs/reference/copula.rst: docs/reference/copula.py src/lsqfitgp/copula/*.py
	$(DOCSPY) --append $<

docs/examplesref.rst: docs/examplesref.py src/lsqfitgp/*.py src/lsqfitgp/*/*.py
	$(DOCSPY) --append $<

docs/reference/kernelsref.rst: docs/reference/kernelsref.py src/lsqfitgp/_kernels/*.py src/lsqfitgp/_patch_jax/*.py src/lsqfitgp/_special/*.py
	$(DOCSPY) --append $<

docs: docs/reference/copula.rst docs/examplesref.rst docs/reference/kernelsref.rst
	make -C docs html
	@echo
	@echo "Now open docs/_build/html/index.html"

covreport:
	coverage combine
	coverage html
	@echo
	@echo "Now open htmlcov/index.html"

resetenv:
	test ! -d pyenv || rm -fr pyenv
	python3 -m venv pyenv
	pyenv/bin/python3 -m pip install --upgrade pip
	pyenv/bin/python3 -m pip install --requirement requirements.txt
	pyenv/bin/python3 -m pip install --editable .
	@echo
	@echo 'Now type ". pyenv/bin/activate"'
