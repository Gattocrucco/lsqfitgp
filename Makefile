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
	@echo " 1) push and check CI completes"
	@echo " 2) $$ make release"
	@echo " 3) $$ make upload"
	@echo " 4) publish the github release"
	@echo " 5) bump version number"
	@echo " 6) switch to branch gh-pages and pull"
	@echo " 7) add new version to index, commit and push"

upload:
	python3 -m twine upload dist/*

release: $(RELEASE_TARGETS)
	test -d build && rm -r build || test -
	test -d dist && rm -r dist || test -
	python3 -m build

PY = MPLBACKEND=agg coverage run
TESTSPY = COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX) $(PY) --context=tests$(COVERAGE_SUFFIX)
EXAMPLESPY = COVERAGE_FILE=.coverage.examples$(COVERAGE_SUFFIX) $(PY) --context=examples$(COVERAGE_SUFFIX)
DOCSPY = cd docs && COVERAGE_FILE=../.coverage.docs$(COVERAGE_SUFFIX) $(PY) --rcfile=../.coveragerc --context=docs$(COVERAGE_SUFFIX)

tests:
	$(TESTSPY) -m pytest -rfEX

# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

EXAMPLES = $(wildcard examples/*.py)
EXAMPLES := $(filter-out examples/runexamples.py, $(EXAMPLES))
EXAMPLES := $(filter-out examples/pdf7.py, $(EXAMPLES))
EXAMPLES := $(filter-out examples/pdf8.py, $(EXAMPLES))
EXAMPLES := $(filter-out examples/pdf9.py, $(EXAMPLES))
.PHONY: $(EXAMPLES)

examples: $(EXAMPLES)

$(EXAMPLES):
	$(EXAMPLESPY) examples/runexamples.py $@

docs/kernelsref.rst: docs/kernelsref.py lsqfitgp/_kernels/*.py lsqfitgp/_patch_jax/*.py lsqfitgp/_special/*.py
	$(DOCSPY) $(notdir $<)

docs/examplesref.rst: docs/examplesref.py lsqfitgp/*.py lsqfitgp/*/*.py
	$(DOCSPY) $(notdir $<)

GENDOCS := $(addsuffix .rst, $(basename $(wildcard docs/*ref.py)))
.PHONY: gendocs
gendocs: $(GENDOCS)

docscode:
	$(DOCSPY) runcode.py *.rst

docs: gendocs
	make -C docs html

covreport:
	coverage combine
	coverage html

resetenv:
	test -d pyenv && rm -fr pyenv || test -
	python3 -m venv pyenv
	pyenv/bin/python3 -m pip install --upgrade 'pip<23.1' # pip 23.1 breaks gvar
	pyenv/bin/python3 -m pip install -r requirements.txt
	@echo
	@echo 'Now type ". pyenv/bin/activate"'
