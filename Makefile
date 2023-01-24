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
TARGETS = upload release $(RELEASE_TARGETS) covreport

.PHONY: all $(TARGETS)

all:
	@echo "available targets: $(TARGETS)"
	@echo "release = $(RELEASE_TARGETS) (in order)"

upload:
	twine upload dist/*

release: $(RELEASE_TARGETS)
	test -d build && rm -r build || test -
	test -d dist && rm -r dist || test -
	python setup.py sdist bdist_wheel

PY = MPLBACKEND=agg coverage run
TESTSPY = COVERAGE_FILE=.coverage.tests$(COVERAGE_SUFFIX) $(PY) --context=tests$(COVERAGE_SUFFIX)
EXAMPLESPY = COVERAGE_FILE=.coverage.examples$(COVERAGE_SUFFIX) $(PY) --context=examples$(COVERAGE_SUFFIX)
DOCSPY = cd docs && COVERAGE_FILE=../.coverage.docs$(COVERAGE_SUFFIX) $(PY) --rcfile=../.coveragerc --context=docs$(COVERAGE_SUFFIX)

tests:
	$(TESTSPY) -m pytest

# I did not manage to make parallel pytest (pytest -n<processes>) work with
# coverage

EXAMPLES = $(wildcard examples/*.py)
EXAMPLES := $(filter-out examples/runexamples.py, $(EXAMPLES))
EXAMPLES := $(filter-out examples/pdf7.py, $(EXAMPLES))
EXAMPLES := $(filter-out examples/pdf8.py, $(EXAMPLES))
.PHONY: $(EXAMPLES)

examples: $(EXAMPLES)

$(EXAMPLES):
	$(EXAMPLESPY) examples/runexamples.py $@

docs/kernelsref.rst: docs/kernelsref.py lsqfitgp/_kernels/*.py lsqfitgp/_patch_jax.py
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

# TODO write a rule that
# - removes an eventual preexisting venv
# - creates a new venv
# - installs requirements
# - activate the venv
# and also a rule just to activate the venv
# so the hierarchy is:
# 1 true rule to create the venv
# 2 phony rule to activate it, depends on 1
# 3 phony rule to delete the venv, recreate and activate it
