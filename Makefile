# lsqfitgp/Makefile
#
# Copyright (c) 2022, 2023, 2024, Giacomo Petrillo
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
TARGETS = upload release $(RELEASE_TARGETS) covreport resetenv resetenv-old clean

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
	@echo " 4) commit, push and check CI completes"
	@echo " 5) $$ make release"
	@echo " 6) repeat 4 and 5 until everything goes smoothly"
	@echo " 7) $$ make upload"
	@echo " 8) publish the github release"
	@echo " 9) bump version number and add .dev0 suffix"
	@echo
	@echo "Dependency maintenance instructions:"
	@echo " 1) $$ make clean"
	@echo " 2) $$ make resetenv"
	@echo " 3) $$ make tests examples docscode docs"
	@echo " 4) repeat 3 until all problems are fixed"

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

examples: $(EXAMPLES)
	$(EXAMPLESPY) examples/runexamples.py $(EXAMPLES)

docs/reference/copula.rst: docs/reference/copula.py src/lsqfitgp/copula/*.py
	$(DOCSPY) --append $<

docs/examplesref.rst: docs/examplesref.py src/lsqfitgp/*.py src/lsqfitgp/*/*.py
	$(DOCSPY) --append $<

docs/reference/kernelsref.rst: docs/reference/kernelsref.py src/lsqfitgp/_kernels/*.py src/lsqfitgp/_jaxext/*.py src/lsqfitgp/_special/*.py
	$(DOCSPY) --append $<

docs/reference/kernelop.rst: docs/reference/kernelop.py src/lsqfitgp/_Kernel/*.py src/lsqfitgp/_kernels/*.py src/lsqfitgp/_jaxext/*.py src/lsqfitgp/_special/*.py
	$(DOCSPY) --append $<

## TODO: since I'm appending to .coverage.docs, delete it when I start a
##  release. How do I tell make to first delete a file but do not re-delete it
##  for each target?

GENDOCS = docs/reference/copula.rst docs/examplesref.rst docs/reference/kernelsref.rst docs/reference/kernelop.rst

docscode: $(GENDOCS)
	$(DOCSPY) docs/runcode.py docs/*.rst docs/*/*.rst
	## am I not missing an --append here?

docs: $(GENDOCS)
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
	@echo using `which python3`
	python3 -m venv pyenv
	pyenv/bin/python3 -m pip install --upgrade pip
	pyenv/bin/python3 -m pip install --editable '.[dev]'
	@echo
	@echo 'Now type ". pyenv/bin/activate"'

resetenv-old:
	test ! -d pyenv-old || rm -fr pyenv-old
	@echo using `which python3`
	test 3.9 = `python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")'` 
	python3 -m venv pyenv-old
	pyenv-old/bin/python3 -m pip install --upgrade pip
	pyenv-old/bin/python3 -m pip install --editable '.[dev,tests-old]'
	@echo
	@echo 'Now type ". pyenv-old/bin/activate"'

clean:
	rm -f $(GENDOCS)
	make -C docs clean
	rm -fr htmlcov
	rm -f .coverage*
	rm -f coverage.xml
	rm -fr dist
	rm -fr src/lsqfitgp.egg-info
	rm -fr .pytest_cache
	rm -fr src/lsqfitgp/__pycache__
	rm -fr src/lsqfitgp/*/__pycache__
	rm -fr tests/__pycache__
	rm -fr tests/*/__pycache__
