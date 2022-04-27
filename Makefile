# lsqfitgp/Makefile
#
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
	rm -r dist
	python setup.py sdist bdist_wheel

tests:
	coverage run --context=tests$(COVERAGE_SUFFIX) --data-file=.coverage.tests$(COVERAGE_SUFFIX) -m pytest

EXAMPLES = $(wildcard examples/*.py)
EXAMPLES := $(filter-out examples/runexamples.py, $(EXAMPLES))
.PHONY: $(EXAMPLES)

examples: $(EXAMPLES)

$(EXAMPLES):
	coverage run --context=examples$(COVERAGE_SUFFIX) --data-file=.coverage.examples$(COVERAGE_SUFFIX) examples/runexamples.py $@

docscode:
	cd docs && coverage run --rcfile=../.coveragerc --context=docs$(COVERAGE_SUFFIX) --data-file=../.coverage.docs$(COVERAGE_SUFFIX) runcode.py *.rst

docs:
	cd docs && python kernelsref.py
	cd docs && python examplesref.py
	make -C docs html

covreport:
	coverage combine
	coverage html --show-contexts
