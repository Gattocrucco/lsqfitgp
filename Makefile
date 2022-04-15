# lsqfitgp/Makefile
#
# Makefile for running tests, prepare and upload a release.

RELEASE_TARGETS = tests examples docscode docs
TARGETS = upload release $(RELEASE_TARGETS)

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
	coverage run -m pytest
	coverage html

EXAMPLES = $(wildcard examples/*.py)
EXAMPLES := $(filter-out examples/runexamples.py, $(EXAMPLES))
.PHONY: $(EXAMPLES)

examples: $(EXAMPLES)

$(EXAMPLES):
	PYTHONPATH=. python examples/runexamples.py $@

docscode:
	cd docs && python runcode.py *.rst

docs:
	cd docs && python kernelsref.py
	cd docs && python examplesref.py
	make -C docs html
