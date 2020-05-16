# lsqfitgp/Makefile
#
# Makefile for running tests, prepare and upload a release.

TARGETS = upload release tests examples docscode docs

.PHONY: all $(TARGETS)

all:
	@echo "available targets: $(TARGETS)"

upload:
	twine upload dist/*

release: tests examples docscode docs
	rm -r dist
	python setup.py sdist bdist_wheel

tests:
	coverage run -m pytest

EXAMPLES = $(wildcard examples/*.py)
.PHONY: $(EXAMPLES)

examples: $(EXAMPLES)

$(EXAMPLES):
	python $@

docscode:
	cd docs && python runcode.py *.rst

docs:
	make -C docs html
