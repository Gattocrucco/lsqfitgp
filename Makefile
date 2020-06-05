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
.PHONY: $(EXAMPLES)

examples: $(EXAMPLES)

$(EXAMPLES):
	python $@

docscode:
	cd docs && python runcode.py *.rst

docs:
	cd docs && python kernelsref.py
	make -C docs html
