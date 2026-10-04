# lsqfitgp/Makefile
#
# Copyright (c) 2022, 2023, 2024, 2025, 2026, Giacomo Petrillo
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

# Refuse -j: recipes manage their own parallelism (pytest-xdist) and the
# release pipeline relies on serial prerequisite order (e.g. build before
# check-dist). Global because GNU make <4.4 can't scope .NOTPARALLEL to a target.
.NOTPARALLEL:

# define command to run python
UV_RUN = uv run --dev

# Anchor all date-based dependency and release policies to the start of the
# current UTC day, as an explicit RFC 3339 instant so the config scripts resolve
# the same absolute cutoff regardless of the runner's local timezone. Repeated
# `make release` runs on the same UTC day resolve identically; crossing into the
# next UTC day may introduce one fresh bump. `:=` samples the date once per make
# invocation.
TODAY := $(shell date -u +%Y-%m-%dT00:00:00Z)

# Dependency cooldown: `update-python-deps` pins pyproject's `[tool.uv]
# exclude-newer` to TODAY minus this many days, so releases skip just-published
# versions. It is pinned to an absolute instant (not uv's relative `1 week` span)
# so the release lock and the uv-lock pre-commit hook resolve the same cutoff and
# don't churn uv.lock.
COOLDOWN_DAYS = 7

# define command to run python with oldest supported dependencies
# OLD_DATE / OLD_DELAY_DAYS / BUMP_PYTHON_VERSION_DATE / NUM_SUPPORTED_PYTHON_RELEASES
# drive the `update-oldest-deps` policy. The old toolchain swaps the `docs`
# dependency group, which is not needed to run the unit tests, with `old` (see
# the comments in pyproject.toml).
OLD_DATE = 2025-10-04T00:00:00Z
OLD_DELAY_DAYS = 365
BUMP_PYTHON_VERSION_DATE = 10-31
NUM_SUPPORTED_PYTHON_RELEASES = 5
OLD_PYTHON = $(shell grep 'requires-python' pyproject.toml | sed 's/.*>=\([0-9.]*\).*/\1/')
# WORKAROUND(ty<=0.0.2): the first final ty release (0.0.2, 0.0.1 is yanked)
# was uploaded on 2025-12-16, newer than OLD_DATE, so the old toolchain cannot
# resolve it. Exempt ty from the cutoff. The exemption is unneeded once OLD_DATE
# passes the upload date of the ty floor; on trigger (`update-oldest-deps`
# raised the floor), drop --exclude-newer-package.
UV_RUN_OLD = $(UV_RUN) --no-group=docs --group=old --python=$(OLD_PYTHON) --resolution=lowest-direct --exclude-newer=$(OLD_DATE) --exclude-newer-package="ty=0 days" --isolated

.PHONY: help
help:
	@echo "Available targets:"
	@echo "- setup: install the pre-commit hooks and check python and jax work"
	@echo "- tests: run unit tests, saving coverage information"
	@echo "- tests-old: run unit tests with oldest supported python and dependencies"
	@echo "- examples: run the examples, saving coverage information and figures"
	@echo "- docscode: run the code in the documentation, saving coverage information and figures"
	@echo "- docs: build html documentation (run examples and docscode first to get the figures)"
	@echo "- covreport: build html coverage report"
	@echo "- covcheck: check coverage is above some thresholds"
	@echo "- diffcov: check changed-lines coverage vs DIFF_BASE (default origin/main)"
	@echo "- update-deps: update-python-deps + update-other-deps"
	@echo "- update-python-deps: pin the dep cooldown, then upgrade uv.lock to the latest allowed deps"
	@echo "- update-other-deps: upgrade pre-commit hooks (not date-pinned)"
	@echo "- update-oldest-deps: advance OLD_DATE and refresh oldest-supported pins in pyproject.toml"
	@echo "- check-committed: verify there are no uncommitted changes"
	@echo "- check-changelog: verify the topmost changelog section is dated today"
	@echo "- build: build the python wheel and sdist"
	@echo "- check-dist: verify dist/ artifacts carry the release version"
	@echo "- smoke-test: check the built wheel and sdist can be installed and imported"
	@echo "- release: run tests, build, and upload to PyPI (run on main)"
	@echo "- version-tag: create local git tag for the topmost changelog version"
	@echo "- push-tag: push the version tag to origin"
	@echo "- upload: upload release to PyPI"
	@echo "- upload-test: upload release to TestPyPI"
	@echo "- gh-release: create draft GitHub release from docs/development/changelog.md"
	@echo "- ipython: start an ipython shell with stuff pre-imported"
	@echo "- ipython-old: start an ipython shell with oldest supported python and dependencies"
	@echo "- lint: run pre-commit hooks on all files"
	@echo "- clean: remove generated files and the virtual environment"
	@echo
	@echo "Release workflow:"
	@echo "- describe release in docs/development/changelog.md, its topmost header"
	@echo "  \`## <version>. <title> (<today>)\` sets the version"
	@echo "- link the versioned docs (docs-<version>) in docs/index.rst"
	@echo "- $$ make setup"
	@echo "- $$ make update-deps"
	@echo "- $$ make release, will not release but runs all tests, iterate and debug"
	@echo "- merge a PR with the changes and fixes"
	@echo "- on main: $$ make release"
	@echo "- merge fix PR and try again until make release passes"
	@echo "- publish the draft github release created by make release"
	@echo "- the tag push deploys the versioned docs, check they are online"


################# SETUP #################

.PHONY: setup
setup:
	$(UV_RUN) pre-commit install --install-hooks
	$(UV_RUN) python -c 'import jax, lsqfitgp; jax.numpy.empty(0); print(lsqfitgp.__version__)'

.PHONY: lint
lint:
	$(UV_RUN) pre-commit run $(if $(ARGS),$(ARGS),--all-files)

GENDOCS = docs/reference/copula.rst docs/examplesref.rst docs/reference/kernelsref.rst docs/reference/kernelop.rst

.PHONY: clean
clean:
	rm -fr .venv
	rm -fr dist
	rm -f $(GENDOCS)
	rm -fr docs/_build
	rm -f docs/*/*.png docs/*/*/*.png
	rm -fr htmlcov
	rm -f .coverage .coverage.* $(COV_DATA) coverage.xml diffcov.md
	rm -fr config/pytest_cache config/ruff_cache


################# TESTS #################

# Number of xdist workers, 0 to disable xdist.
NPROC ?= 4

# Appended to the coverage file names, to tell apart the files produced by
# different CI jobs.
COVERAGE_SUFFIX =

TESTS_VARS = COVERAGE_FILE=.coverage.$@$(COVERAGE_SUFFIX)
TESTS_COMMAND = python -m pytest --cov --numprocesses=$(NPROC) --dist=worksteal

.PHONY: tests
tests:
	$(TESTS_VARS) $(UV_RUN) $(TESTS_COMMAND) $(ARGS)

.PHONY: tests-old
tests-old:
	$(TESTS_VARS) $(UV_RUN_OLD) $(TESTS_COMMAND) $(ARGS)

# commands to run scripts with coverage, as examples and documentation
PY = MPLBACKEND=agg $(UV_RUN) coverage run
EXAMPLESPY = COVERAGE_FILE=.coverage.examples$(COVERAGE_SUFFIX) $(PY) --context=examples
DOCSPY = COVERAGE_FILE=.coverage.docs$(COVERAGE_SUFFIX) $(PY) --context=docs

EXAMPLES = $(wildcard examples/*.py)
EXAMPLES := $(filter-out examples/runexamples.py, $(EXAMPLES)) # runner script
EXAMPLES := $(filter-out examples/pdf7.py, $(EXAMPLES)) # slow
EXAMPLES := $(filter-out examples/pdf8.py, $(EXAMPLES)) # slow
EXAMPLES := $(filter-out examples/pdf9.py, $(EXAMPLES)) # slow

.PHONY: examples
examples:
	$(EXAMPLESPY) examples/runexamples.py $(EXAMPLES)


################# DOCS #################

docs/reference/copula.rst: docs/reference/copula.py src/lsqfitgp/copula/*.py
	$(DOCSPY) --append $<

docs/examplesref.rst: docs/examplesref.py src/lsqfitgp/*.py src/lsqfitgp/*/*.py
	$(DOCSPY) --append $<

docs/reference/kernelsref.rst: docs/reference/kernelsref.py src/lsqfitgp/_kernels/*.py src/lsqfitgp/_jaxext/*.py src/lsqfitgp/_special/*.py
	$(DOCSPY) --append $<

docs/reference/kernelop.rst: docs/reference/kernelop.py src/lsqfitgp/_Kernel/*.py src/lsqfitgp/_kernels/*.py src/lsqfitgp/_jaxext/*.py src/lsqfitgp/_special/*.py
	$(DOCSPY) --append $<

.PHONY: docscode
docscode: $(GENDOCS)
	$(DOCSPY) --append docs/runcode.py docs/*.rst docs/*/*.rst

.PHONY: docs
docs: $(GENDOCS)
	$(UV_RUN) make -C docs html
	@echo
	@echo "Now open docs/_build/html/index.html"


################# COVERAGE #################

# Each run writes its own .coverage.<run> file (COVERAGE_FILE), merged into the
# default data file .coverage-combined (set in pyproject.toml). The default data
# file has a different prefix on purpose: the coverage reporting commands
# automatically merge and delete the `<data file>.*` files, while this way the
# single-run files are kept, and .coverage-combined is rebuilt from scratch when
# they change.
COV_DATA = .coverage-combined

$(COV_DATA): $(wildcard .coverage.*)
	rm -f $@
	$(UV_RUN) coverage combine --keep --data-file=$@ $^

.PHONY: covcombine
covcombine: $(COV_DATA)

.PHONY: covreport
covreport: covcombine
	$(UV_RUN) coverage html --include='src/*'
	@echo
	@echo "Now open htmlcov/index.html"

.PHONY: covcheck
covcheck: covcombine
	$(UV_RUN) coverage report --include='tests/**/test_*.py'
	$(UV_RUN) coverage report --include='src/*'
	$(UV_RUN) coverage report --include='tests/**/test_*.py' --fail-under=97 --format=total
	$(UV_RUN) coverage report --include='src/*' --fail-under=94 --format=total

# Branch (changed-lines) coverage: fail if new/modified lines in src and tests
# are not covered above the threshold. DIFF_BASE is the ref to diff against;
# locally a feature branch is compared to origin/main. Writes a markdown report
# (used by CI to populate the job summary) and prints the text report.
DIFF_BASE ?= origin/main
DIFFCOV_FAIL_UNDER ?= 99
DIFFCOV_REPORT ?= diffcov.md

.PHONY: diffcov
diffcov: covcombine
	# -i: the xml is only an input to diff-cover, which assesses just the
	# changed files (always present in the checkout); never fail xml generation
	# over an unrelated path missing in the combined data.
	$(UV_RUN) coverage xml -i -o coverage.xml
	$(UV_RUN) diff-cover coverage.xml --compare-branch=$(DIFF_BASE) --fail-under=$(DIFFCOV_FAIL_UNDER) --format report:- --format markdown:$(DIFFCOV_REPORT)


################# DEPENDENCIES #################

# `update-deps` = latest python deps (uv) + everything else (pre-commit).
# Only `update-python-deps` runs inside `release`: it pins pyproject's
# exclude-newer to a fixed instant before locking, so repeated same-day release
# runs (and the uv-lock hook) resolve identically and don't churn. pre-commit
# has no date knob, so it's bumped once by hand via `make update-deps` at the
# start of the release, outside the release loop.
.PHONY: update-deps
update-deps: update-python-deps update-other-deps

.PHONY: update-python-deps
update-python-deps:
	$(UV_RUN) python config/update_cooldown.py --today=$(TODAY) --cooldown-days=$(COOLDOWN_DAYS)
	uv lock --upgrade

.PHONY: update-other-deps
update-other-deps:
	# --freeze pins revs to commit SHAs (tags are mutable)
	$(UV_RUN) pre-commit autoupdate --freeze

.PHONY: update-oldest-deps
update-oldest-deps:
	$(UV_RUN) python config/update_python_version.py --bump-date=$(BUMP_PYTHON_VERSION_DATE) --num-supported=$(NUM_SUPPORTED_PYTHON_RELEASES) --today=$(TODAY)
	$(UV_RUN) python config/update_oldest_deps.py --min-old-date=$(OLD_DATE) --delay-days=$(OLD_DELAY_DAYS) --today=$(TODAY)
	uv lock


################# RELEASE #################

.PHONY: check-committed
check-committed:
	git diff --quiet
	git diff --quiet --staged

.PHONY: check-changelog
check-changelog:
	$(UV_RUN) python config/util.py check_changelog --today=$(TODAY)

.PHONY: build
build:
	# remove stale artifacts: uv publish would upload everything in dist/
	rm -fr dist
	uv build

# The version is derived from the git tag at build time (hatch-vcs), so the
# tag must exist before `build` (`check-dist` verifies this on the
# artifacts). It is created locally first and pushed only after the build
# artifacts pass `check-dist` and `smoke-test`, to avoid editing a published
# tag if something fails in between.
.PHONY: release
release: check-changelog clean setup update-oldest-deps update-python-deps check-committed tests tests-old examples docscode docs version-tag build upload gh-release
	@echo "Done!"

.PHONY: version-tag
version-tag: check-committed
	test $(shell git rev-parse --abbrev-ref HEAD) = main
	git fetch --tags
	$(eval VERSION_TAG := v$(shell $(UV_RUN) python config/util.py get_version))
	@if git rev-parse -q --verify refs/tags/$(VERSION_TAG) >/dev/null; then \
		test "$$(git rev-list -n 1 $(VERSION_TAG))" = "$$(git rev-parse HEAD)" \
			|| { echo "Tag $(VERSION_TAG) exists but points to a different commit;"; \
			     echo "if it is a leftover never pushed, delete it: git tag -d $(VERSION_TAG)"; exit 1; }; \
		echo "Tag $(VERSION_TAG) already exists on current commit"; \
	else \
		git tag --message=$(VERSION_TAG) $(VERSION_TAG); \
	fi

.PHONY: push-tag
push-tag: version-tag check-dist smoke-test
	git push origin $(VERSION_TAG)

# Untagged builds carry a +g<commit> local version segment, which PyPI and
# TestPyPI reject; this catches dist/ built before tagging, or gone stale.
.PHONY: check-dist
check-dist:
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
	test -e "dist/lsqfitgp-$$VERSION.tar.gz" && test -e "dist/lsqfitgp-$$VERSION-py3-none-any.whl" || { \
		echo "dist/ does not carry the release version $$VERSION:"; \
		ls dist/ 2>/dev/null; \
		echo "build with the tag in place: make version-tag build"; \
		exit 1; }

.PHONY: smoke-test
smoke-test:
	uv run --isolated --no-project --with dist/*.whl python -c 'import lsqfitgp; print(lsqfitgp.__version__)'
	uv run --isolated --no-project --with dist/*.tar.gz python -c 'import lsqfitgp; print(lsqfitgp.__version__)'

.PHONY: upload
upload: push-tag
	@echo "Enter PyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN && \
	uv publish
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
	echo "Try to install lsqfitgp $$VERSION from PyPI" && \
	uv tool run --exclude-newer-package="lsqfitgp=0 days" --with="lsqfitgp==$$VERSION" python -c 'import lsqfitgp; print(lsqfitgp.__version__)'

# Like `upload`, but the tag stays local: TestPyPI uploads are rehearsals.
.PHONY: upload-test
upload-test: version-tag check-dist smoke-test
	@echo "Enter TestPyPI token:"
	@read -s UV_PUBLISH_TOKEN && \
	export UV_PUBLISH_TOKEN && \
	uv publish --check-url=https://test.pypi.org/simple/ --publish-url=https://test.pypi.org/legacy/
	@VERSION=$$($(UV_RUN) python config/util.py get_version) && \
	echo "Try to install lsqfitgp $$VERSION from TestPyPI" && \
	uv tool run --exclude-newer-package="lsqfitgp=0 days" --index=https://test.pypi.org/simple/ --index-strategy=unsafe-best-match --with="lsqfitgp==$$VERSION" python -c 'import lsqfitgp; print(lsqfitgp.__version__)'

.PHONY: gh-release
gh-release: push-tag
	$(UV_RUN) python config/util.py gh_release --today=$(TODAY)


################# IPYTHON SHELL #################

.PHONY: ipython
ipython:
	IPYTHONDIR=config/ipython $(UV_RUN) python -m IPython $(ARGS)

.PHONY: ipython-old
ipython-old:
	IPYTHONDIR=config/ipython $(UV_RUN_OLD) python -m IPython $(ARGS)
