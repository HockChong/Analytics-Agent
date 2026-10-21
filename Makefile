ifeq ($(OS),Windows_NT)
PYTHON ?= python
PY := .venv\Scripts\python.exe
else
PYTHON ?= python3
PY := .venv/bin/python
endif
.PHONY: setup fetch build test clean-warehouse

setup:  ## create the virtualenv and install the package
	$(PYTHON) -m venv .venv && $(PY) -m pip install -q -e ".[dev,extractors]"

fetch:  ## download sources listed in data/sources.yaml and check pinned hashes
	$(PY) -m finrag.fetch

build:  ## parse (cached) + evidence + gates + publish, incremental
	$(PY) -m finrag.evidence.publish

test:
	$(PY) -m pytest -q

clean-warehouse:  ## drop the published warehouse (the parse cache is kept)
	$(PY) -c "import shutil; shutil.rmtree('warehouse', ignore_errors=True)"
