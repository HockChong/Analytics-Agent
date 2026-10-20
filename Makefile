PY := .venv/bin/python
.PHONY: setup fetch build test clean-warehouse

setup:  ## create the virtualenv and install the package
	python3 -m venv .venv && $(PY) -m pip install -q -e ".[dev]"

fetch:  ## download sources listed in data/sources.yaml and check pinned hashes
	$(PY) -m finrag.fetch

build:  ## parse (cached) + evidence + gates + publish, incremental
	$(PY) -m finrag.evidence.publish

test:
	$(PY) -m pytest -q

clean-warehouse:  ## drop the published warehouse (the parse cache is kept)
	rm -rf warehouse
