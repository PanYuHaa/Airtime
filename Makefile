PYTHON ?= python3

.PHONY: help install install-dev check lint format test validate-configs env-check clean

help:
	@echo "Airtime Engineering Targets:"
	@echo "  make install           Install package in editable mode"
	@echo "  make install-dev       Install package with dev tools (pytest, ruff)"
	@echo "  make validate-configs  Validate configs/*.json schema integrity"
	@echo "  make lint              Run ruff static checks"
	@echo "  make format            Check/format code with ruff"
	@echo "  make test              Run pytest unit test suite"
	@echo "  make env-check         Run airtime-check environment diagnostics"
	@echo "  make check             Run validate-configs + lint + test"
	@echo "  make clean             Remove Python build/cache artifacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

validate-configs:
	$(PYTHON) scripts/validate_configs.py

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts

test:
	PYTHONPATH=src $(PYTHON) -m pytest -v

env-check:
	PYTHONPATH=src $(PYTHON) -m airtime.cfg

check: validate-configs lint test

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find src tests scripts -name "__pycache__" -type d -exec rm -rf {} +
