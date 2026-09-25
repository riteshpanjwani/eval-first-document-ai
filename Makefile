PYTHON ?= python3
VENV ?= .venv
RUN := $(VENV)/bin/python

.PHONY: setup fixtures demo app test lint check clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[dev,app,llm]'

fixtures:
	$(RUN) scripts/generate_fixtures.py

demo:
	$(RUN) -m evaldocai.cli demo

app:
	$(VENV)/bin/streamlit run app.py

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check .

check: lint test

clean:
	rm -rf artifacts .pytest_cache .ruff_cache htmlcov .coverage
