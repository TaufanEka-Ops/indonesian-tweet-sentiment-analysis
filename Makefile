PYTHON ?= python
VENV   ?= .venv

.PHONY: help setup test lint all audit novel model benchmark notebooks clean

help:
	@echo "Targets:"
	@echo "  setup      create .venv and install requirements.txt"
	@echo "  test       run the pytest suite"
	@echo "  audit      scripts/01 -- data audit for both corpora"
	@echo "  novel      scripts/02 + 03 -- preprocess & explore the Novel Baswedan set"
	@echo "  benchmark  scripts/04 -- np.vectorize vs real vectorisation"
	@echo "  model      scripts/05 + 06 -- train & error-analyse the sentiment classifier"
	@echo "  notebooks  build & execute the notebooks (outputs embedded)"
	@echo "  all        audit + novel + benchmark + model + notebooks"
	@echo "  clean      remove generated caches"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

test:
	$(PYTHON) -m pytest tests/ -v

audit:
	$(PYTHON) scripts/01_audit_data.py

novel:
	$(PYTHON) scripts/02_preprocess_novel_baswedan.py
	$(PYTHON) scripts/03_explore_novel_baswedan.py

benchmark:
	$(PYTHON) scripts/04_vectorization_benchmark.py

model:
	$(PYTHON) scripts/05_train_modi_sentiment.py
	$(PYTHON) scripts/06_error_analysis.py

notebooks:
	$(PYTHON) scripts/07_build_and_execute_notebooks.py

all: audit novel benchmark model notebooks
	@echo "Done. See results/ for tables and figures."

clean:
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
