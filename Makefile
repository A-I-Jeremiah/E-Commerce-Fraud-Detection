.PHONY: install eda process train evaluate api monitor test lint clean

install:
	pip install -r requirements.txt

eda:
	python notebooks/01_eda.py

process:
	python scripts/process_data.py

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

api:
	python scripts/run_api.py

monitor:
	python scripts/monitor.py

test:
	pytest tests/unit -v
	pytest tests/integration -v || true

lint:
	ruff check src api scripts --ignore E501 || true
	black --check src api scripts || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true