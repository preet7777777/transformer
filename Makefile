.PHONY: install test style checks data benchmark

install:
	pip install -e ".[dev]"

test:
	pytest -q

style:
	black src/
	isort src/

checks:
	black --check src/
	ruff check src/

data:
	python -m transformer_from_scratch.prepare_synthetic

benchmark:
	python -m transformer_from_scratch.benchmark
