.PHONY: install test style checks data benchmark public-eval public-benchmark

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

public-eval:
	python -m transformer_from_scratch.public_eval

public-benchmark:
	python -m transformer_from_scratch.public_benchmark
