.PHONY: setup lint typecheck test coverage release-tag

setup:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	black --check src tests
	isort --check-only src tests

typecheck:
	mypy src

test:
	pytest tests

coverage:
	pytest --cov=alchemi --cov-report=term-missing tests

release-tag:
	@if git rev-parse v0.1.0 >/dev/null 2>&1; then \
		echo "Tag v0.1.0 already exists" >&2; \
		exit 1; \
	fi
	git tag -a v0.1.0 -m "Phase-1 complete"
	@echo "Created tag v0.1.0"
