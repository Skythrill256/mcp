# Development helpers
.PHONY: lint ruff mypy pydocstyle bandit secrets all run

lint: ruff mypy pydocstyle
	@echo "Linting complete"

ruff:
	ruff check .

mypy:
	mypy --strict .

pydocstyle:
	pydocstyle core

bandit:
	bandit -r core -lll

secrets:
	detect-secrets scan > .secrets.baseline || true

all: lint bandit secrets
	@echo "All checks complete"

run:
	uv run uvicorn main:app --host 127.0.0.1 --port 8000
