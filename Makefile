.PHONY: setup lint format test run eval index up down clean

PYTHON ?= python3
PIP ?= pip
DOCKER_COMPOSE ?= docker compose -f infra/docker-compose.yml

setup:
	$(PIP) install -e "backend[dev,mlops]"

lint:
	ruff check backend
	mypy --config-file backend/pyproject.toml backend/src/sanad

format:
	ruff format backend
	ruff check --fix backend

test:
	pytest backend/tests --cov=sanad --cov-report=term-missing --cov-fail-under=70

run:
	uvicorn sanad.api.app:app --host 0.0.0.0 --port 8000 --reload

eval:
	python -m sanad.cli eval

index:
	python -m sanad.cli ingest

up:
	$(DOCKER_COMPOSE) up -d --build

down:
	$(DOCKER_COMPOSE) down -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

