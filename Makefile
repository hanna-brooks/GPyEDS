.PHONY: help install lint format test docs build

help:  ## Show this help menu
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies using uv
	uv sync --all-extras

lint: ## Run Ruff and Mypy
	uv run ruff check .
	uv run mypy .

format: ## Format code using Ruff
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run tests with pytest
	uv run pytest --cov .

docs: ## Serve MkDocs documentation locally
	uv run mkdocs serve

build: ## Build the Python package
	uv build

# Catch-all target: route all unknown targets to help
%:
	@echo "Unknown target: $@. Redirecting to help..."
	@$(MAKE) help
