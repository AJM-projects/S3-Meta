# Simple Makefile for linting and formatting Python code using Ruff

PYTHON ?= python3
RUFF ?= $(PYTHON) -m ruff
RUFF_PKG ?= ruff

.DEFAULT_GOAL := help

.PHONY: help setup lint lint-fix format format-check check

help: ## Show this help
	@printf "Available targets:\n"; \
	awk -F ':|##' '/^[a-zA-Z0-9_\-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$NF}' $(MAKEFILE_LIST)

setup: ## Install/upgrade tooling (ruff)
	$(PYTHON) -m pip install -U $(RUFF_PKG)

lint: ## Run lint checks (ruff)
	$(RUFF) check .

lint-fix: ## Auto-fix lint issues (ruff)
	$(RUFF) check --fix .

format: ## Format code (ruff)
	$(RUFF) format .

format-check: ## Check formatting (ruff)
	$(RUFF) format --check .

check: ## Run lint and formatting checks
	$(MAKE) lint
	$(MAKE) format-check


