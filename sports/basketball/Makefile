# Reproducible pipeline for the NBA fantasy-archetype project (see docs/PROJECT_PLAN.md).
# Run from sports/basketball/. Stages are added as the project is built.

CONFIG ?= config/nba_archetypes.yaml

.PHONY: help setup fetch fetch-large test lint

help:
	@echo "Targets:"
	@echo "  setup        uv sync (shared env at repo root)"
	@echo "  fetch        pull NBA base data (sportsdataverse) -> data/raw"
	@echo "  fetch-large  also pull shots / boxscore (heavy)"
	@echo "  test / lint  pytest / ruff"
	@echo "  (build / archetypes / compose / predict stages: TODO — see PROJECT_PLAN)"

setup:
	cd ../../ && uv sync

fetch:
	uv run python scripts/fetch_data.py --config $(CONFIG)

fetch-large:
	uv run python scripts/fetch_data.py --config $(CONFIG) --include-large

test:
	cd ../../ && uv run pytest sports/basketball/tests

lint:
	cd ../../ && uv run ruff check sports/basketball
