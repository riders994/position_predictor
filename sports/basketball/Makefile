# Reproducible pipeline for the NBA fantasy-archetype project (see docs/PROJECT_PLAN.md).
# Run from sports/basketball/. Stages are added as the project is built.

CONFIG ?= config/nba_archetypes.yaml

.PHONY: help setup fetch fetch-large build features archetypes fantasy compose fetch-yahoo compose-yahoo model-phase2 simulate test lint

help:
	@echo "Targets:"
	@echo "  setup        uv sync (shared env at repo root)"
	@echo "  fetch        pull NBA base data (sportsdataverse) -> data/raw"
	@echo "  fetch-large  also pull shots / boxscore (heavy)"
	@echo "  build        pivot long stats -> wide player-season -> data/interim"
	@echo "  features     engineer style features + z-scores -> data/processed"
	@echo "  archetypes   Phase 1: soft-GMM archetypes -> reports/REPORT_archetypes.md"
	@echo "  fantasy      Phase 2: max_pf optimal-lineup 9-cat success labels -> data/processed"
	@echo "  compose      Phase 2: roster archetype composition + join to success -> data/processed"
	@echo "  fetch-yahoo  Phase 2 aug: Yahoo season-long rosters + category-win labels (heavy, cached)"
	@echo "  compose-yahoo Phase 2 aug: Yahoo weeks-weighted composition + join -> data/processed"
	@echo "  model-phase2 Phase 2: composition -> 9-cat success model -> reports/REPORT_phase2_model.md"
	@echo "  simulate     Phase 2: draft sim + representation shootout -> reports/REPORT_phase2_simulation.md"
	@echo "  test / lint  pytest / ruff"
	@echo "  (predict stage: TODO — see PROJECT_PLAN)"

setup:
	cd ../../ && uv sync

fetch:
	uv run python scripts/fetch_data.py --config $(CONFIG)

fetch-large:
	uv run python scripts/fetch_data.py --config $(CONFIG) --include-large

build:
	uv run python scripts/build_dataset.py --config $(CONFIG)

features:
	uv run python scripts/build_features.py --config $(CONFIG)

archetypes:
	uv run python scripts/discover_archetypes.py --config $(CONFIG)

fantasy:
	uv run python scripts/fetch_fantasy.py --config $(CONFIG)

compose:
	uv run python scripts/compose.py --config $(CONFIG)

fetch-yahoo:
	uv run python scripts/fetch_yahoo.py --config $(CONFIG)

compose-yahoo:
	uv run python scripts/compose_yahoo.py --config $(CONFIG)

model-phase2:
	uv run python scripts/model_phase2.py --config $(CONFIG)

simulate:
	uv run python scripts/simulate_phase2.py --config $(CONFIG)

test:
	cd ../../ && uv run pytest sports/basketball/tests

lint:
	cd ../../ && uv run ruff check sports/basketball
