# Reproducible pipeline for position_predictor.
# Each target runs one stage of the experiment (see docs/PROJECT_PLAN.md §3).
# Default experiment config; override with `make experiment CONFIG=config/<other>.yaml`.

CONFIG ?= config/football_rb.yaml

.PHONY: help setup fetch build features eligibility eda benchmark experiment report progress project redraft postseason handcuff clean test lint

help:
	@echo "Targets:"
	@echo "  setup        uv sync (env from pyproject/uv.lock)"
	@echo "  fetch        pull nflverse data -> data/raw"
	@echo "  build        join sources -> player-season dataset"
	@echo "  features     engineer features -> data/processed"
	@echo "  eligibility  derive games/snap cutoff (analysis + chosen rule)"
	@echo "  eda          coverage / distribution / target-stability tables + figures"
	@echo "  benchmark    fetch preseason ADP/ECR market benchmark -> data/external"
	@echo "  experiment   walk-forward CV x {10,20,30}yr x models"
	@echo "  report       build metrics tables + figures -> reports/"
	@echo "  progress     cross-version progress report -> reports/versions/"
	@echo "  project      project upcoming season -> reports/projections_<stem>.csv"
	@echo "  redraft      check/fetch/project new-season draft board (QB/RB/WR) -> reports/redraft_<season>.csv"
	@echo "  postseason   grade model/ECR/ADP vs actual finish -> reports/postseason_<season>.{md,csv}"
	@echo "  handcuff     rank RB backups to draft by starter injury risk -> reports/handcuff_<season>.{md,csv}"
	@echo "  test / lint  pytest / ruff"

setup:
	uv sync --all-extras

fetch:
	uv run python scripts/fetch_data.py --config $(CONFIG)

build:
	uv run python scripts/build_dataset.py --config $(CONFIG)

features:
	uv run python scripts/build_features.py --config $(CONFIG)

eligibility:
	uv run python scripts/run_eligibility.py --config $(CONFIG)

eda:
	uv run python scripts/run_eda.py --config $(CONFIG)

benchmark:
	uv run python scripts/fetch_benchmark.py --config $(CONFIG)

experiment:
	uv run python scripts/run_experiment.py --config $(CONFIG)

report:
	uv run python scripts/make_report.py --config $(CONFIG)

progress:
	uv run python scripts/make_progress.py --config $(CONFIG)

project:
	uv run python scripts/project.py --config $(CONFIG)

# Redraft draft board across all positions; override season with `make redraft SEASON=2026`.
redraft:
	uv run python scripts/redraft.py $(if $(SEASON),--season $(SEASON),)

# Postseason report grading model/ECR/ADP vs actuals; `make postseason SEASON=2024`.
postseason:
	uv run python scripts/postseason.py $(if $(SEASON),--season $(SEASON),)

# Handcuff board — RB backups to draft by starter injury risk; `make handcuff SEASON=2026`.
handcuff:
	uv run python scripts/handcuff.py $(if $(SEASON),--season $(SEASON),)

test:
	uv run pytest

lint:
	uv run ruff check .
