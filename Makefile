# Reproducible pipeline for position_predictor.
# Each target runs one stage of the experiment (see docs/PROJECT_PLAN.md §3).
# Default experiment config; override with `make experiment CONFIG=config/<other>.yaml`.

CONFIG ?= config/football_rb.yaml

.PHONY: help setup fetch build features eligibility eda benchmark experiment report clean test lint

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

test:
	uv run pytest

lint:
	uv run ruff check .
