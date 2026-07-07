"""Load experiment configs (``config/*.yaml``) with dotted-key access.

Configs are the single source of truth for an experiment (seasons, target, eligibility
grid, validation windows, models). No experiment numbers live in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .io import resolve


class Config:
    """Thin wrapper over a parsed YAML dict supporting ``cfg.get("a.b.c", default)``."""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = path

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = resolve(path)
        with open(p) as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data, path=p)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required config key: {dotted_key!r} in {self.path}")
        return value

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def seasons(self) -> list[int]:
        """Inclusive list of seasons [earliest_season .. latest_completed_season]."""
        start = int(self.require("data.earliest_season"))
        end = int(self.require("data.latest_completed_season"))
        return list(range(start, end + 1))

    def stem(self) -> str:
        """Cache-file stem: ``{sport}_{position}``, or ``{sport}_{position}_{cohort}`` when
        ``experiment.cohort`` isn't the default ``"returning"`` (e.g. a rookie config shares its
        ``position`` with the veteran config for that same position — QB/RB/WR/TE — so without
        this the two would collide on the same interim/processed filenames)."""
        sport = self.get("experiment.sport", "sport")
        position = self.require("experiment.position")
        cohort = self.get("experiment.cohort", "returning")
        base = f"{sport}_{position}"
        if cohort and cohort != "returning":
            base = f"{base}_{cohort}"
        return base.lower()

    def __repr__(self) -> str:
        name = self.get("experiment.name", "?")
        return f"Config(name={name!r}, path={self.path})"
