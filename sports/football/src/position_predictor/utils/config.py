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

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """A deep copy of this config with the given dotted keys replaced.

        Serving tools re-point an experiment config at a different season or scoring format
        without touching the committed YAML, which stays the canonical description of the
        experiment. Intermediate dicts are created as needed, so a key the YAML never had
        (``target.scoring`` on an older config) can still be set.
        """
        import copy

        data = copy.deepcopy(self._data)
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            node = data
            for part in parts[:-1]:
                nxt = node.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[part] = nxt
                node = nxt
            node[parts[-1]] = value
        return Config(data, path=self.path)

    def seasons(self) -> list[int]:
        """Inclusive list of seasons [earliest_season .. latest_completed_season]."""
        start = int(self.require("data.earliest_season"))
        end = int(self.require("data.latest_completed_season"))
        return list(range(start, end + 1))

    def __repr__(self) -> str:
        name = self.get("experiment.name", "?")
        return f"Config(name={name!r}, path={self.path})"
