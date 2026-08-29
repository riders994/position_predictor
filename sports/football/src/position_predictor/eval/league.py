"""League settings — the roster shape and scoring format a draft board is built for.

A projection is a per-player number; a *draft board* is that number seen through a league. Two
settings decide what the board looks like:

* **scoring** (:mod:`position_predictor.scoring`) — changes the model's training target, so each
  format is its own dataset → features → fit. Formats are not interchangeable after the fact.
* **roster shape** — teams × started slots sets each position's replacement level, which is the
  only honest way to compare a QB to a WR. Two dedicated QB slots in a 10-team league makes ~20
  QBs starters instead of ~12, and the resulting board barely resembles the 1QB one.

Leagues live in ``config/leagues/*.yaml`` so a real league is a committed, diffable artifact
rather than a pile of CLI flags. They are read with the same :class:`~utils.config.Config` wrapper
as the experiment configs — no second config system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..scoring import RECEPTION_POINTS, normalize_scoring
from ..utils.config import Config

MODELED_POS = ("QB", "RB", "WR", "TE")
# A started slot may name a modeled position or the catch-all FLEX.
SLOT_KEYS = frozenset(MODELED_POS) | {"FLEX"}


@dataclass(frozen=True)
class LeagueConfig:
    """One league's draftable shape. ``starters`` includes QB (unlike the older keeper format
    strings) and may include ``FLEX``; ``flex_positions`` says who may fill those flex slots."""

    name: str
    label: str
    scoring: str
    teams: int
    starters: dict[str, int]
    flex_positions: tuple[str, ...] = ("RB", "WR", "TE")
    roster_size: int = 16
    bestball: bool = False
    top_n: dict[str, int] = field(default_factory=dict)   # explicit per-position depth overrides

    @property
    def total_picks(self) -> int:
        """Players drafted league-wide — the natural cut for a cross-position board."""
        return self.teams * self.roster_size

    @property
    def qb_starters(self) -> int:
        """The most QBs a team can start — dedicated QB slots plus a QB-eligible flex.

        Superflex leagues express the second QB as a flex slot, true 2QB leagues as a second
        dedicated slot; both let a manager start two, which is what moves the market board.
        """
        qb = self.starters.get("QB", 0)
        if "QB" in self.flex_positions:
            qb += self.starters.get("FLEX", 0)
        return qb

    @property
    def is_superflex(self) -> bool:
        """True when a team can start more than one QB (true 2QB *or* superflex).

        The market publishes one board for both shapes, so they share a benchmark even though
        their replacement levels differ (a 2QB league must fill both slots; superflex may punt).
        """
        return self.qb_starters >= 2

    def slot_summary(self) -> str:
        """``2QB / 2RB / 2WR / 1TE / 1FLEX`` — for report headers."""
        order = [*MODELED_POS, "FLEX"]
        return " / ".join(f"{self.starters[p]}{p}" for p in order if self.starters.get(p))


def league_from_dict(data: dict, *, name: str | None = None) -> LeagueConfig:
    """Validate a league mapping into a :class:`LeagueConfig`.

    Every failure names the offending key: a mis-typed roster silently produces a plausible but
    wrong board, which is worse than a crash.
    """
    name = str(data.get("name") or name or "league")

    scoring = normalize_scoring(data.get("scoring"))   # raises on an unknown format

    if "teams" not in data:
        raise ValueError(f"league {name!r}: missing required key 'teams'")
    teams = int(data["teams"])
    if not 4 <= teams <= 20:
        raise ValueError(f"league {name!r}: teams must be between 4 and 20; got {teams}")

    raw_starters = data.get("starters")
    if not isinstance(raw_starters, dict) or not raw_starters:
        raise ValueError(f"league {name!r}: 'starters' must be a non-empty mapping, "
                         f"e.g. {{QB: 1, RB: 2, WR: 3, TE: 1, FLEX: 1}}")
    starters: dict[str, int] = {}
    for key, value in raw_starters.items():
        slot = str(key).upper()
        if slot not in SLOT_KEYS:
            raise ValueError(f"league {name!r}: unknown starter slot {key!r}; "
                             f"expected one of {sorted(SLOT_KEYS)}")
        count = int(value)
        if count < 0:
            raise ValueError(f"league {name!r}: starter count for {slot} must be >= 0")
        starters[slot] = count
    for pos in MODELED_POS:
        starters.setdefault(pos, 0)
    starters.setdefault("FLEX", 0)
    if not any(starters[p] for p in MODELED_POS):
        raise ValueError(f"league {name!r}: no dedicated starter slots at any modeled position")

    # Defaults to TE-eligible (see keeper.FLEX_POS); RB/WR-only leagues say so.
    flex = tuple(str(p).upper() for p in data.get("flex_positions",
                                                  ("RB", "WR", "TE")))
    unknown = [p for p in flex if p not in MODELED_POS]
    if unknown:
        raise ValueError(f"league {name!r}: flex_positions {unknown} are not modeled "
                         f"(expected a subset of {list(MODELED_POS)})")
    if starters["FLEX"] and not flex:
        raise ValueError(f"league {name!r}: {starters['FLEX']} FLEX slot(s) but flex_positions "
                         f"is empty — nobody can fill them")

    roster_size = int(data.get("roster_size", 16))
    started = sum(starters.values())
    if roster_size < started:
        raise ValueError(f"league {name!r}: roster_size {roster_size} is smaller than the "
                         f"{started} started slots")

    top_n = {str(k).upper(): int(v) for k, v in (data.get("top_n") or {}).items()}
    bad_depth = [k for k in top_n if k not in MODELED_POS]
    if bad_depth:
        raise ValueError(f"league {name!r}: top_n keys {bad_depth} are not modeled positions")

    return LeagueConfig(
        name=name,
        label=str(data.get("label") or name),
        scoring=scoring,
        teams=teams,
        starters=starters,
        flex_positions=flex,
        roster_size=roster_size,
        bestball=bool(data.get("bestball", False)),
        top_n=top_n,
    )


def load_league(path) -> LeagueConfig:
    """Load one ``config/leagues/*.yaml`` (path is resolved relative to the project root)."""
    cfg = Config.load(path)
    stem = cfg.path.stem if cfg.path is not None else None
    return league_from_dict(cfg.data, name=stem)


def load_leagues(paths) -> list[LeagueConfig]:
    """Load several league YAMLs, rejecting duplicate names (they'd overwrite each other's
    report files)."""
    leagues = [load_league(p) for p in paths]
    seen: set[str] = set()
    for lg in leagues:
        if lg.name in seen:
            raise ValueError(f"duplicate league name {lg.name!r} — report files would collide")
        seen.add(lg.name)
    return leagues


# Re-exported so callers can validate a scoring name without reaching into two modules.
__all__ = ["LeagueConfig", "MODELED_POS", "RECEPTION_POINTS", "league_from_dict", "load_league",
           "load_leagues"]
