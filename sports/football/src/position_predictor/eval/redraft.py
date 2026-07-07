"""Redraft-league assistant — project a *new* season into a multi-position draft board.

Use case #2 (after the keeper tool). One entry point, :func:`run_redraft`, drives the three
steps a redraft user wants for the coming season:

(a) **gate** — :func:`position_predictor.data.availability.check_season_available` confirms the
    just-completed (feature) season is published before we model it (hard stop otherwise);
(b) **fetch** — refresh only the stale caches (the core datasets were materialised before the new
    season published) so the board sees the latest completed season;
(c) **project** — for each position fit the era ensemble on all labeled history and rank the
    latest feature season (reusing :func:`eval.projection.project_position`), then return the top
    board: by default top 32 QB / 110 RB / 130 WR / 30 TE (see :data:`DEFAULT_TOP_N`).

**QB/RB/WR/TE rookies are real projections, not a market-based gap.** A ``config/football_<pos>_
rookie.yaml`` predicts a rookie's OWN draft-year production from draft-day-known info (draft
capital, combine, landing spot — see ``data/rookie_build.py``), since a true incoming rookie has
no prior-NFL-season row to roll forward from the veteran pipeline. When both configs for a
position are passed in, its rookie board is concatenated with the veteran board and **re-ranked
together** by ``proj_ppg`` — a rookie projected at 12 ppg slots in at the right spot among
returning players, not as a separate list. K has no rookie model (rookie kickers are almost
always UDFA, draft-capital signal barely applies) and DST has no rookie concept at all (a team,
not a draftable individual) — both keep the older mechanism: model projections stay pure (ECR/
market never changes a model number), and we only *subtract a market-estimated rookie count* from
the list length so top-N still reflects roughly the right number of real draft slots.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime

from ..data.availability import AvailabilityReport, check_season_available, datasets_needing_refresh
from ..data.benchmark import ECR_URL, REDRAFT_OVERALL
from ..data.build import build_dataset
from ..data.rookie_build import build_dataset as build_rookie_dataset
from ..data.team_build import build_dataset as build_team_dataset
from ..eval.keeper import _norm
from ..eval.projection import project_position
from ..features.build import build_features
from ..utils.config import Config

# Default board depth per position. Sized for the user's 12-team/22-round (264-pick) RT Sports
# league: deep enough that each position's returned count is capped by the eligible-player
# universe itself, not by top_n (confirmed 2026-07-06: WR 111/130, RB 98/110, QB 31/32, TE 29/30
# requested/returned) — i.e. this is effectively the full board, not a truncated top-N. DST is
# all 32 teams — there's no "returning universe" concept to be capped by, so top_n=32 covers it
# outright.
DEFAULT_TOP_N = {"QB": 32, "RB": 110, "WR": 130, "TE": 30, "K": 24, "DST": 32}
# PFR draft-class position → our fantasy position. No rookie concept for DST (a team, not a
# draftable individual), so it's absent here on purpose — estimate_rookie_count degrades to 0
# for any position missing from this map.
_DRAFT_POS_TO_FANTASY = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "K": "K"}


@dataclass
class PositionSummary:
    position: str
    requested_top_n: int
    rookies_subtracted: int   # market-estimated rookies left as gaps (K only — no rookie model)
    returned: int
    proj_season: int
    rookies_modeled: int = 0  # real rookie-model rows merged into the board (QB/RB/WR/TE)


@dataclass
class RedraftResult:
    draft_season: int
    feature_season: int
    availability: AvailabilityReport
    refreshed: list[str] = field(default_factory=list)
    board: object = None  # combined tidy DataFrame (set on success)
    summaries: list[PositionSummary] = field(default_factory=list)
    rookie_note: str = ""

    @property
    def ready(self) -> bool:
        return self.availability.ok


def _override_season(config: Config, feature_season: int) -> Config:
    """A copy of ``config`` with ``data.latest_completed_season`` pinned to ``feature_season``.

    Keeps the committed YAML untouched while letting the build clamp the censoring boundary to the
    season we want as the live-board feature season (``data/build.py`` clamps to the min of this
    and the max season actually present)."""
    data = copy.deepcopy(config.data)
    data.setdefault("data", {})["latest_completed_season"] = int(feature_season)
    return Config(data, path=config.path)


def _rookie_market_context(draft_season: int):
    """Load the rookie class + the latest market board for ``draft_season``.

    Returns ``(ecr_df_or_None, rookie_names_by_pos, note)``. ``ecr_df`` is the latest
    redraft-overall ECR scrape dated in ``draft_season`` (any date — in early summer the
    just-before-kickoff window may not exist yet); ``rookie_names_by_pos`` maps our positions to
    the normalised names of that year's drafted players. Best-effort: any failure yields ``None``
    and a human note so the board falls back to the full top-N (no rookie adjustment).
    """
    import pandas as pd

    from ..utils.io import DATA_RAW

    rookie_names: dict[str, set[str]] = {"QB": set(), "RB": set(), "WR": set(), "TE": set(),
                                          "K": set()}
    try:
        dp = pd.read_parquet(DATA_RAW / "draft_picks.parquet", columns=["season", "position",
                                                                         "pfr_player_name"])
        klass = dp[dp["season"] == draft_season]
        for _, r in klass.iterrows():
            pos = _DRAFT_POS_TO_FANTASY.get(str(r["position"]).upper())
            if pos:
                rookie_names[pos].add(_norm(r["pfr_player_name"]))
    except Exception as exc:  # noqa: BLE001 — no draft class → no rookie adjustment
        return None, rookie_names, f"draft class unavailable ({type(exc).__name__})"

    if not any(rookie_names.values()):
        return None, rookie_names, f"no {draft_season} draft class cached (run with refresh)"

    try:
        ecr = pd.read_parquet(ECR_URL, columns=["player", "pos", "ecr", "ecr_type", "scrape_date"])
        ecr = ecr[ecr["ecr_type"] == REDRAFT_OVERALL].copy()
        ecr["scrape_date"] = pd.to_datetime(ecr["scrape_date"], errors="coerce")
        ecr = ecr.dropna(subset=["scrape_date", "ecr"])
        yr = ecr[ecr["scrape_date"].dt.year == draft_season]
        if yr.empty:
            return None, rookie_names, f"no {draft_season} market board published yet"
        latest = yr[yr["scrape_date"] == yr["scrape_date"].max()].copy()
        return latest, rookie_names, ""
    except Exception as exc:  # noqa: BLE001 — market source down → no rookie adjustment
        return None, rookie_names, f"market board unavailable ({type(exc).__name__})"


def estimate_rookie_count(ecr_df, rookie_names_by_pos, position: str, top_n: int) -> int:
    """How many rookies the market expects inside the position's top ``top_n``.

    Market is used **only to count rookies** — never to rank or alter the model's returning
    players. Returns 0 when no market board is available (full top-N is then returned).
    """
    if ecr_df is None:
        return 0
    rookies = rookie_names_by_pos.get(position.upper(), set())
    if not rookies:
        return 0
    pos_board = ecr_df[ecr_df["pos"] == position.upper()].sort_values("ecr").head(top_n)
    return int(sum(_norm(p) in rookies for p in pos_board["player"]))


def run_redraft(configs, *, draft_season=None, refresh: bool = True, top_n=None,
                rookie_context=None) -> RedraftResult:
    """Run the full redraft workflow for the given position configs. See module docstring."""
    import pandas as pd

    from ..data.fetch import fetch_all

    configs = list(configs)
    if not configs:
        raise ValueError("run_redraft needs at least one position config")
    top_n = {**DEFAULT_TOP_N, **(top_n or {})}

    # Split by cohort — a position may have both a veteran config and a rookie config passed in;
    # the rookie board gets merged into the veteran board (re-ranked together) rather than kept
    # as a separate list. See module docstring.
    veteran_cfgs, rookie_cfgs = [], {}
    for cfg in configs:
        pos = cfg.require("experiment.position").upper()
        if cfg.get("experiment.cohort") == "rookie":
            rookie_cfgs[pos] = cfg
        else:
            veteran_cfgs.append(cfg)
    if not veteran_cfgs:
        raise ValueError("run_redraft needs at least one non-rookie position config")

    base = veteran_cfgs[0]
    draft_season = int(draft_season or datetime.now().year)
    horizon = int(base.get("target.predict_horizon", 1))
    feature_season = draft_season - horizon
    earliest = int(base.require("data.earliest_season"))

    # (a) gate -------------------------------------------------------------------------------
    report = check_season_available(feature_season)
    result = RedraftResult(draft_season=draft_season, feature_season=feature_season,
                           availability=report)
    if not report.ok:
        return result  # caller hard-stops; nothing else is safe to do

    # (b) fetch only what is stale ----------------------------------------------------------
    if refresh:
        stale = datasets_needing_refresh(feature_season)
        if stale:
            fetch_all(list(range(earliest, draft_season + 1)), datasets=stale, overwrite=True)
        result.refreshed = stale

    # rookie context (shared across positions) ---------------------------------------------
    if rookie_context is None:
        rookie_context = _rookie_market_context(draft_season)
    ecr_df, rookie_names, note = rookie_context
    result.rookie_note = note

    # (c) project each position -------------------------------------------------------------
    frames, summaries = [], []
    for cfg in veteran_cfgs:
        position = cfg.require("experiment.position").upper()
        cfg2 = _override_season(cfg, feature_season)
        (build_team_dataset if position == "DST" else build_dataset)(cfg2, write=True)
        build_features(cfg2, write=True)
        proj = project_position(cfg2, write=False)
        if proj.empty:
            continue
        proj_season = int(proj["proj_season"].iloc[0])
        if proj_season != draft_season:
            raise RuntimeError(
                f"{position}: projected season {proj_season} != draft season {draft_season} "
                f"(feature season {feature_season} not the latest available — data not refreshed?)")

        rookie_cfg = rookie_cfgs.get(position)
        rookie_proj = pd.DataFrame()
        if rookie_cfg is not None:
            # No _override_season here: the rookie build's own "is this draft class's season
            # complete yet" check reads target.latest_completed_season straight off the config
            # (pinning it to draft_season would wrongly mark the incoming class "complete").
            build_rookie_dataset(rookie_cfg, write=True)
            build_features(rookie_cfg, write=True)
            rookie_proj = project_position(rookie_cfg, feature_season=draft_season, write=False)
            if not rookie_proj.empty:
                r_season = int(rookie_proj["proj_season"].iloc[0])
                if r_season != draft_season:
                    raise RuntimeError(
                        f"{position} rookie: projected season {r_season} != draft season "
                        f"{draft_season} (this year's draft class not cached yet?)")

        if rookie_proj.empty:
            # No rookie model for this position (K), or the current draft class isn't cached yet
            # — fall back to the market-estimated-count mechanism so top-N still leaves roughly
            # the right number of slots for rookies the model can't score.
            n = top_n.get(position, len(proj))
            rookies = estimate_rookie_count(ecr_df, rookie_names, position, n)
            keep = max(n - rookies, 0)
            top = proj.head(keep).copy()
            top["requested_top_n"] = n
            top["rookies_subtracted"] = rookies
            frames.append(top)
            summaries.append(PositionSummary(position, n, rookies, len(top), proj_season))
            continue

        # Real rookie projections exist — merge with the veteran board and re-rank together by
        # proj_ppg, so a rookie slots in at the value-correct spot instead of a separate list.
        combined = pd.concat([proj, rookie_proj], ignore_index=True)
        combined = combined.sort_values("proj_ppg", ascending=False).reset_index(drop=True)
        combined["proj_pos_rank"] = combined["proj_ppg"].rank(
            ascending=False, method="min").astype(int)

        n = top_n.get(position, len(combined))
        keep = min(n, len(combined))
        top = combined.head(keep).copy()
        rookies_modeled = int(top["player_id"].isin(rookie_proj["player_id"]).sum())
        top["requested_top_n"] = n
        top["rookies_subtracted"] = 0
        frames.append(top)
        summaries.append(PositionSummary(position, n, 0, len(top), proj_season,
                                         rookies_modeled=rookies_modeled))

    result.board = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    result.summaries = summaries
    return result
