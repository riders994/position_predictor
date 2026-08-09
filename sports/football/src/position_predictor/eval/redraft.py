"""Redraft-league assistant — project a *new* season into a multi-position draft board.

Use case #2 (after the keeper tool). One entry point, :func:`run_redraft`, drives the three
steps a redraft user wants for the coming season:

(a) **gate** — :func:`position_predictor.data.availability.check_season_available` confirms the
    just-completed (feature) season is published before we model it (hard stop otherwise);
(b) **fetch** — refresh only the stale caches (the core datasets were materialised before the new
    season published) so the board sees the latest completed season;
(c) **project** — for each position fit the era ensemble on all labeled history and rank the
    latest feature season (reusing :func:`eval.projection.project_position`);
(d) **value** — turn those per-position projections into one board *per league*
    (:mod:`eval.league`), using the keeper tool's VORP engine so positions are comparable.

**Leagues, not one board.** A projection is a number; a draft board is that number seen through
a league's scoring and roster shape. Scoring changes the model's training target, so each format
is a separate dataset → features → fit — the loop below is therefore keyed on *scoring*, and two
leagues that share a format share one pipeline pass. Roster shape only affects replacement
levels, which is pure post-processing.

**Top-N is "returning players within the top N", not N players.** The model only ranks returning
players (rookies are excluded by ``scope: returning_only``), so a real top-20 board includes a
few incoming rookies the model can't score. We keep model projections pure — ECR/market never
changes a model number (the "two independent opinions" rule) — and only *subtract an estimated
rookie count* from the list length: if the market expects 3 rookie QBs inside the top 20, we
return the top 17 returning QBs.

**Board depth follows the league.** A fixed top-20 QB list is meaningless in a 10-team 2QB league
where 20 QBs are *starters*; depth is derived from teams × started slots (see :func:`board_depth`)
and floored at the historical defaults so a 1QB board never gets shallower than it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..data.availability import AvailabilityReport, check_season_available, datasets_needing_refresh
from ..data.benchmark import ECR_URL, REDRAFT_OVERALL
from ..data.build import build_dataset
from ..eval.keeper import _norm, build_board
from ..eval.league import MODELED_POS, LeagueConfig, league_from_dict
from ..eval.projection import project_position
from ..features.build import build_features
from ..scoring import DEFAULT_SCORING
from ..utils.config import Config

# Floor for board depth per position. TE is shallow — a top-24 board covers the startable
# TE1/TE2 tiers. League-derived depth (:func:`board_depth`) can only go deeper than these.
DEFAULT_TOP_N = {"QB": 20, "RB": 50, "WR": 75, "TE": 24}
# Draftable players per started slot. 1.75 keeps roughly one backup for every starter plus the
# waiver-worthy tail, which is the depth a draft board actually needs to cover.
DEPTH_PER_STARTER = 1.75
# The league assumed when a caller doesn't supply one — the 12-team 1QB PPR shape every redraft
# board used before league configs existed, so the default output is unchanged.
DEFAULT_LEAGUE = {"name": "ppr_1qb", "label": "12-team 1QB PPR", "scoring": DEFAULT_SCORING,
                  "teams": 12, "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
                  "flex_positions": ["RB", "WR", "TE"], "roster_size": 16}
# PFR draft-class position → our fantasy position.
_DRAFT_POS_TO_FANTASY = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}


@dataclass
class PositionSummary:
    position: str
    requested_top_n: int
    rookies_subtracted: int
    returned: int
    proj_season: int


@dataclass
class LeagueBoard:
    """One league's finished draft board."""

    league: LeagueConfig
    board: object = None                  # tidy DataFrame incl. vorp + proj_overall_rank
    replacement: dict = field(default_factory=dict)
    starters: dict = field(default_factory=dict)
    summaries: list[PositionSummary] = field(default_factory=list)
    bestball_lambdas: dict = field(default_factory=dict)


@dataclass
class RedraftResult:
    draft_season: int
    feature_season: int
    availability: AvailabilityReport
    refreshed: list[str] = field(default_factory=list)
    board: object = None  # first league's board (back-compat with the single-board callers)
    summaries: list[PositionSummary] = field(default_factory=list)
    rookie_note: str = ""
    leagues: list[LeagueBoard] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.availability.ok


def board_depth(league: LeagueConfig, *, overrides=None) -> dict:
    """Per-position board depth for ``league``: teams × started slots × a bench allowance.

    Floored at :data:`DEFAULT_TOP_N` so no board is shallower than the historical defaults, then
    scaled up if the positions together still wouldn't cover every pick in the draft — an 18-round
    Underdog draft spends 216 picks, and a board that stops at 170 leaves the last three rounds
    unguided. Explicit overrides (the league's own ``top_n``, then the ``--top-*`` flags) are
    applied last and are never rescaled.
    """
    flex_share = (league.starters.get("FLEX", 0) / len(league.flex_positions)
                  if league.flex_positions else 0)
    depth = {}
    for pos in MODELED_POS:
        slots = league.starters.get(pos, 0) + (flex_share if pos in league.flex_positions else 0)
        derived = round(league.teams * slots * DEPTH_PER_STARTER)
        depth[pos] = max(DEFAULT_TOP_N.get(pos, 0), derived)
    total = sum(depth.values())
    if total and total < league.total_picks:
        scale = league.total_picks / total
        depth = {pos: round(n * scale) for pos, n in depth.items()}
    depth.update({k.upper(): int(v) for k, v in (league.top_n or {}).items()})
    depth.update({k.upper(): int(v) for k, v in (overrides or {}).items()})
    return depth


def _override_season(config: Config, feature_season: int) -> Config:
    """A copy of ``config`` with ``data.latest_completed_season`` pinned to ``feature_season``.

    Keeps the committed YAML untouched while letting the build clamp the censoring boundary to the
    season we want as the live-board feature season (``data/build.py`` clamps to the min of this
    and the max season actually present)."""
    return config.with_overrides({"data.latest_completed_season": int(feature_season)})


def _override_scoring(config: Config, scoring: str) -> Config:
    """A copy of ``config`` with ``target.scoring`` pinned to ``scoring``.

    Same trick as :func:`_override_season`: the committed per-position YAML stays as the canonical
    PPR experiment, while a run can build any format's dataset/features/model beside it (artifacts
    are namespaced by :func:`utils.naming.artifact_stem`, so nothing is overwritten).
    """
    return config.with_overrides({"target.scoring": scoring})


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

    rookie_names: dict[str, set[str]] = {"QB": set(), "RB": set(), "WR": set(), "TE": set()}
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


def _project_scoring(configs, *, scoring, feature_season, draft_season, top_n, ecr_df,
                     rookie_names):
    """Build → feature → project every position under one scoring format.

    Returns ``(board_df, summaries)``. This is the whole expensive part of a redraft run, which is
    why the caller runs it once per *format* rather than once per league.
    """
    import pandas as pd

    frames, summaries = [], []
    for cfg in configs:
        position = cfg.require("experiment.position").upper()
        cfg2 = _override_scoring(_override_season(cfg, feature_season), scoring)
        build_dataset(cfg2, write=True)
        build_features(cfg2, write=True)
        proj = project_position(cfg2, write=False)
        if proj.empty:
            continue
        proj_season = int(proj["proj_season"].iloc[0])
        if proj_season != draft_season:
            raise RuntimeError(
                f"{position}: projected season {proj_season} != draft season {draft_season} "
                f"(feature season {feature_season} not the latest available — data not refreshed?)")
        n = top_n.get(position, len(proj))
        rookies = estimate_rookie_count(ecr_df, rookie_names, position, n)
        keep = max(n - rookies, 0)
        top = proj.head(keep).copy()
        top["requested_top_n"] = n
        top["rookies_subtracted"] = rookies
        frames.append(top)
        summaries.append(PositionSummary(position, n, rookies, len(top), proj_season))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), summaries


def _attach_bestball(proj, league, *, feature_season, lambdas=None):
    """Add ``sigma`` / ``bestball_ppg`` for a best-ball league. See :mod:`eval.bestball` — the
    fitted upside weight is zero, so this is reporting context unless a lambda is passed."""
    import pandas as pd

    from ..utils.io import DATA_RAW
    from .bestball import (DEFAULT_LAMBDAS, apply_bestball, projected_sigma, season_volatility,
                           weekly_fantasy_points)

    lambdas = {**DEFAULT_LAMBDAS, **(lambdas or {})}
    weekly = pd.read_parquet(DATA_RAW / "weekly.parquet",
                             columns=["player_id", "season", "week", "season_type", "position",
                                      "position_group", "fantasy_points", "receptions"])
    weekly_pts = weekly_fantasy_points(weekly, scoring=league.scoring)
    sigma = projected_sigma(season_volatility(weekly_pts), board_season=feature_season + 1)
    return apply_bestball(proj, sigma, lambdas), lambdas


def run_redraft(configs, *, draft_season=None, refresh: bool = True, top_n=None,
                rookie_context=None, leagues=None, bestball_lambdas=None) -> RedraftResult:
    """Run the full redraft workflow for the given position configs. See module docstring.

    ``leagues`` is an iterable of :class:`~eval.league.LeagueConfig`; omitted, it defaults to the
    12-team 1QB PPR shape so existing callers get exactly the board they got before. ``top_n``
    overrides the league-derived :func:`board_depth` per position.
    """
    from ..data.fetch import fetch_all

    configs = list(configs)
    if not configs:
        raise ValueError("run_redraft needs at least one position config")
    leagues = list(leagues) if leagues else [league_from_dict(DEFAULT_LEAGUE)]

    base = configs[0]
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

    # (c) project — once per *scoring format*, shared by every league that uses it -----------
    # Depth is per-league, so a format's pass uses the deepest board any of its leagues asks for;
    # each league then trims to its own depth in (d). Projecting deeper costs nothing extra.
    depths = {lg.name: board_depth(lg, overrides=top_n) for lg in leagues}
    projections = {}
    for scoring in sorted({lg.scoring for lg in leagues}):
        pooled = {p: max(depths[lg.name].get(p, 0) for lg in leagues if lg.scoring == scoring)
                  for p in MODELED_POS}
        projections[scoring] = _project_scoring(
            configs, scoring=scoring, feature_season=feature_season, draft_season=draft_season,
            top_n=pooled, ecr_df=ecr_df, rookie_names=rookie_names)

    # (d) value each league ------------------------------------------------------------------
    for lg in leagues:
        proj, summaries = projections[lg.scoring]
        if proj is None or proj.empty:
            result.leagues.append(LeagueBoard(league=lg))
            continue
        depth = depths[lg.name]
        # proj_pos_rank is 1-based within position, so the league's depth is a plain row filter.
        limit = proj["position"].map(depth).fillna(len(proj))
        trimmed = proj[proj["proj_pos_rank"] <= limit].reset_index(drop=True)
        lambdas = {}
        value_col = "proj_ppg"
        if lg.bestball:
            trimmed, lambdas = _attach_bestball(trimmed, lg, feature_season=feature_season,
                                                lambdas=bestball_lambdas)
            if any(lambdas.get(p) for p in MODELED_POS):
                value_col = "bestball_ppg"
        board, replacement, starters = build_board(
            trimmed, teams=lg.teams, roster=lg.starters, flex_positions=lg.flex_positions,
            value_col=value_col)
        board = board.head(lg.total_picks).reset_index(drop=True)
        board["league"] = lg.name
        # Summaries describe what each league actually kept, not the pooled projection pass.
        lg_summaries = [
            PositionSummary(s.position, depth.get(s.position, s.requested_top_n),
                            s.rookies_subtracted,
                            int((board["position"] == s.position).sum()), s.proj_season)
            for s in summaries]
        result.leagues.append(LeagueBoard(league=lg, board=board, replacement=replacement,
                                          starters=starters, summaries=lg_summaries,
                                          bestball_lambdas=lambdas))

    first = result.leagues[0] if result.leagues else None
    if first is not None:
        result.board = first.board
        result.summaries = first.summaries
    return result


def render_markdown(result: RedraftResult, lb: LeagueBoard, *, top: int = 60) -> str:
    """Render one league's draft board to markdown (the same content the CSV carries)."""
    lg = lb.league
    lines = [f"# Draft Board — {result.draft_season} · {lg.label}", ""]
    lines.append(f"_{lg.teams}-team · **{lg.scoring.replace('_', ' ')}** · {lg.slot_summary()} · "
                 f"{lg.roster_size} rounds ({lg.total_picks} picks). Features from "
                 f"{result.feature_season}._")
    lines.append("")
    lines.append("_Model-only: ECR/ADP are benchmarks and never inputs. The market is used for "
                 "one thing — counting how many rookies belong in each top-N, since the model "
                 "only ranks returning players._")
    lines.append("")

    if lb.board is None or lb.board.empty:
        lines.append("_No projections produced (missing features?)._")
        return "\n".join(lines) + "\n"

    board = lb.board
    if lg.bestball:
        lines.append("## Best ball: upside is not priced in — on purpose")
        lines.append("")
        lam = lb.bestball_lambdas or {}
        if any(lam.get(p) for p in MODELED_POS):
            lines.append(f"Upside weight applied: "
                         f"{', '.join(f'{p} {lam.get(p, 0):.2f}' for p in MODELED_POS)} "
                         f"(passed explicitly, not the fitted default).")
        else:
            lines.append("Fitting the upside weight against 13 walk-forward season pairs "
                         "(2012-2024) returns **zero at every position**: the Spearman-vs-lambda "
                         "curve is flat to ~0.003 and no position clears p<0.05. Weekly sigma is "
                         "~0.90 rank-correlated with weekly mean at RB/WR/TE, so volatility is "
                         "mostly a restatement of quality. **Rank by projected PPG; don't pay up "
                         "for spike-week reputations.** `sigma` below is context, not a tiebreak.")
        lines.append("")

    lines.append("## Replacement level")
    lines.append("")
    lines.append("_The first non-starter at each position once every league-wide slot (dedicated "
                 "+ flex) is filled. VORP = projected PPG − this._")
    lines.append("")
    lines.append("| position | started league-wide | replacement PPG |")
    lines.append("|---|---|---|")
    for pos in MODELED_POS:
        if pos in lb.replacement:
            lines.append(f"| {pos} | {lb.starters.get(pos, 0)} | {lb.replacement[pos]:.2f} |")
    lines.append("")

    lines.append(f"## Overall board (top {min(top, len(board))} of {len(board)})")
    lines.append("")
    has_sigma = "sigma" in board.columns
    header = "| # | player | pos | pos rank | proj PPG | VORP |"
    divider = "|---|---|---|---|---|---|"
    if has_sigma:
        header += " sigma |"
        divider += "---|"
    lines.extend([header, divider])
    for _, r in board.head(top).iterrows():
        row = (f"| {int(r['proj_overall_rank'])} | {r['player_name']} | {r['position']} | "
               f"{r['position']}{int(r['proj_pos_rank'])} | {r['proj_ppg']:.2f} | "
               f"{r['vorp']:.2f} |")
        if has_sigma:
            row += f" {r['sigma']:.2f} |"
        lines.append(row)
    lines.append("")

    lines.append("## By position")
    lines.append("")
    for s in lb.summaries:
        sub = board[board["position"] == s.position]
        if sub.empty:
            continue
        note = (f" (top {s.requested_top_n} − {s.rookies_subtracted} rookies)"
                if s.rookies_subtracted else f" (top {s.requested_top_n})")
        lines.append(f"### {s.position} — {len(sub)} returning players{note}")
        lines.append("")
        lines.append("| pos rank | player | proj PPG | VORP | overall |")
        lines.append("|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['proj_pos_rank'])} | {r['player_name']} | "
                         f"{r['proj_ppg']:.2f} | {r['vorp']:.2f} | "
                         f"{int(r['proj_overall_rank'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"
