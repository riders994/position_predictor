"""Keeper valuation — turn per-position projections into a cross-position keeper board.

A keeper costs you the **pick you drafted the player at**. So a keep is good when the player's
projected draft value beats that pick. To compare across positions and against a pick number we
need one currency: **value over replacement (VORP/VBD)**. We compute a replacement level per
position from league settings (teams × started slots, flex filled by the best remaining RB/WR),
subtract it from each projection, and rank everyone by VORP → a projected draft board. Then

    surplus = (pick you'd pay) − (player's projected overall board slot)

ranks the keepers: a player you keep at pick 100 who projects as a top-30 board slot is a +70
steal. This is model-only — ECR stays a benchmark, never blended in.

Scope: QB/RB/WR/TE (the modeled positions). K/DST and unmatched names are reported as *unscored*.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Per-team started slots. ``--format`` on the keeper CLI is shorthand for a QB slot count:
# superflex is the fractional 1.7 heuristic (a QB fills the flex most, but not all, of the time).
# Leagues configured in ``config/leagues/*.yaml`` set every slot explicitly instead — see
# :mod:`position_predictor.eval.league`.
QB_SLOTS_PER_TEAM = {"1qb": 1.0, "sf": 1.7, "2qb": 2.0}
DEFAULT_ROSTER = {"RB": 2, "WR": 2, "TE": 1, "FLEX": 1}   # excludes QB (set by format)
# TE is flex-eligible by default: every league in config/leagues/ runs a TE-eligible flex, and it
# is the common modern setting. It matters — an elite TE winning flex slots pushes the league
# deeper into the TE pool, which lowers TE replacement level and lifts every TE's VORP. Leagues
# whose flex is RB/WR-only must say so explicitly (`flex_positions: [RB, WR]`).
FLEX_POS = ("RB", "WR", "TE")
MODELED_POS = ("QB", "RB", "WR", "TE")


def roster_from_format(fmt, roster=None) -> dict:
    """Fold the keeper CLI's ``1qb|sf|2qb`` shorthand into a full ``roster`` dict incl. QB."""
    fmt = str(fmt).lower()
    if fmt not in QB_SLOTS_PER_TEAM:
        raise ValueError(f"format must be one of {sorted(QB_SLOTS_PER_TEAM)}; got {fmt!r}")
    return {**(roster or DEFAULT_ROSTER), "QB": QB_SLOTS_PER_TEAM[fmt]}


def replacement_levels(proj, *, teams, roster=None, fmt=None, flex_positions=FLEX_POS):
    """Projected-PPG replacement level per position for the given league.

    Replacement = the projected points of the **first non-starter** at each position once every
    started slot (dedicated + flex) is filled league-wide. Returns ``(replacement, starters)``.

    ``roster`` is per-team started slots **including QB** (fractional counts allowed, for the
    superflex heuristic). Passing ``fmt`` instead supplies the QB count from the ``1qb|sf|2qb``
    shorthand — the keeper CLI's interface, kept working. ``flex_positions`` are the positions
    eligible for the ``FLEX`` slots: most leagues run RB/WR, Underdog also allows TE, and that
    alone moves the TE board.
    """
    if fmt is not None:
        roster = roster_from_format(fmt, roster)
    elif roster is None:
        raise ValueError("replacement_levels needs either a roster dict or a fmt shorthand")
    flex_positions = tuple(flex_positions)
    unknown = [p for p in flex_positions if p not in MODELED_POS]
    if unknown:
        raise ValueError(f"flex_positions {unknown} are not modeled positions {list(MODELED_POS)}")
    pools = {p: sorted(proj.loc[proj["position"] == p, "proj_ppg"], reverse=True)
             for p in MODELED_POS}
    starters = {p: round(teams * roster.get(p, 0)) for p in MODELED_POS}
    # Flex: hand each slot to whichever eligible position's next-best player is higher.
    idx = {p: starters[p] for p in flex_positions}
    for _ in range(round(teams * roster.get("FLEX", 0))):
        cand = {p: pools[p][idx[p]] for p in flex_positions if idx[p] < len(pools[p])}
        if not cand:
            break
        best = max(cand, key=cand.get)
        starters[best] += 1
        idx[best] += 1
    replacement = {}
    for p, pool in pools.items():
        s = starters[p]
        replacement[p] = pool[s] if s < len(pool) else (pool[-1] if pool else 0.0)
    return replacement, starters


def build_board(proj, *, teams, roster=None, fmt=None, flex_positions=FLEX_POS,
                value_col="proj_ppg"):
    """Add ``vorp`` + ``proj_overall_rank`` (1 = best board value). Returns
    ``(board, replacement, starters)`` with the board sorted by VORP.

    ``value_col`` is the currency VORP is computed in — ``proj_ppg`` normally, or a format-
    adjusted value such as bestball's upside-weighted PPG. Replacement is measured in the *same*
    currency: comparing an upside-weighted number against a mean-PPG replacement level would add
    the same bias to every player and change nothing but the scale.
    """
    board = proj.copy()
    pools_src = board if value_col == "proj_ppg" else board.rename(
        columns={"proj_ppg": "_mean_ppg", value_col: "proj_ppg"})
    replacement, starters = replacement_levels(pools_src, teams=teams, roster=roster, fmt=fmt,
                                               flex_positions=flex_positions)
    board["vorp"] = (board[value_col] - board["position"].map(replacement).fillna(0.0)).round(2)
    # VORP is rounded to 2dp, so ties are common (~1 pair per 20 players on a real board). Sorting
    # on VORP alone leaves them to pandas' non-stable quicksort, which reorders tied players
    # between otherwise identical runs. Break ties explicitly — higher raw projection first, then
    # player_id — so a board is reproducible and two runs diff cleanly.
    order = [("vorp", False), (value_col, False), ("player_id", True)]
    seen: set[str] = set()
    keys = [(c, asc) for c, asc in order
            if c in board.columns and not (c in seen or seen.add(c))]
    board = board.sort_values([c for c, _ in keys], ascending=[asc for _, asc in keys])
    board = board.reset_index(drop=True)
    board["proj_overall_rank"] = range(1, len(board) + 1)
    return board, replacement, starters


# A generational suffix only counts as one in *trailing* position. Unanchored, the `v`
# alternative eats a standalone "v" anywhere in the name: "V Jones" normalised to "jones", a bare
# surname that then fuzzy-matches any Jones on the board. Anchoring keeps "Sammy Watkins IV" ->
# "sammy watkins" while leaving "Robert V Smith" intact.
_SUFFIX = re.compile(r"\s*\b(jr|sr|ii|iii|iv|v)\b\s*$")
# Runs of single-letter tokens are one initialled name: "a j brown" -> "aj brown", so a typed
# "AJ Brown" reaches "A.J. Brown" by *exact* match. It matched before only via the fuzzy pass at
# ~0.94, and `get_close_matches(n=1)` returns the best candidate over the cutoff rather than a
# safe one — so an initialled name was competing on ratio against the whole board and could
# resolve to the wrong row. Collapsing removes it from fuzzy contention entirely.
_INITIAL_RUN = re.compile(r"\b([a-z])\s+(?=[a-z]\b)")


def _norm(name) -> str:
    """Normalise a player name for matching: lowercase, drop punctuation, fold runs of initials,
    and drop a trailing generational suffix.

    NB ``jr`` and ``sr`` both drop, so a father/son pair collapses to one key and would match
    *exactly* — no cutoff guards it. Deliberate: dropping the suffix is what lets a typed "Brian
    Thomas" find "Brian Thomas Jr.". Harmless while the pool holds no such pair (verified: zero
    collisions across the 616-row board), and cheaper as a known edge than as a rule every reader
    of this function has to carry.
    """
    s = str(name).lower().replace(".", " ").replace("'", "").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    prev = None
    while prev != s:                      # "a b c smith" needs more than one pass
        prev, s = s, _INITIAL_RUN.sub(r"\1", s)
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def resolve_players(board, picks):
    """Match input ``picks`` (cols: player, pick, optional position) to board rows by name.

    Exact normalised match first, then a conservative fuzzy fallback. Returns
    ``(matched_df, unmatched_df)``; ``matched`` carries the board columns + ``pick``.
    """
    import difflib

    import pandas as pd

    by_norm = {}
    for _, r in board.iterrows():
        by_norm.setdefault(_norm(r["player_name"]), r)
    norms = list(by_norm)

    matched, unmatched = [], []
    for _, p in picks.iterrows():
        key = _norm(p["player"])
        row = by_norm.get(key)
        if row is None:
            near = difflib.get_close_matches(key, norms, n=1, cutoff=0.88)
            row = by_norm[near[0]] if near else None
        if row is None:
            unmatched.append({"player": p["player"], "pick": p.get("pick"),
                              "reason": "no projection (K/DST or name not matched)"})
            continue
        rec = row.to_dict()
        rec["pick"] = p["pick"]
        rec["input_name"] = p["player"]
        matched.append(rec)
    return pd.DataFrame(matched), pd.DataFrame(unmatched)


def evaluate_keepers(board, picks):
    """Attach pick cost + ``surplus`` (= pick − projected board slot) and sort best-first."""
    matched, unmatched = resolve_players(board, picks)
    if matched.empty:
        return matched, unmatched
    matched["surplus"] = matched["pick"].astype(float) - matched["proj_overall_rank"]
    matched["pos_rank"] = matched["position"] + matched["proj_pos_rank"].astype(int).astype(str)
    matched["keep"] = matched["surplus"] > 0
    return matched.sort_values("surplus", ascending=False).reset_index(drop=True), unmatched


@dataclass
class KeeperResult:
    """Everything one keeper run produced — the inputs to :func:`render_markdown`."""

    season: int
    label: str                                  # human league description for the header
    teams: int
    ranked: object = None                       # keepers with pick/surplus/keep, best-first
    unmatched: object = None                    # names with no projection
    replacement: dict = field(default_factory=dict)
    starters: dict = field(default_factory=dict)
    scoring: str = "ppr"
    slot_summary: str = ""                      # e.g. "2QB / 2RB / 2WR / 1TE / 1FLEX"
    league_name: str | None = None              # None when run from --teams/--format


def slot_summary(roster: dict) -> str:
    """``1QB / 2RB / 2WR / 1TE / 1FLEX`` from a roster dict, for report headers.

    Mirrors :meth:`eval.league.LeagueConfig.slot_summary` so a board run from the
    ``--teams``/``--format`` shorthand still states the shape it assumed. Fractional counts (the
    superflex 1.7 heuristic) are printed as-is rather than rounded into a lie.
    """
    parts = []
    for pos in (*MODELED_POS, "FLEX"):
        n = roster.get(pos, 0)
        if not n:
            continue
        parts.append(f"{int(n) if float(n).is_integer() else n}{pos}")
    return " / ".join(parts)


def pick_round(pick, teams: int):
    """The draft round a pick number falls in — keeper costs are argued about in rounds."""
    try:
        return max(1, math.ceil(float(pick) / teams))
    except (TypeError, ValueError):
        return None


def render_markdown(result: KeeperResult) -> str:
    """Render a keeper board to markdown (the same content the CSV carries, ranked)."""
    lines = [f"# Keeper Board — {result.season} · {result.label}", ""]
    slots = f" · {result.slot_summary}" if result.slot_summary else ""
    lines.append(f"_{result.teams}-team · **{result.scoring.replace('_', ' ')}**{slots}._")
    lines.append("")
    lines.append("_A keeper costs you the pick you drafted him at, so_ "
                 "**`surplus = pick paid − projected board slot`**_. Positive means the player is "
                 "worth more than the pick you'd spend; negative means you should re-draft him "
                 "later (or let him go). Model-only — ECR/ADP are benchmarks, never blended in._")
    lines.append("")

    ranked = result.ranked
    if ranked is None or len(ranked) == 0:
        lines.append("_None of the input players matched a projection._")
    else:
        keeps = ranked[ranked["keep"]]
        lines.append(f"## Verdict — keep {len(keeps)} of {len(ranked)}")
        lines.append("")
        lines.append("| | player | pos | proj PPG | board slot | pick (round) | surplus |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in ranked.iterrows():
            rnd = pick_round(r["pick"], result.teams)
            pick_str = f"{int(r['pick'])} (rd {rnd})" if rnd else str(r["pick"])
            mark = "**KEEP**" if r["keep"] else "pass"
            lines.append(
                f"| {mark} | {r['player_name']} | {r['pos_rank']} | {r['proj_ppg']:.2f} | "
                f"{int(r['proj_overall_rank'])} | {pick_str} | {r['surplus']:+.0f} |")
        lines.append("")

    if result.replacement:
        lines.append("## Replacement level")
        lines.append("")
        lines.append("_The first non-starter at each position once every league-wide slot "
                     "(dedicated + flex) is filled. It is what makes a QB and a WR comparable._")
        lines.append("")
        lines.append("| position | started league-wide | replacement PPG |")
        lines.append("|---|---|---|")
        for pos in MODELED_POS:
            if pos in result.replacement:
                lines.append(f"| {pos} | {result.starters.get(pos, 0)} | "
                             f"{result.replacement[pos]:.2f} |")
        lines.append("")

    unmatched = result.unmatched
    if unmatched is not None and len(unmatched):
        lines.append("## Unscored")
        lines.append("")
        lines.append("_No model projection — K/DST, or the name didn't match._")
        lines.append("")
        lines.append("| player | pick |")
        lines.append("|---|---|")
        for _, r in unmatched.iterrows():
            lines.append(f"| {r['player']} | {r['pick']} |")
        lines.append("")
    return "\n".join(lines) + "\n"
