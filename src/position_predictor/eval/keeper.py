"""Keeper valuation — turn per-position projections into a cross-position keeper board.

A keeper costs you the **pick you drafted the player at**. So a keep is good when the player's
projected draft value beats that pick. To compare across positions and against a pick number we
need one currency: **value over replacement (VORP/VBD)**. We compute a replacement level per
position from league settings (teams × started slots, flex filled by the best remaining RB/WR),
subtract it from each projection, and rank everyone by VORP → a projected draft board. Then

    surplus = (pick you'd pay) − (player's projected overall board slot)

ranks the keepers: a player you keep at pick 100 who projects as a top-30 board slot is a +70
steal. This is model-only — ECR stays a benchmark, never blended in.

Scope: RB/WR/QB (the modeled positions). TE/K/DST and unmatched names are reported as *unscored*.
"""
from __future__ import annotations

import re

# Per-team started slots. QB depth is set by league format; flex is filled from RB/WR (we don't
# model TE, so TE neither contributes flex nor gets a replacement level here — documented).
QB_SLOTS_PER_TEAM = {"1qb": 1.0, "sf": 1.7, "2qb": 2.0}
DEFAULT_ROSTER = {"RB": 2, "WR": 2, "FLEX": 1}   # excludes QB (by format) and TE (unscored)
FLEX_POS = ("RB", "WR")
MODELED_POS = ("QB", "RB", "WR")


def replacement_levels(proj, *, teams, fmt, roster=None):
    """Projected-PPG replacement level per position for the given league.

    Replacement = the projected points of the **first non-starter** at each position once every
    started slot (dedicated + flex) is filled league-wide. Returns ``(replacement, starters)``.
    """
    roster = roster or DEFAULT_ROSTER
    fmt = fmt.lower()
    if fmt not in QB_SLOTS_PER_TEAM:
        raise ValueError(f"format must be one of {sorted(QB_SLOTS_PER_TEAM)}; got {fmt!r}")
    pools = {p: sorted(proj.loc[proj["position"] == p, "proj_ppg"], reverse=True)
             for p in MODELED_POS}
    starters = {"QB": round(teams * QB_SLOTS_PER_TEAM[fmt]),
                "RB": teams * roster["RB"], "WR": teams * roster["WR"]}
    # Flex (RB/WR-eligible here): hand each slot to whichever position's next-best player is higher.
    idx = {p: starters[p] for p in FLEX_POS}
    for _ in range(teams * roster.get("FLEX", 0)):
        cand = {p: pools[p][idx[p]] for p in FLEX_POS if idx[p] < len(pools[p])}
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


def build_board(proj, *, teams, fmt, roster=None):
    """Add ``vorp`` + ``proj_overall_rank`` (1 = best keeper-board value). Returns
    ``(board, replacement, starters)`` with the board sorted by VORP."""
    replacement, starters = replacement_levels(proj, teams=teams, fmt=fmt, roster=roster)
    board = proj.copy()
    board["vorp"] = (board["proj_ppg"]
                     - board["position"].map(replacement).fillna(0.0)).round(2)
    board = board.sort_values("vorp", ascending=False).reset_index(drop=True)
    board["proj_overall_rank"] = range(1, len(board) + 1)
    return board, replacement, starters


_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def _norm(name) -> str:
    """Normalise a player name for matching: lowercase, drop punctuation & generational suffixes."""
    s = str(name).lower().replace(".", " ").replace("'", "").replace("-", " ")
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
                              "reason": "no projection (TE/K/DST or name not matched)"})
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
