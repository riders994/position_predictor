"""Phase 3 — build the **leak-safe N->N+1** modeling table for the next-season archetype predictor.

One row per (player, season ``N``) that has a season ``N+1`` in the data (returning players only, per
§1). Every feature is knowable **through season N**; the target is the player's archetype in ``N+1``.

Features (all as-of N):
- **Persistence signal** — the current soft membership ``p0..p11`` (+ ``top_prob``, ``entropy``). A model
  handed these can, at floor, reproduce the persistence baseline; lift over it must come from the rest.
- **Style** — the 19 z-scored play-style features (same inputs Phase 1 clustered on).
- **Trajectory** — one-year style **deltas** ``z_N - z_{N-1}`` (0 when N-1 is missing, flagged by
  ``has_prior``) and the current hard archetype's **tenure** (consecutive seasons in it). Drift is where
  the signal over persistence is expected to live (slasher -> spot-up, rim-runner -> stretch big).
- **Experience** — ``yoe`` = seasons since the player's first appearance in the 2013+ data (Model A's
  age proxy; leak-safe, censored for pre-2014 debuts — a true-age variant is fetched separately).

Targets: ``target_arch`` (hard label in N+1) and ``t0..t11`` (the soft membership vector in N+1, for
soft/log-loss scoring). The archetype label space is stable across the table because Phase-1 assignment
uses one fixed GMM over all seasons.
"""
from __future__ import annotations

import numpy as np

from ..utils.io import DATA_INTERIM, DATA_PROCESSED, read_parquet, write_parquet

IDENTITY = ["athlete_id", "season", "player_name"]
AGES_PATH = DATA_INTERIM / "nba_player_ages.parquet"        # optional true-age source (Model B)


def _prob_cols(membership):
    return sorted((c for c in membership.columns if c.startswith("p") and c[1:].isdigit()),
                  key=lambda c: int(c[1:]))


def build_predict_table(membership=None, features=None, *, zcols=None, write=True):
    """Join membership + style features into the leak-safe N->N+1 predictor table.

    Returns the table (one row per predictable player-season) with feature and target columns.
    """
    if membership is None:
        membership = read_parquet(DATA_PROCESSED / "nba_archetype_membership.parquet")
    if features is None:
        features = read_parquet(DATA_PROCESSED / "nba_archetype_features.parquet")
    pcols = _prob_cols(membership)
    if zcols is None:
        zcols = [c for c in features.columns if c.endswith("_z")]

    feat = features[["athlete_id", "season", *zcols]].copy()
    df = membership.merge(feat, on=["athlete_id", "season"], how="left").sort_values(
        ["athlete_id", "season"]).reset_index(drop=True)
    if AGES_PATH.exists():                                   # Model B: attach true age if fetched
        df = df.merge(read_parquet(AGES_PATH)[["athlete_id", "season", "age"]],
                      on=["athlete_id", "season"], how="left")

    g = df.groupby("athlete_id", sort=False)
    # trajectory: one-year style deltas (leak-safe: uses N and N-1 only)
    prev_season = g["season"].shift(1)
    has_prior = ((df["season"] - prev_season) == 1).fillna(False).to_numpy()
    for z in zcols:
        delta = (df[z] - g[z].shift(1)).to_numpy()
        df[f"d_{z}"] = np.where(has_prior, delta, 0.0)
    df["has_prior"] = has_prior.astype(int)
    df["yoe"] = (df["season"] - g["season"].transform("min")).astype(int)
    df["arch_tenure"] = _consecutive_arch_tenure(df)

    # target: archetype in N+1 (only when the very next season exists for this player)
    next_season = g["season"].shift(-1)
    df["target_arch"] = g["arch"].shift(-1)
    for j, c in enumerate(pcols):
        df[f"t{j}"] = g[c].shift(-1)
    keep = (next_season - df["season"] == 1) & df["target_arch"].notna()
    out = df[keep].copy()
    out["target_arch"] = out["target_arch"].astype(int)
    out.attrs["zcols"] = zcols
    out.attrs["pcols"] = pcols
    if write:
        write_parquet(out, DATA_PROCESSED / "phase3_predict_table.parquet")
    return out


def _consecutive_arch_tenure(df):
    """Number of consecutive prior seasons (incl. current) the player has held the current archetype.

    Uses only past/current rows -> leak-safe. Resets when the archetype changes or a season is skipped.
    """
    tenure = np.ones(len(df), dtype=int)
    aid = df["athlete_id"].to_numpy()
    season = df["season"].to_numpy()
    arch = df["arch"].to_numpy()
    for i in range(1, len(df)):
        if aid[i] == aid[i - 1] and season[i] - season[i - 1] == 1 and arch[i] == arch[i - 1]:
            tenure[i] = tenure[i - 1] + 1
    return tenure


def feature_columns(table, *, kind="yoe"):
    """Model feature columns. ``kind='yoe'`` = Model A (experience proxy); ``kind='age'`` swaps in a
    true ``age`` column (Model B) in place of ``yoe`` (falls back to yoe if ``age`` is absent)."""
    zcols = table.attrs.get("zcols") or [c for c in table.columns if c.endswith("_z")]
    pcols = table.attrs.get("pcols") or _prob_cols(table)
    base = [*pcols, "top_prob", "entropy", *zcols, *[f"d_{z}" for z in zcols],
            "has_prior", "arch_tenure"]
    if kind == "age" and "age" in table.columns:
        return [*base, "age"]
    return [*base, "yoe"]


def target_prob_columns(table):
    """The soft next-season membership target columns ``t0..t11``."""
    return sorted((c for c in table.columns if c.startswith("t") and c[1:].isdigit()),
                  key=lambda c: int(c[1:]))
