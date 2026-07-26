"""CFBD college data — extends the college layer past cfbfastR's 2021 ceiling.

cfbfastR-data stops publishing after 2021, which costs nothing for *fitting* (a QB whose last
college season is 2022+ enters the NFL in 2023+ and is right-censored for the label anyway) but
leaves the project unable to **score today's prospects** — the thing it exists to do. CFBD
(collegefootballdata.com) covers the current season and fills that gap.

Credentials
-----------
Read from ``CFBD_API_KEY`` or, failing that, ``~/.config/cfbd/api_key``. The key is never stored
in the repository.

What this is not
----------------
**CFBD's PPA is not cfbfastR's EPA.** They are different models fit on different data, and PPA is
per *play* while our cfbfastR column is per *dropback*. Concatenating them into one feature would
put a source discontinuity at the 2021/2022 boundary that a model would happily read as signal.
So this module keeps CFBD values in their own columns with an explicit ``source`` tag, and
:func:`compare_to_cfbfastr` measures the relationship on the 2013–2021 overlap rather than
assuming one. Whether the two can be spliced is an empirical question, answered in
``REPORT_qb_breakout_cfbd.md``.

Coverage (measured, not documented upstream):

===================  ==========
Endpoint             First year
===================  ==========
``/ppa/players/season``      2013
``/stats/player/season``     ~2008 (thin until 2013)
===================  ==========
"""

from __future__ import annotations

import os
from pathlib import Path

BASE = "https://api.collegefootballdata.com"

# PPA (CFBD's expected-points model) only has player-season aggregates from 2013.
FIRST_PPA_SEASON = 2013

KEY_FILE = Path.home() / ".config" / "cfbd" / "api_key"

# /stats/player/season returns one row per (player, statType); these are the ones we keep.
_PASSING_STATS = {"ATT": "attempts", "COMPLETIONS": "completions", "YDS": "pass_yds",
                  "TD": "pass_td", "INT": "interceptions"}
_RUSHING_STATS = {"CAR": "rush_att", "YDS": "rush_yds", "TD": "rush_td"}


def api_key() -> str:
    """Return the CFBD key from the environment or the user's config file."""
    key = os.environ.get("CFBD_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    raise SystemExit(
        "No CFBD API key. Set CFBD_API_KEY or write the key to ~/.config/cfbd/api_key "
        "(free key from https://collegefootballdata.com/key)."
    )


def _get(path: str, params: dict, session=None):
    import requests

    sess = session or requests.Session()
    resp = sess.get(f"{BASE}{path}", params=params,
                    headers={"Authorization": f"Bearer {api_key()}"}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_qb_season_stats(season: int, *, session=None):
    """Fetch one season of QB passing and rushing totals plus PPA, as one row per player.

    Rushing is filtered to players who also threw, so a running back's carries do not create a
    phantom QB row. Games played is **not** available from these endpoints and is deliberately
    absent rather than guessed.
    """
    import pandas as pd

    def season_stats(category, wanted):
        rows = _get("/stats/player/season", {"year": season, "category": category},
                    session=session)
        recs = {}
        for r in rows:
            col = wanted.get(r.get("statType"))
            if col is None:
                continue
            key = (r.get("playerId"), r.get("player"), r.get("team"))
            rec = recs.setdefault(key, {})
            try:
                rec[col] = float(r.get("stat"))
            except (TypeError, ValueError):
                continue
        frame = pd.DataFrame(
            [{"cfbd_player_id": k[0], "player": k[1], "team": k[2], **v} for k, v in recs.items()]
        )
        # The API emits one row per (player, statType) and simply omits stat types a player has
        # none of, so a column can be missing entirely for a whole season. Guarantee the schema
        # rather than letting a downstream derived column raise KeyError.
        keys = ["cfbd_player_id", "player", "team"]
        return frame.reindex(columns=keys + list(wanted.values())) if len(frame) else \
            pd.DataFrame(columns=keys + list(wanted.values()))

    passing = season_stats("passing", _PASSING_STATS)
    rushing = season_stats("rushing", _RUSHING_STATS)
    if passing.empty:
        return pd.DataFrame()

    out = passing.merge(rushing, on=["cfbd_player_id", "player", "team"], how="left")
    for col in ("rush_att", "rush_yds", "rush_td"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)

    ppa_rows = _get("/ppa/players/season", {"year": season, "position": "QB"}, session=session)
    ppa = pd.DataFrame([{
        "cfbd_player_id": r.get("id"),
        "cfbd_position": r.get("position"),
        "conference": r.get("conference"),
        "ppa_all": (r.get("averagePPA") or {}).get("all"),
        "ppa_pass": (r.get("averagePPA") or {}).get("pass"),
        "ppa_rush": (r.get("averagePPA") or {}).get("rush"),
        "ppa_third_down": (r.get("averagePPA") or {}).get("thirdDown"),
        "ppa_passing_downs": (r.get("averagePPA") or {}).get("passingDowns"),
        "total_ppa_all": (r.get("totalPPA") or {}).get("all"),
    } for r in ppa_rows])

    if not ppa.empty:
        out = out.merge(ppa, on="cfbd_player_id", how="left")
    out["season"] = season
    out["source"] = "cfbd"
    return out


def build_cfbd_qb_seasons(seasons, *, min_attempts: int = 50, progress=None):
    """Fetch several seasons and keep rows with enough passing volume to be a QB-season.

    ``min_attempts`` mirrors the cfbfastR layer's dropback floor closely enough for the two to be
    compared; it is attempts rather than dropbacks because CFBD does not expose sacks per player.
    """
    import pandas as pd
    import requests

    sess = requests.Session()
    frames = []
    for season in seasons:
        df = fetch_qb_season_stats(season, session=sess)
        if df.empty:
            if progress:
                progress(season, 0)
            continue
        df = df[df["attempts"].fillna(0) >= min_attempts]
        if progress:
            progress(season, len(df))
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.assign(
        completion_pct=out["completions"] / out["attempts"],
        yards_per_attempt=out["pass_yds"] / out["attempts"],
        td_rate=out["pass_td"] / out["attempts"],
        int_rate=out["interceptions"] / out["attempts"],
        rush_yds_per_att=(out["rush_yds"] / out["rush_att"]).where(out["rush_att"] > 0),
        # Same mobility axis as the cfbfastR layer. The denominator differs slightly (attempts,
        # not dropbacks, since CFBD gives no per-player sack count), so the two are close but not
        # identical — which is exactly what compare_to_cfbfastr quantifies.
        rush_share=out["rush_att"] / (out["attempts"] + out["rush_att"]),
    )


def compare_to_cfbfastr(cfbd_seasons, cfbfastr_seasons):
    """Measure agreement between the two sources on their overlapping seasons.

    Splicing two differently-derived feature series is only safe if they measure the same thing on
    the same scale. This joins on ``(season, player, team)`` and reports, per metric, the
    correlation and the mean difference — evidence for or against treating them as one column.

    Returns ``(joined, summary)``.
    """
    import numpy as np
    import pandas as pd

    from .college_link import normalize_school
    from .link import normalize_name

    a = cfbd_seasons.copy()
    b = (cfbfastr_seasons.to_pandas() if hasattr(cfbfastr_seasons, "to_pandas")
         else cfbfastr_seasons).copy()
    for frame in (a, b):
        frame["_k"] = frame["player"].map(normalize_name)
        frame["_t"] = frame["team"].map(normalize_school)

    joined = a.merge(b, on=["season", "_k", "_t"], suffixes=("_cfbd", "_cfbfastr"))

    pairs = [
        ("attempts", "attempts_cfbd", "attempts_cfbfastr"),
        ("pass_yds", "pass_yds_cfbd", "pass_yds_cfbfastr"),
        ("pass_td", "pass_td_cfbd", "pass_td_cfbfastr"),
        ("rush_att", "rush_att_cfbd", "rush_att_cfbfastr"),
        ("rush_share", "rush_share_cfbd", "rush_share_cfbfastr"),
        ("efficiency (ppa_pass vs epa/db)", "ppa_pass", "pass_epa_per_db"),
    ]
    rows = []
    for label, left, right in pairs:
        if left not in joined.columns or right not in joined.columns:
            continue
        sub = joined[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 3:
            continue
        rows.append({
            "metric": label,
            "n": len(sub),
            "corr": round(float(np.corrcoef(sub[left], sub[right])[0, 1]), 3),
            "mean_cfbd": round(float(sub[left].mean()), 3),
            "mean_cfbfastr": round(float(sub[right].mean()), 3),
            "mean_diff": round(float((sub[left] - sub[right]).mean()), 3),
        })
    return joined, pd.DataFrame(rows)


def fit_calibration(joined, *, source_col="ppa_pass", target_col="pass_epa_per_db"):
    """Fit CFBD PPA onto the cfbfastR EPA scale, and report how well it actually transfers.

    A model trained on cfbfastR efficiency cannot be handed a raw PPA value — the scales differ by
    more than the target's own mean. A linear map fixes the scale, but only usefully if the
    relationship is tight, so this returns the fit **together with the error it carries**:

    ``r2``            variance of the target explained by the mapped source.
    ``resid_sd``      standard deviation of the calibration error, in target units.
    ``target_sd``     spread of the target across players.
    ``noise_ratio``   ``resid_sd / target_sd`` — the fraction of between-player spread the mapping
                      *fails* to reproduce. Near 0 is a safe splice; near 1 means the calibrated
                      column is mostly noise and should not be used as if it were measured.
    """
    import numpy as np
    import pandas as pd

    sub = joined[[source_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    x, y = sub[source_col].to_numpy(), sub[target_col].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    resid = y - pred
    target_sd = float(y.std(ddof=1))
    resid_sd = float(resid.std(ddof=1))
    ss_res, ss_tot = float((resid ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(sub)),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r2": round(1 - ss_res / ss_tot, 3),
        "resid_sd": round(resid_sd, 4),
        "target_sd": round(target_sd, 4),
        "noise_ratio": round(resid_sd / target_sd, 3),
    }


def apply_calibration(cfbd_seasons, calibration, *, source_col="ppa_pass",
                      out_col="pass_epa_per_db_est"):
    """Map CFBD PPA onto the cfbfastR EPA scale using :func:`fit_calibration` output.

    The result is named ``*_est`` and carries ``efficiency_is_estimated=1`` so a calibrated value
    can never be mistaken downstream for a measured one.
    """
    out = cfbd_seasons.copy()
    out[out_col] = calibration["slope"] * out[source_col] + calibration["intercept"]
    out["efficiency_is_estimated"] = 1
    return out
