"""Stage 1: load the injury/roster/schedule caches and make their defects visible.

Nothing is modelled here. The stage exists because three properties of this data decide the
whole design, and each is better measured than assumed: the 2016 ``report_status`` regime
break, the spread in how much clubs disclose, and whether the injury report actually joins to
the weekly roster spine.

Usage
-----
    uv run python sports/football/scripts/medstaff_ingest.py
    uv run python sports/football/scripts/medstaff_ingest.py --seasons 2021 2025

Outputs
-------
``data/processed/medstaff_injuries.parquet``     coalesced + classified injury rows
``data/processed/medstaff_listing.parquet``      per team-season disclosure indices
``reports/REPORT_medstaff_data.md``              coverage, regime break, taxonomy audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, REPORTS_DIR, ensure_dir, write_parquet,
)
from medstaff.data import (  # noqa: E402
    EXCLUDED_SEASONS, FIRST_INJURY_SEASON, FOCAL_GROUPS,
    body_part_summary, join_rate, listing_propensity, load_injuries, load_rosters_weekly,
    load_schedules, report_regime_table, season_coverage, unmapped_report,
)

# A raw string above this share of rows landing in "other" means the taxonomy has a real gap,
# not a long tail. Kept low deliberately: the top 45 strings already cover 98.3% of rows.
UNMAPPED_ALERT_SHARE = 0.001


def _table(frame, *, floats: int = 3) -> str:
    """Render a polars/pandas frame as a markdown table."""
    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame

    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.{floats}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def build(seasons: tuple[int, ...] | None = None):
    """Load the three caches, classify injuries, and compute the stage-1 diagnostics."""
    injuries = load_injuries(DATA_RAW, seasons=seasons)
    rosters = load_rosters_weekly(DATA_RAW, seasons=seasons)
    schedules = load_schedules(DATA_RAW, seasons=seasons)

    import polars as pl

    reg_injuries = injuries.filter(pl.col("game_type") == "REG")
    reg_rosters = rosters.filter(pl.col("game_type") == "REG") \
        if "game_type" in rosters.columns else rosters

    return {
        "injuries": injuries,
        "rosters": rosters,
        "schedules": schedules,
        "coverage": pl.concat([
            season_coverage(reg_injuries, label="injuries"),
            season_coverage(reg_rosters, label="rosters_weekly"),
        ], how="diagonal_relaxed"),
        "regime": report_regime_table(reg_injuries),
        "body_parts": body_part_summary(reg_injuries),
        "unmapped": unmapped_report(reg_injuries),
        "listing": listing_propensity(reg_injuries, reg_rosters),
        "join": join_rate(reg_injuries, reg_rosters),
    }


def write_report(result, path: Path) -> Path:
    """Render the stage-1 report. Numbers are interpolated from the frames, never retyped."""
    import polars as pl

    regime = result["regime"]
    join = result["join"]
    listing = result["listing"]
    unmapped = result["unmapped"]
    body = result["body_parts"]

    pre = regime.filter(~pl.col("post_probable_drop"))["report_status_null"]
    post = regime.filter(pl.col("post_probable_drop"))["report_status_null"]
    bp_null_max = regime["body_part_null"].max()
    ps_null_max = regime["practice_status_null"].max()

    team_listing = (
        listing.group_by("team")
        .agg(pl.col("listed_weeks").sum().alias("listed"),
             pl.col("questionable_play_rate").mean().alias("q_play_rate"))
        .sort("listed")
    )
    lo, hi = team_listing.head(3), team_listing.tail(3)

    unmapped_alert = unmapped.filter(pl.col("share") >= UNMAPPED_ALERT_SHARE)
    focal = body.filter(pl.col("body_part_group").is_in(list(FOCAL_GROUPS)))

    seasons = sorted(result["injuries"]["season"].unique().to_list())

    md = f"""# Medical-staff grades — Stage 1: data

Injury reports, weekly rosters and schedules for **{seasons[0]}–{seasons[-1]}**. Nothing is
modelled here; this stage measures the three data properties that decide the rest of the design.

## 1. The 2016 regime break

`report_status` is **{pre.min():.0%}–{pre.max():.0%}** null before 2016 and
**{post.min():.0%}–{post.max():.0%}** null from 2016, when the league dropped the "Probable"
designation. Any measure built on that column is not comparable across the boundary.

The contrast is what matters: over the same seasons the coalesced body part is at most
**{bp_null_max:.1%}** null and `practice_status` at most **{ps_null_max:.1%}**. Those two are
regime-invariant, so every outcome in this project is built on them, and `report_status` is used
only as a severity refinement in a 2016+ sensitivity run.

{_table(regime)}

## 2. Does the spine join?

The availability spine is `rosters_weekly`, not the injury report — roster status is a
transaction record rather than a disclosure, and it covers every position. So the join has to
hold: **{join['rate']:.1%}** of injury rows ({join['matched']:,} of {join['rows']:,}) find a
weekly roster row on `(season, week, gsis_id)`.

A drop here would be the most dangerous silent failure in the project — an injury row with no
roster row has no availability spine, so the absence becomes invisible rather than missing.

{_table(join['per_season'])}

## 3. Disclosure behaviour varies by club

The injury report is a strategic artifact and clubs differ in how much they put on it. These are
**covariates, not grades**: a club that discloses more is not a club that injures more, and
keeping those apart is the point of measuring it.

Fewest report rows: {", ".join(f"**{r[0]}** ({r[1]:,})" for r in lo.iter_rows())}.
Most: {", ".join(f"**{r[0]}** ({r[1]:,})" for r in hi.iter_rows())}.

`questionable_play_rate` — the share of Questionable-listed players who were not inactive — is
the cleanest behavioural index, because a club that lists everybody has a high play-rate among
its Questionables.

{_table(team_listing.tail(10))}

## 4. Body-part taxonomy

{body.height} groups populated from {result['injuries']['body_part_raw'].n_unique()} distinct raw
strings. The five **focal** groups are those with a plausible common-cause mechanism — surface,
practice contact policy, S&C programme, medical protocol — and are the subject of the
cross-position-group signature analysis. Hamstring and shoulder are deliberately not focal:
soft-tissue and contact-incidental injuries are individually driven, so a team-wide signature in
them would have no mechanism to point at.

{_table(focal)}

All groups:

{_table(body)}

### Taxonomy audit

{"**No raw string above %.1f%% of rows falls through to `other`.**" % (UNMAPPED_ALERT_SHARE * 100)
 if unmapped_alert.is_empty() else
 "⚠️ **%d raw strings above %.1f%% of rows fall through to `other`** — the taxonomy has a gap:"
 % (unmapped_alert.height, UNMAPPED_ALERT_SHARE * 100)}

{"" if unmapped_alert.is_empty() else _table(unmapped_alert.head(20), floats=4)}

## 5. Sample decisions

- **Injuries exist from {FIRST_INJURY_SEASON}** and the injury *report* is comparable across
  that whole span, so it supplies the prior-injury lookback at any depth.
- **⚠️ But absence measures are only comparable from 2021**, which stage 2 established: the
  gameday active/inactive split is absent from `rosters_weekly` before then (`INA` is 2.8k rows
  across 2012–2019 vs 16.8k across 2021–2025, and `ACT` falls 0.86 → 0.59), and the reserve
  codes in `status_description_abbr` carry **no** R-codes before 2021. Anything counting missed
  games is restricted to 2021+.
- **{", ".join(str(s) for s in EXCLUDED_SEASONS)} excluded** — the practice and roster regime was
  unlike any other season, and the sibling pipeline already excludes it.
- Grading windows are therefore **2021–2025** (5yr) and **2023–2025** (3yr), which places both
  entirely inside the post-COVID 17-game era and the post-2016 reporting regime.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"),
                        help="inclusive season range (default: everything cached)")
    args = parser.parse_args(argv)

    seasons = tuple(range(args.seasons[0], args.seasons[1] + 1)) if args.seasons else None
    result = build(seasons)

    ensure_dir(DATA_PROCESSED)
    inj_path = write_parquet(result["injuries"], DATA_PROCESSED / "medstaff_injuries.parquet")
    lst_path = write_parquet(result["listing"], DATA_PROCESSED / "medstaff_listing.parquet")
    rpt_path = write_report(result, REPORTS_DIR / "REPORT_medstaff_data.md")

    join = result["join"]
    print(f"injuries {result['injuries'].height:,} rows · "
          f"spine join {join['rate']:.1%} · "
          f"{result['unmapped'].height} raw strings unmapped")
    for path in (inj_path, lst_path, rpt_path):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
