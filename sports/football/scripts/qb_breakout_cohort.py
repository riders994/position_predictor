"""Stage 1 of the late-breakout QB project: build the cohort and describe it.

Reads the nflverse caches the sibling ``position_predictor`` pipeline already maintains
(``weekly``, ``draft_picks``, ``rosters``), derives per-season QB ranks and career breakout
labels, and writes both the machine-readable cohort and a human-readable report.

Usage
-----
    uv run python sports/football/scripts/qb_breakout_cohort.py
    uv run python sports/football/scripts/qb_breakout_cohort.py --late-threshold 3

Outputs
-------
``data/processed/qb_breakout_seasons.parquet``  per QB-season, with rank and breakout flags
``data/processed/qb_breakout_careers.parquet``  one row per QB career, all three label variants
``reports/REPORT_qb_breakout_cohort.md``        the descriptive write-up
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, REPORTS_DIR, ensure_dir,
)
from qb_breakout.eda import (  # noqa: E402
    cohort_crosstab, draft_situation, label_comparison, late_roster, pending_watchlist,
    timing_distribution, trough_profile,
)
from qb_breakout.labels import (  # noqa: E402
    BREAKOUT_RANK, LATE_YEAR_THRESHOLD, MIN_GAMES, build_qb_careers, build_qb_seasons,
    rank_qb_seasons,
)

# The label the project models. Chosen empirically over "first top-20" and "first top-12":
# only the sustained bar puts Mayfield/Tannehill/Geno in the late cell while keeping
# Brady/Allen/Burrow/Purdy on time. See docs/QB_BREAKOUT_PLAN.md §3.2.
TARGET_LABEL = "late_sustained"
TARGET_RELOC = "sustained_relocated"


def _read(name):
    import polars as pl

    path = DATA_RAW / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun the sibling pipeline's fetch stage first:\n"
            f"  make -C sports/football fetch"
        )
    return pl.read_parquet(path)


def _md_table(df, *, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = [
        "| " + " | ".join("" if v != v else str(v) for v in row) + " |"
        for row in d.itertuples(index=False, name=None)
    ]
    return "\n".join([head, sep, *rows])


def build(late_threshold: int = LATE_YEAR_THRESHOLD, min_games: int = MIN_GAMES):
    """Build ranked QB seasons and the career cohort from the cached nflverse data."""
    seasons = build_qb_seasons(_read("weekly"))
    ranked = rank_qb_seasons(seasons, min_games=min_games)
    careers = build_qb_careers(
        ranked, _read("draft_picks"), _read("rosters"), late_threshold=late_threshold
    )
    return ranked, careers


def write_report(ranked, careers, path: Path, *, late_threshold: int) -> Path:
    n_late = int((careers[TARGET_LABEL] == 1).sum())
    n_defined = int(careers[TARGET_LABEL].notna().sum())
    seasons_span = f"{int(ranked['season'].min())}–{int(ranked['season'].max())}"

    md = f"""# Late-breakout QBs — cohort and descriptive analysis

Which quarterbacks became fantasy-relevant *after* the league had moved on, and what they had in
common before they got there. This report covers the **NFL-side ground truth only**; the
pre-NFL (high-school and college) evidence that the models are restricted to is built in later
stages. Design rationale: [`docs/QB_BREAKOUT_PLAN.md`](../docs/QB_BREAKOUT_PLAN.md).

- **Seasons covered:** {seasons_span} (nflverse weekly stats)
- **Cohort:** {len(careers)} QBs entering the NFL in 1999 or later
- **Breakout tier:** top-{BREAKOUT_RANK} PPR points-per-game among QBs clearing {MIN_GAMES}
  games — superflex-relevant, since QB15 is where draft-day value lives
- **Late:** the breakout arrived in NFL year {late_threshold} or later
- **Late breakouts found:** **{n_late}** of {n_defined} QBs who ever broke out

---

## 1. Defining "breakout" is the whole problem

Three candidate definitions were built and scored against QBs whose careers are not in dispute.
They disagree sharply:

{_md_table(label_comparison(careers))}

`late` (first top-{BREAKOUT_RANK} season) is too loose. The bar is rank 20 of roughly 32 QBs who
play, so one ordinary season clears it — **Baker Mayfield's 2018 rookie year ranks exactly 20th**,
and Ryan Tannehill's 2014 ranks 10th. Both therefore read as *on-time* breakouts, which is the
opposite of what their careers show.

`late_qb1` (first top-12 season) over-corrects: it labels **Tom Brady** a late breakout because
his first top-12 fantasy season came in year 6, even though he was a quality starter from year 2.

`{TARGET_LABEL}` requires the tier to **hold** — top-{BREAKOUT_RANK} in at least 2 of the 3
seasons starting with the breakout. This is the definition the project models. It is the only one
of the three that puts Mayfield (year 6), Tannehill (year 8) and Geno Smith (year 10) in the late
cell while leaving Brady, Josh Allen, Joe Burrow and Brock Purdy on time.

## 2. Developed late, or needed a new building?

{_md_table(cohort_crosstab(careers, late_col=TARGET_LABEL, reloc_col=TARGET_RELOC), index=True)}

## 3. Which draft situations produce late breakouts

Censored careers — QBs who entered too recently to have had a year-{late_threshold} season yet —
are excluded so recent draft classes do not inflate the never-broke-out rate.

{_md_table(draft_situation(careers, late_col=TARGET_LABEL), index=True)}

Read `pct_late_given_breakout` rather than `pct_ever_broke_out`: the first says a bucket produces
good quarterbacks, the second says it produces them *slowly*.

## 4. How deep was the trough?

Median profile over each QB's first three NFL seasons — the rookie-contract window the league
judges them on. Seasons too short to be ranked are counted as rank 40, worse than the worst real
rank, so "did not play enough to rank" registers as the bad outcome it is.

{_md_table(trough_profile(careers, ranked, late_col=TARGET_LABEL), index=True)}

**This is the central result, and it justifies the whole project.** Through three NFL seasons,
future late breakouts look far more like QBs who never broke out than like QBs who broke out on
time: median early PPG of 5.97 against 4.18 for the busts and 15.68 for the on-time group, and a
median early rank pinned at the unranked floor for both the late and the never groups. They do get
more early playing time than the busts (median 14.5 games vs 7), so there is *some* separation —
but on production, early NFL evidence barely distinguishes a future late breakout from a bust.

If early NFL performance cannot separate them, then a model that waits for NFL evidence is waiting
for information that does not arrive. That is the case for going back to what was knowable before
the player was drafted at all — which is what the remaining stages build.

## 5. When the breakout arrives

{_md_table(timing_distribution(careers, late_col=TARGET_LABEL))}

## 6. The late-breakout roster

The cohort every later stage is built to predict from pre-NFL evidence alone.

{_md_table(late_roster(careers, late_col=TARGET_LABEL))}

## 7. Still open

QBs whose window has not closed — either too recent to have reached year {late_threshold}, or
holding one breakout season that has not yet had time to prove it sustains. **Sam Darnold is the
live case**: his 2024 with Minnesota ranked 9th, but 2025 came in at 25th, so the sustained label
is still pending rather than earned. These rows are excluded from every base rate above, and they
are precisely who a working model would be scoring today.

{_md_table(pending_watchlist(careers, late_col=TARGET_LABEL).head(20))}

---

## Caveats

- **Small N.** {n_late} late breakouts across {seasons_span} is the honest ceiling on this
  question. Any model here is evidence-weighting, not prediction at scale, and the validation
  design has to reflect that.
- **Right-censoring is real.** {int(careers['censored'].sum())} QBs are too recent to label. They
  are excluded from base rates rather than counted as failures.
- **2020 is kept.** The sibling project drops the COVID season from supervised use, but a
  breakout is a career milestone — dropping 2020 would erase the first good season of anyone who
  broke out that year.
- **Fantasy PPG is the tier, not football quality.** Rushing production inflates QB fantasy
  scoring relative to passing quality, so a mobile QB clears the tier on less passing skill.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--late-threshold", type=int, default=LATE_YEAR_THRESHOLD,
                    help="NFL year at/after which a breakout counts as late")
    ap.add_argument("--min-games", type=int, default=MIN_GAMES,
                    help="games-played bar for a season to be rankable")
    args = ap.parse_args(argv)

    ranked, careers = build(args.late_threshold, args.min_games)

    ensure_dir(DATA_PROCESSED)
    ranked.to_parquet(DATA_PROCESSED / "qb_breakout_seasons.parquet", index=False)
    careers.to_parquet(DATA_PROCESSED / "qb_breakout_careers.parquet", index=False)

    report = write_report(
        ranked, careers, REPORTS_DIR / "REPORT_qb_breakout_cohort.md",
        late_threshold=args.late_threshold,
    )
    n_late = int((careers[TARGET_LABEL] == 1).sum())
    print(f"cohort: {len(careers)} QBs | late breakouts ({TARGET_LABEL}): {n_late}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
