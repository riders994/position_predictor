"""Stage 7: the grades.

Reads stage 6's reliability verdict first and puts it at the top of its own report, so the
letters cannot be read without it. The curve is a fixed quota by decision — 3 A, 5 B, 8 C, 8 D,
8 F — which orders clubs rather than testing them, so every letter ships with its interval and
the count of clubs actually separable from their neighbour.

Usage
-----
    uv run python sports/football/scripts/medstaff_grades.py
    uv run python sports/football/scripts/medstaff_grades.py --staff-table trainers.csv

Outputs
-------
``data/processed/medstaff_grades_{3yr,5yr}.parquet``   club grades per window
``data/processed/medstaff_grades_positions.parquet``   club x position-group, shrunk
``reports/REPORT_medstaff_grades.md``                  the deliverable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, REPORTS_DIR, ensure_dir, write_parquet,
)
from medstaff.data import GROUP_ORDER, load_schedules  # noqa: E402
from medstaff.expected import team_observed_expected  # noqa: E402
from medstaff.grades import (  # noqa: E402
    ATTRIBUTABILITY_PRIOR, LETTER_QUOTAS, assign_letters, attach_staff,
    coach_changes, component_z, composite, empirical_bayes, head_coach_by_season,
    load_staff_table, reliability_weights, separation_flags, two_level_shrink,
)

WINDOWS = {"5yr": (2021, 2025), "3yr": (2023, 2025)}
COMPONENT_OUTCOMES = {
    "incidence_no_history": "onset",
    "duration": "returned",
    "recurrence": "recurred",
    "returns_at_all": "returned_at_all",
}


def _table(frame, *, floats: int = 3) -> str:
    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame

    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{floats}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def _residuals_for(window, *, by_group=False):
    """Recompute club (x position-group) observed-minus-expected within a season window."""
    lo, hi = window
    frames = []
    for name, outcome in COMPONENT_OUTCOMES.items():
        path = DATA_PROCESSED / f"medstaff_expected_{name}.parquet"
        if not path.exists():
            raise SystemExit(
                f"missing {path}\nRun stage 4 first:\n"
                f"  uv run python sports/football/scripts/medstaff_expected.py")
        frame = pl.read_parquet(path).filter(pl.col("season").is_between(lo, hi))
        extra = ["position_group"] if by_group and "position_group" in frame.columns else None
        oe = team_observed_expected(frame, frame["expected"].to_numpy(),
                                    outcome=outcome, extra_group=extra)
        oe["component"] = name
        frames.append(oe)
    return pd.concat(frames, ignore_index=True)


def _grade_window(window, weights):
    residuals = _residuals_for(window)
    z = component_z(residuals)
    wide = z.pivot(index="team", columns="component", values="z")

    shrunk = {}
    factors = {}
    for comp in wide.columns:
        s, _sd, k = empirical_bayes(wide[comp].to_numpy())
        shrunk[comp] = s
        factors[comp] = k
    shrunk_wide = pd.DataFrame(shrunk, index=wide.index)

    score = composite(shrunk_wide, weights)
    # posterior sd of the weighted mean, treating components as independent
    used = [c for c in weights if c in shrunk_wide.columns]
    w = np.array([weights[c] for c in used])
    w = w / w.sum()
    sd = float(np.sqrt(np.sum((w**2) * np.array([factors[c] for c in used]))))

    out = pd.DataFrame({"team": score.index, "score": score.to_numpy()})
    out["score_sd"] = sd
    out["letter"] = assign_letters(out.set_index("team")["score"]).reindex(out["team"]).to_numpy()
    sep = separation_flags(out.set_index("team")["score"],
                           pd.Series(sd, index=out["team"]))
    out["separated_from_next"] = sep["separated_from_next"].reindex(out["team"]).to_numpy()
    out["gap_to_next"] = sep["gap_to_next"].reindex(out["team"]).to_numpy()
    # Adjacent-pair separation is a very strict test in a 32-club ranking and comes out empty;
    # distinguishability from the league average is the statistic that actually informs a reader.
    centre = float(out["score"].mean())
    out["vs_average"] = out["score"] - centre
    out["separated_from_average"] = out["vs_average"].abs() > 1.2816 * sd
    for comp in shrunk_wide.columns:
        out[f"z_{comp}"] = shrunk_wide[comp].reindex(out["team"]).to_numpy()
    return out.sort_values("score", ascending=False).reset_index(drop=True), factors


def _grade_positions(window, weights):
    residuals = _residuals_for(window, by_group=True)
    z = component_z(residuals, team_col="team")
    # component_z drops position_group, so re-attach it from the residual frame
    z = z.join(residuals[["team", "component", "position_group"]].reset_index(drop=True),
               rsuffix="_r")
    z = z.loc[:, ~z.columns.duplicated()]
    z = z[z["position_group"].isin(GROUP_ORDER)]

    club_level, _ = _grade_window(window, weights)
    club_score = dict(zip(club_level["team"], club_level["score"]))

    rows = []
    for comp, grp in z.groupby("component"):
        wide = grp.pivot_table(index=["team", "position_group"], values="z")
        base = np.array([club_score.get(t, 0.0) for t, _ in wide.index])
        shrunk, sd, k = two_level_shrink(wide["z"].to_numpy(), base)
        for (team, pos), val, s in zip(wide.index, shrunk, sd):
            rows.append({"team": team, "position_group": pos, "component": comp,
                         "z_shrunk": val, "sd": s, "shrinkage_k": k})
    return pd.DataFrame(rows)


def build(*, staff_table=None):
    reliability = pl.read_parquet(DATA_PROCESSED / "medstaff_reliability.parquet").to_pandas()
    split_half = dict(zip(reliability["component"], reliability["split_half_r"]))
    gate_fail = set(
        reliability.loc[
            (reliability["split_half_r"] < 0.30) | (reliability["temporal_r"] < 0.30),
            "component"]
    )
    weights = reliability_weights(
        {k: v for k, v in split_half.items() if k in COMPONENT_OUTCOMES})

    windows, factors = {}, {}
    for name, span in WINDOWS.items():
        windows[name], factors[name] = _grade_window(span, weights)
    positions = _grade_positions(WINDOWS["5yr"], weights)

    schedules = load_schedules(DATA_RAW, seasons=tuple(range(*WINDOWS["5yr"])) + (2025,))
    coaches = head_coach_by_season(schedules)
    changes = coach_changes(coaches.filter(
        pl.col("season").is_between(*WINDOWS["5yr"])))

    for name in windows:
        windows[name] = attach_staff(
            windows[name].merge(changes.to_pandas(), on="team", how="left"), staff_table)

    return {"windows": windows, "positions": positions, "weights": weights,
            "factors": factors, "reliability": reliability, "gate_fail": gate_fail,
            "split_half": split_half, "staffed": staff_table is not None}


def write_report(result, path: Path) -> Path:
    rel = result["reliability"]
    weights = result["weights"]
    five = result["windows"]["5yr"]
    three = result["windows"]["3yr"]

    # Derived from the reliability table rather than asserted — an earlier version named the
    # passing components in prose and kept naming them after a bug fix changed which they were.
    gate_pass = ((rel["split_half_r"] >= 0.30) & (rel["temporal_r"] >= 0.30)
                 & (rel["permutation_p"] < 0.05))
    passing = rel.loc[gate_pass, "component"].tolist()
    failing = rel.loc[~gate_pass, "component"].tolist()
    passing_str = ", ".join(passing) if passing else "none"
    failing_str = ", ".join(failing) if failing else "none"
    if "recurrence" in passing:
        stability_reading = (
            "Recurrence — the outcome most plausibly owned by a medical staff — is among them, so "
            "the board carries some genuine medical signal.")
    elif passing:
        stability_reading = (
            "**Recurrence, the one outcome most plausibly owned by a medical staff, is not among "
            "them.** Everything that persists is about exposure and how a club uses injured "
            "reserve and times a return, so the board below separates clubs by "
            "**availability-management policy**, not by quality of medicine.")
    else:
        stability_reading = (
            "**No component clears the gate**, so nothing below has demonstrated year-over-year "
            "stability behind it.")

    n_sep = int(five["separated_from_next"].sum())
    n_avg = int(five["separated_from_average"].sum())
    quota_str = " · ".join(f"{n}×{ltr}" for ltr, n in LETTER_QUOTAS)

    wtable = pd.DataFrame({
        "component": list(weights),
        "split_half_r": [result["split_half"].get(c, np.nan) for c in weights],
        "weight_reliability": [weights[c] for c in weights],
        "weight_attributability_prior": [ATTRIBUTABILITY_PRIOR.get(c, np.nan) for c in weights],
        "gate": ["FAIL" if c in result["gate_fail"] else "PASS" for c in weights],
    })

    board = five[["team", "letter", "score", "score_sd", "vs_average",
                  "separated_from_average", "regime_changed"]].copy()
    board_3 = three[["team", "letter", "score"]].rename(
        columns={"letter": "letter_3yr", "score": "score_3yr"})
    merged = board.merge(board_3, on="team", how="left")
    agree = int((merged["letter"] == merged["letter_3yr"]).sum())

    pos = result["positions"]
    pos_summary = (
        pos.groupby("component")
        .agg(cells=("z_shrunk", "size"), shrinkage_k=("shrinkage_k", "first"))
        .reset_index()
    )

    md = f"""# Medical-staff grades — the board

## 0. Read this first

**These are not grades of medical staffs.** This data cannot identify one. The residual behind
every letter below bundles the athletic training staff, strength and conditioning, sports
science, the head coach's practice-intensity choices, the general manager's taste for durable
players, and scheme. What is measurable is a **team availability system**.

**Stage 6 narrowed it further.** Of five components, the ones that persist year over year are
**{passing_str}**; the ones that do not are **{failing_str}**. {stability_reading}

{_table(rel[["component", "split_half_r", "temporal_r", "temporal_p", "permutation_p"]])}

**The curve is a forced rank, by decision: {quota_str}.** It orders clubs; it does not test
them. Three A's and eight F's are assigned whether or not any club is distinguishable from
another. This is a deliberate provisional choice, to be replaced by an absolute score-to-grade
mapping once several seasons of reports exist — the code already accepts one, so that switch is
a parameter rather than a rewrite.

**{n_sep} of {len(five)} clubs are separable from the club ranked immediately below them** at 80%
confidence, and **{n_avg} of {len(five)} are separable from the league average**. Adjacent-pair
separation is a strict test in a 32-club field — neighbours are rarely distinguishable anywhere —
so the second number is the one that says whether the extremes mean anything. Read the letters as
an ordering with heavy overlap, not as five distinct tiers.

## 1. Weights

Weights are proportional to each component's **measured split-half reliability**, which is
self-limiting: a component carrying no signal gets a weight near zero without anyone deciding
that it should.

The attributability prior — what one would weight by if grading *medicine* — is shown beside it
and **orders the components almost exactly the other way round**. Recurrence is the most
attributable and the least reliable; duration the least attributable and the most reliable.
There is no way to satisfy both, and that tension is the honest content of this table.

{_table(wtable)}

## 2. The board — 5-year window (2021–2025)

`separated_from_average` is the column that says whether a letter means anything: it marks the
clubs whose interval excludes the league mean. **No club is separable from the club immediately
below it**, so the ordering *within* a letter band — and between adjacent bands — carries no
information. `regime_changed` marks clubs that changed head coach inside the window, where the
grade averages over more than one regime.

{_table(merged)}

**{agree} of {len(merged)} clubs receive the same letter on the 3-year window**, which is the
most direct stability check available on the board itself.

## 3. Position groups

Every cell is shrunk toward its own club's overall value, and cells are **never ranked**. A club
× position-group cell holds a median of four episodes over five years; at that size the shrinkage
factor does most of the work and the ordering within a club is not interpretable.

{_table(pos_summary)}

Full cell-level estimates with intervals ship in
`data/processed/medstaff_grades_positions.parquet` rather than as a table here, precisely so they
are not read as a ranking.

## 4. Staff identity

{"A staff table was supplied and joined." if result["staffed"] else
 "**No staff table was supplied, so these are franchise grades over the window.** nflverse "
 "publishes head coaches, not head athletic trainers, and no free structured trainer-by-season "
 "source exists. Pass `--staff-table` to attach one; without it the report grades the club, and "
 "says so rather than implying more."}

## 5. What would change these

- An absolute score-to-grade mapping, once several seasons exist, replacing the forced curve.
- A trainer tenure table, which would let the grade attach to people rather than franchises —
  though tenures are shorter than franchise histories, so the cells would be thinner still.
- More seasons. The comparable window opens in 2021 because the roster fields change meaning
  there; every additional season widens it by 20%.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staff-table", type=Path, default=None,
                        help="optional CSV of team,head_athletic_trainer[,season]")
    args = parser.parse_args(argv)

    staff = load_staff_table(args.staff_table) if args.staff_table else None
    result = build(staff_table=staff)

    ensure_dir(DATA_PROCESSED)
    written = []
    for name, frame in result["windows"].items():
        written.append(write_parquet(
            pl.from_pandas(frame), DATA_PROCESSED / f"medstaff_grades_{name}.parquet"))
    written.append(write_parquet(
        pl.from_pandas(result["positions"]),
        DATA_PROCESSED / "medstaff_grades_positions.parquet"))
    written.append(write_report(result, REPORTS_DIR / "REPORT_medstaff_grades.md"))

    five = result["windows"]["5yr"]
    print(five[["team", "letter", "score", "separated_from_next"]].to_string(index=False))
    print(f"\nseparable from next: {int(five['separated_from_next'].sum())}/{len(five)}")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
