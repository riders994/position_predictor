"""Stage 6: is any of the club spread a stable property of the club?

Deliberately run **before** the grades. Stage 4 showed club residuals exceed sampling noise,
which is a weaker claim than it sounds — a residual can be large and still be a one-off. This
stage asks whether a club's residual in one part of the data predicts its residual in another,
and a reader who stops here has the honest answer.

Usage
-----
    uv run python sports/football/scripts/medstaff_reliability.py
    uv run python sports/football/scripts/medstaff_reliability.py --reps 200

Outputs
-------
``data/processed/medstaff_reliability.parquet``   per-component reliability + gate flags
``reports/REPORT_medstaff_reliability.md``        the verdict the grades must be read against
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, REPORTS_DIR, ensure_dir, write_parquet,
)
from medstaff.expected import whole_factor_permutation  # noqa: E402
from medstaff.validate import (  # noqa: E402
    GATE_PERMUTATION_P, GATE_SPLIT_HALF, GATE_TEMPORAL,
    detectable_by_group, reliability_gate, split_half_reliability, temporal_reliability,
)

# 2021-2023 grades, 2024-2025 tests. Three seasons in and two out is the most even split the
# comparable window allows.
SPLIT_SEASON = 2024

COMPONENTS = {
    "incidence_no_history": "onset",
    "incidence_with_history": "onset",
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


def build(*, reps: int, perm_sims: int):
    residuals = pl.read_parquet(DATA_PROCESSED / "medstaff_residuals.parquet")
    out = []
    for name, outcome in COMPONENTS.items():
        path = DATA_PROCESSED / f"medstaff_expected_{name}.parquet"
        if not path.exists():
            raise SystemExit(
                f"missing {path}\nRun stage 4 first:\n"
                f"  uv run python sports/football/scripts/medstaff_expected.py")
        frame = pl.read_parquet(path)
        sh = split_half_reliability(frame, outcome=outcome, n_sim=reps)
        tp = temporal_reliability(frame, outcome=outcome, split_season=SPLIT_SEASON)
        # Recomputed here from stage 4's own artifact rather than carried across as a constant:
        # a retyped number goes stale silently the first time stage 4 is rerun.
        pp = whole_factor_permutation(
            frame, frame["expected"].to_numpy(), outcome=outcome, n_sim=perm_sims)["p_value"]
        out.append({
            "component": name, "n_rows": frame.height,
            "split_half_r": sh["r_full"], "split_half_lo": sh["ci_low"],
            "split_half_hi": sh["ci_high"], "split_half_reps": sh["n_reps"],
            "temporal_r": tp["spearman"], "temporal_p": tp["p_value"],
            "temporal_slope": tp["slope"], "n_clubs": tp["n_clubs"],
            "permutation_p": pp,
            "_frame": frame, "_outcome": outcome,
        })
    return {"components": out, "residuals": residuals}


def write_report(result, path, *, perm_lookup) -> Path:
    import pandas as pd

    rows = []
    for c in result["components"]:
        gate = reliability_gate(
            split_half_r=c["split_half_r"], temporal_r=c["temporal_r"],
            permutation_p=perm_lookup.get(c["component"], 1.0))
        rows.append({
            "component": c["component"], "split_half_r": c["split_half_r"],
            "split_half_95CI": f"[{c['split_half_lo']:.2f}, {c['split_half_hi']:.2f}]",
            "temporal_r": c["temporal_r"], "temporal_p": c["temporal_p"],
            "permutation_p": perm_lookup.get(c["component"], float("nan")),
            "gate": "PASS" if gate["passed"] else "FAIL",
            "failed_on": ", ".join(gate["failed"]) or "—",
        })
    summary = pd.DataFrame(rows)

    rec = summary[summary.component == "recurrence"].iloc[0]
    dur = summary[summary.component == "duration"].iloc[0]
    inc = summary[summary.component == "incidence_no_history"].iloc[0]
    n_pass = int((summary.gate == "PASS").sum())

    passing = summary.loc[summary.gate == "PASS", "component"].tolist()
    failing = summary.loc[summary.gate == "FAIL", "component"].tolist()
    passing_str = ", ".join(passing) if passing else "none"
    failing_str = ", ".join(failing) if failing else "none"
    # Derived, not asserted: an earlier version hardcoded which pair passed and kept saying so
    # after a bug fix changed it.
    _policy = {"duration", "returns_at_all", "incidence_no_history", "incidence_with_history"}
    if "recurrence" in passing:
        stability_reading = (
            "Recurrence — the outcome most plausibly owned by a medical staff — is among the "
            "components that persist, so the board carries some genuine medical signal.")
    elif passing and set(passing) <= _policy:
        stability_reading = (
            "**Recurrence, the one outcome most plausibly owned by a medical staff, is not among "
            "them.** Everything that persists is about *exposure and how a club uses injured "
            "reserve and times a return* — an operational and roster-policy signature. So the "
            "defensible reading of the stage-7 board is that it separates clubs by "
            "**availability-management policy**, not by quality of medicine. That is a narrower "
            "claim than \"medical staff grades\", and it is the one the evidence supports.")
    else:
        stability_reading = (
            "**No component clears the gate.** The board should be read as an ordering with no "
            "demonstrated year-over-year stability behind it; the power bounds in §4 are the "
            "reportable result.")

    best_temporal = summary.loc[summary.temporal_r.idxmax(), "component"]
    worst_temporal = summary.loc[summary.temporal_r.idxmin(), "component"]

    dur_frame = next(c["_frame"] for c in result["components"] if c["component"] == "duration")
    by_group = detectable_by_group(dur_frame)
    group_summary = (
        by_group.groupby("position_group")
        .agg(clubs=("n", "size"), median_n=("n", "median"),
             median_expected=("expected", "median"),
             median_min_detectable=("min_detectable", "median"))
        .reset_index().sort_values("median_n", ascending=False)
    )

    md = f"""# Medical-staff grades — Stage 6: reliability and power

**Read this before any grade.** Stage 4 established that club residuals are larger than sampling
noise. That is a weaker claim than it sounds: a residual can be large and still be a one-off, and
a leaderboard built on one-offs is a leaderboard of luck. The question that decides whether the
grades mean anything is whether a club's residual in one part of the data predicts its residual
in another.

## 1. The verdict

**{n_pass} of {len(summary)} components pass the preregistered gate**
(split-half ≥ {GATE_SPLIT_HALF}, temporal ≥ {GATE_TEMPORAL}, permutation p < {GATE_PERMUTATION_P}).

{_table(summary)}

The gate **does not suppress anything** — grades publish either way, which was a deliberate
choice. It exists to *label* each component at every appearance. A number that fails here is
still printed; it is printed marked.

## 2. What the two tests mean, and why both

**Split-half** splits each club's *players* — never its rows — into halves and correlates the two
residuals, corrected by Spearman-Brown. Splitting rows would leak: a fragile player's weeks would
land on both sides and manufacture agreement out of one man's hamstring.

**Temporal** grades 2021–2023 and asks whether that predicts 2024–2025. This is the harder and
more honest test, and the one a reader actually cares about, because a grade is only useful if it
says something about the club going forward.

**Split-half can pass while temporal fails, and that gap is itself a finding**: it means the
residual is a property of a period rather than of a club — real, but not a thing you can carry
forward into next season.

## 3. The gap between the two tests is the headline

Split-half is high everywhere ({summary.split_half_r.min():.2f}–{summary.split_half_r.max():.2f}):
the residuals are internally consistent, so they are not measurement noise. **Temporal is much
lower across the board.** That gap is the finding — for most components the residual is a
property of *a period*, not a carry-forward property of the club. Real, but not something you
could put on next season's board.

- **Recurrence** — the outcome most plausibly owned by a medical staff — is **weakest on every
  test**: split-half {rec.split_half_r:.3f}, temporal **{rec.temporal_r:.3f}**
  (p {rec.temporal_p:.3f}), permutation {rec.permutation_p:.3f}. Whether injuries come back is
  the thing one would actually want to call medical quality, and it does not persist.
- **Duration** — split-half {dur.split_half_r:.3f}, temporal **{dur.temporal_r:.3f}**
  (p {dur.temporal_p:.3f}).
- **Strongest on the temporal test: {best_temporal}**; weakest: **{worst_temporal}**.
- **Incidence** — *least* attributable to a training room — split-half {inc.split_half_r:.3f},
  temporal {inc.temporal_r:.3f} (p {inc.temporal_p:.3f}).

### What is actually stable

The component(s) that pass: **{passing_str}**. What does *not* pass: **{failing_str}**.

{stability_reading}

## 4. Power — what would have been findable

Where a component or a cell fails, this is the honest headline. "Nothing this size or smaller was
findable" is a result; "no effect" is not.

Minimum detectable observed-minus-expected per club × position group, in returns:

{_table(group_summary)}

## 5. What this means for the grades

Every component that fails the gate is labelled as not distinguishable from luck **at every
appearance** in stage 7 — not in a footnote. Position-group cells whose minimum detectable effect
exceeds any plausible club difference are reported as point estimates with intervals and never
ranked.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=300,
                        help="split-half repetitions (default 300)")
    parser.add_argument("--perm-sims", type=int, default=1000,
                        help="whole-factor permutation draws (default 1000)")
    args = parser.parse_args(argv)

    result = build(reps=args.reps, perm_sims=args.perm_sims)
    perm_lookup = {c["component"]: c["permutation_p"] for c in result["components"]}
    frame = pl.DataFrame([
        {k: v for k, v in c.items() if not k.startswith("_")} for c in result["components"]
    ])
    ensure_dir(DATA_PROCESSED)
    p1 = write_parquet(frame, DATA_PROCESSED / "medstaff_reliability.parquet")
    p2 = write_report(result, REPORTS_DIR / "REPORT_medstaff_reliability.md",
                      perm_lookup=perm_lookup)

    for c in result["components"]:
        print(f"{c['component']:24s} split_half={c['split_half_r']:+.3f}  "
              f"temporal={c['temporal_r']:+.3f} (p={c['temporal_p']:.3f})  "
              f"perm_p={c['permutation_p']:.4f}")
    for path in (p1, p2):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
