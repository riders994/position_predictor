"""Stage 4: expectation models — what each club's injury outcomes should have looked like.

Fits the three components (incidence, duration, recurrence) plus the returns-at-all guard, all
with leave-one-team-out folds so a club can never inform its own expectation, and produces the
observed-minus-expected residuals every grade is built from.

Nothing here is a grade. Stage 6 decides whether the residuals are distinguishable from luck;
stage 7 turns them into letters.

Usage
-----
    uv run python sports/football/scripts/medstaff_expected.py
    uv run python sports/football/scripts/medstaff_expected.py --sims 500   # faster nulls

Outputs
-------
``data/processed/medstaff_expected_<component>.parquet``  per-row out-of-fold expectations
``data/processed/medstaff_residuals.parquet``             club x component observed-minus-expected
``reports/REPORT_medstaff_expected.md``                   fit, calibration, and the bound
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
    DATA_PROCESSED, DATA_RAW, REPORTS_DIR, ensure_dir, write_parquet,
)
from medstaff.data import load_rosters_weekly  # noqa: E402
from medstaff.episodes import PRIMARY_HORIZON, HORIZONS  # noqa: E402
from medstaff.expected import (  # noqa: E402
    DURATION_CATEGORICAL, DURATION_NUMERIC, HISTORY_NUMERIC, INCIDENCE_CATEGORICAL,
    INCIDENCE_NUMERIC, RECURRENCE_CATEGORICAL, RECURRENCE_NUMERIC,
    detectable_effect, duration_frame, fit_predict_loto, incidence_frame, overdispersion,
    player_block_null, poisson_binomial_null, recurrence_frame, returns_at_all_frame,
    team_observed_expected, two_sided_p, whole_factor_permutation,
)
from medstaff.exposure import player_attributes  # noqa: E402


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


def _calibration(y, p, *, bins: int = 10):
    """Predicted vs observed by decile of predicted risk."""
    import pandas as pd

    order = np.argsort(p)
    edges = np.array_split(order, bins)
    rows = []
    for i, idx in enumerate(edges, start=1):
        if len(idx) == 0:
            continue
        rows.append({"decile": i, "n": len(idx),
                     "predicted": float(p[idx].mean()), "observed": float(y[idx].mean())})
    return pd.DataFrame(rows)


def _component(name, frame, numeric, categorical, outcome, *, sims, seed=17):
    """Fit one component out-of-fold and summarise its club residuals."""
    p = fit_predict_loto(frame, numeric, categorical, outcome)
    oe = team_observed_expected(frame, p, outcome=outcome)

    y = np.asarray(frame[outcome]).astype(int)
    pdf = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    teams = np.asarray(pdf["team"])
    units = np.asarray(pdf["gsis_id"])

    p_indep, p_block = [], []
    for club in oe["team"]:
        mask = teams == club
        obs = int(y[mask].sum())
        indep = poisson_binomial_null(p[mask], n_sim=sims, seed=seed)
        block = player_block_null(p[mask], units[mask], n_sim=max(sims // 4, 250), seed=seed)
        p_indep.append(two_sided_p(obs, indep))
        # the project uses whichever null is more conservative
        p_block.append(two_sided_p(obs, block))
    oe["p_indep"] = p_indep
    oe["p_block"] = p_block
    oe["p_used"] = np.maximum(oe["p_indep"], oe["p_block"])
    oe["component"] = name

    perm = whole_factor_permutation(frame, p, outcome=outcome, n_sim=min(sims, 2000))
    disp = overdispersion(oe["diff"].to_numpy(), oe["var_indep"].to_numpy())
    power = detectable_effect(oe["var_indep"].to_numpy())

    return {"name": name, "expected": p, "oe": oe, "permutation": perm,
            "dispersion": disp, "power": power, "calibration": _calibration(y, p),
            "base_rate": float(y.mean()), "n": len(y), "frame": frame, "outcome": outcome}


def build(*, sims: int):
    exposure = pl.read_parquet(DATA_PROCESSED / "medstaff_exposure.parquet")
    episodes = pl.read_parquet(DATA_PROCESSED / "medstaff_episodes.parquet")
    panel = pl.read_parquet(DATA_PROCESSED / "medstaff_panel.parquet")
    seasons = tuple(sorted(episodes["season"].unique().to_list()))
    attrs = player_attributes(load_rosters_weekly(DATA_RAW, seasons=seasons))

    inc = incidence_frame(exposure, episodes)
    dur = duration_frame(episodes, attrs)
    rec = recurrence_frame(episodes, panel, horizon=HORIZONS[PRIMARY_HORIZON],
                           attributes=attrs)
    ret = returns_at_all_frame(episodes, attrs)

    results = {}
    # Incidence is fitted TWICE — see the report. Prior-injury history is partly the club's own
    # output, so adjusting for it adjusts away part of the effect being measured.
    results["incidence_no_history"] = _component(
        "incidence_no_history", inc, INCIDENCE_NUMERIC, INCIDENCE_CATEGORICAL, "onset",
        sims=sims)
    results["incidence_with_history"] = _component(
        "incidence_with_history", inc, (*INCIDENCE_NUMERIC, *HISTORY_NUMERIC),
        INCIDENCE_CATEGORICAL, "onset", sims=sims)
    results["duration"] = _component(
        "duration", dur, DURATION_NUMERIC, DURATION_CATEGORICAL, "returned", sims=sims)
    results["recurrence"] = _component(
        "recurrence", rec, RECURRENCE_NUMERIC, RECURRENCE_CATEGORICAL, "recurred", sims=sims)
    results["returns_at_all"] = _component(
        "returns_at_all", ret, DURATION_NUMERIC, DURATION_CATEGORICAL, "returned_at_all",
        sims=sims)
    return results


def write_report(results, path: Path) -> Path:
    import pandas as pd

    summary = pd.DataFrame([{
        "component": r["name"], "rows": r["n"], "base_rate": r["base_rate"],
        "permutation_p": r["permutation"]["p_value"],
        "intraclass": r["dispersion"]["intraclass"],
        "signal_sd": r["dispersion"]["sd_signal"],
        "min_detectable": r["power"]["min_detectable"],
    } for r in results.values()])

    lo = results["incidence_with_history"]["dispersion"]["intraclass"]
    hi = results["incidence_no_history"]["dispersion"]["intraclass"]
    rec, dur = results["recurrence"], results["duration"]

    def extremes(r, k=5):
        oe = r["oe"].sort_values("diff")
        return pd.concat([oe.head(k), oe.tail(k)])[
            ["team", "n", "observed", "expected", "diff", "z_indep", "p_used"]]

    md = f"""# Medical-staff grades — Stage 4: expectation models

Every grade in this project is observed minus expected. This stage builds the expected side and
reports the residuals. **Nothing here is a grade** — stage 6 decides whether these residuals are
distinguishable from luck, and only then does stage 7 turn them into letters.

## 1. Why this is a penalised logistic hazard and not a boosted tree

The estimand is a **residual**. A flexible learner eats residual variance: a boosted model
adjusting on roster-composition proxies absorbs precisely the between-club variation being
measured, and the grades would then read as "how well does a GBM predict this club", not "how
did this club do". Penalised logistic regression with splines on age and week is the more
complex thing that is *less* dangerous here.

Censoring is pervasive — seasons end, players are traded and released — and discrete-time hazard
handles it exactly by construction. Duration and recurrence are person-period expansions, so a
spell that ran out of season contributes the weeks it actually had rather than being dropped
(which biases towards fast returns) or counted as a return (which is a lie).

## 2. Why leave-one-team-out

Club T is scored by a model fit on the other 31. A random fold contains some of T's own rows, so
the model learns T's elevated rate and predicts it back, shrinking the residual toward zero.
There is a test that injects a club effect of known size and asserts LOTO recovers it while
random k-fold shrinks it.

**Club identity is never a feature**, as a column or a proxy — there is a guard that raises if it
reaches the design matrix.

## 3. The components

{_table(summary)}

`permutation_p` is the **primary inference**: does club identity explain *any* variance in the
residual? One test rather than 32, so there is no multiplicity problem. The null reassigns whole
players to clubs, preserving each player's own correlated run of weeks.

`intraclass` is the share of between-club spread that survives after subtracting the sampling
variance the model already explains. **It is not the share attributable to a medical staff**, and
reading it that way is the single most likely misuse of this table. It says the spread is not
Poisson noise; the surviving variance can still be roster construction, disclosure behaviour,
scheme or stadium. The model here is deliberately restrained (§1), which means composition the
covariates do not capture stays *in* the residual rather than being absorbed — that is the right
trade for measuring a club effect, but it inflates this number relative to anything causal.

`min_detectable` is what a typical club would have had to differ by, in the outcome's own units,
to be distinguishable.

### The pattern worth reading first

**The components differ in how attributable they are, and they run in the opposite direction to
the evidence.** Recurrence is the outcome most plausibly owned by a medical staff — it is
downstream of the return-to-play decision and least contaminated by luck — and it is the **only
component that does not clear its permutation test**
(p {rec["permutation"]["p_value"]:.3f}, intraclass {rec["dispersion"]["intraclass"]:.3f}).
Incidence is the component *least* attributable to a training room — conditioning, scheme,
surface and luck — and it shows the strongest club separation.

That ordering is a caution, not a finding. It is consistent with clubs differing mainly in how
much football their rosters are exposed to and how they use injured reserve, rather than in
medicine. Stage 6 tests whether any of it is stable year over year, which is the question that
decides whether the stage-7 letters mean anything.

## 4. The bound that adjustment forces

Incidence is fitted **twice**, and both are reported, because prior injury history is not a
neutral covariate. A club with a poor availability system *manufactures* players who look
fragile, so history is partly its own output — adjusting for it adjusts away part of the effect
being measured.

- **without history** — intraclass **{hi:.3f}** — the *upper* bound on the club effect
- **with history** — intraclass **{lo:.3f}** — the *lower* bound

The truth is inside that interval and this data cannot say where. Reporting either number alone
would be a choice dressed as a measurement.

## 5. Calibration

Predicted against observed by decile of predicted risk. A model that is not on the diagonal here
produces residuals that are model error rather than club effect.

### Incidence (no history)

{_table(results["incidence_no_history"]["calibration"])}

### Recurrence

{_table(rec["calibration"])}

## 6. Club residuals

Most extreme five in each direction. **These are not grades** — the p-values are uncorrected,
there are 32 of them per component, and roughly 1.6 clubs per component clear 0.05 by chance
alone. `p_used` is the more conservative of the independent-Bernoulli and player-block nulls,
because a fragile player is fragile all season and his weeks are correlated.

### Recurrence — the most attributable component

{_table(extremes(rec))}

### Duration

{_table(extremes(dur))}

## 7. The censoring guard

`returns_at_all` is fitted as its own outcome rather than folded into duration. Without it a club
scores well on return-to-play simply by having its unrecovered cases quietly censored — the
failure mode is gameable, so it is measured directly.

Base rate: **{results["returns_at_all"]["base_rate"]:.3f}** of episodes resolved within the
season; permutation p **{results["returns_at_all"]["permutation"]["p_value"]:.3f}**.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sims", type=int, default=2000,
                        help="null simulation draws (default 2000)")
    args = parser.parse_args(argv)

    results = build(sims=args.sims)

    ensure_dir(DATA_PROCESSED)
    written = []
    for name, r in results.items():
        frame = r["frame"].with_columns(pl.Series("expected", r["expected"]))
        written.append(write_parquet(
            frame, DATA_PROCESSED / f"medstaff_expected_{name}.parquet"))
    residuals = pl.concat(
        [pl.from_pandas(r["oe"]) for r in results.values()], how="diagonal_relaxed")
    written.append(write_parquet(residuals, DATA_PROCESSED / "medstaff_residuals.parquet"))
    written.append(write_report(results, REPORTS_DIR / "REPORT_medstaff_expected.md"))

    for name, r in results.items():
        print(f"{name:24s} n={r['n']:>7,}  base={r['base_rate']:.3f}  "
              f"perm_p={r['permutation']['p_value']:.4f}  "
              f"intraclass={r['dispersion']['intraclass']:.3f}")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
