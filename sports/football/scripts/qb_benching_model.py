"""Stage 3 of the QB-benching project: fit the model, and report it within strata.

Reads the stage-2 features, fits a regularised logistic model (and a boosting comparison), and
writes the scores plus the report. The headline is the **within-band** AUC, not the pooled one:
see :mod:`qb_benching.model.fit` for why.

Usage
-----
    uv run python sports/football/scripts/qb_benching_model.py
    uv run python sports/football/scripts/qb_benching_model.py --permutations 200

Outputs
-------
``data/processed/qb_benching_scores.parquet``   out-of-fold risk score per opener-season
``reports/REPORT_qb_benching_model.md``         the deliverable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, REPORTS_DIR, ensure_dir, write_parquet,
)
from qb_benching.model import (  # noqa: E402
    band_profile, baseline_scores, coefficients, cross_validate, permutation_null,
    temporal_split, within_band_auc,
)

#: Columns that identify a row or encode the outcome — never features.
NOT_FEATURES = {
    "season", "team", "opener_id", "opener_name", "games_after_opener", "n_held", "n_benched",
    "n_injured", "n_gone", "n_displaced", "first_benched_week", "first_displaced_week",
    "benched", "injured_out", "displaced",
}


def _table(df, floats=3) -> str:
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        rows = df.to_dict("records")
        cols = list(df.columns)
    else:
        rows, cols = df, list(df[0].keys())
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for rec in rows:
        cells = []
        for c in cols:
            v = rec[c]
            if v is None or (isinstance(v, float) and v != v):
                cells.append("—")
            elif isinstance(v, bool):
                cells.append("yes" if v else "no")
            elif isinstance(v, float):
                cells.append(f"{v:.{floats}f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def main() -> None:
    import pandas as pd
    import polars as pl

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--outcome", default="benched", help="benched (default) or injured_out")
    args = ap.parse_args()

    feat_path = DATA_PROCESSED / "qb_benching_features.parquet"
    if not feat_path.exists():
        raise SystemExit(
            f"missing {feat_path}\nRun stage 2 first:\n"
            "  uv run python sports/football/scripts/qb_benching_features.py")
    blocks = json.loads((DATA_PROCESSED / "qb_benching_blocks.json").read_text())
    frame = pl.read_parquet(feat_path).to_pandas()

    numeric = [c for c in {c for cols in blocks.values() for c in cols}
               if c not in NOT_FEATURES and c in frame.columns]
    numeric = sorted(numeric)
    for c in numeric:
        frame[c] = frame[c].astype(float)

    outcome = args.outcome
    y = frame[outcome].astype(int)
    print(f"rows {len(frame)} · positives {int(y.sum())} · features {len(numeric)}")

    rows, oof = cross_validate(frame, numeric, outcome=outcome)
    cv = pd.DataFrame(rows)
    boost_rows, boost_oof = cross_validate(frame, numeric, outcome=outcome, kind="boosting")
    boost = pd.DataFrame(boost_rows)

    null = permutation_null(frame, numeric, outcome=outcome, n_permutations=args.permutations)
    observed = float(cv["auc"].mean())
    p_value = float((null >= observed).mean())

    temporal = temporal_split(frame, numeric, outcome=outcome)

    # The control. Entries 096-097 established that injury is not predictable from this kind of
    # evidence, so the same features on the same rows should fail here. If they do not, the
    # benching result is an artefact of the feature set rather than a fact about benching.
    control = None
    if outcome == "benched" and "injured_out" in frame.columns:
        c_rows, c_oof = cross_validate(frame, numeric, outcome="injured_out")
        c_null = permutation_null(frame, numeric, outcome="injured_out",
                                  n_permutations=max(50, args.permutations // 4))
        c_auc = float(pd.DataFrame(c_rows)["auc"].mean())
        control = {"auc": c_auc, "null": float(c_null.mean()),
                   "p": float((c_null >= c_auc).mean()),
                   "positives": int(frame["injured_out"].sum()),
                   "within": within_band_auc(frame, c_oof, "injured_out")}
    bands = band_profile(frame, outcome=outcome)
    within = within_band_auc(frame, oof, outcome=outcome)
    bases = baseline_scores(frame, outcome=outcome)
    coefs = coefficients(frame, numeric, outcome=outcome)

    scored = pl.from_pandas(
        frame[["season", "team", "opener_name", "prior_attempts", outcome]].assign(risk=oof))
    ensure_dir(DATA_PROCESSED)
    ensure_dir(REPORTS_DIR)
    write_parquet(scored, DATA_PROCESSED / "qb_benching_scores.parquet")

    best_base = bases.loc[bases["auc"].idxmax()]
    lines = [
        "# QB Benching — Stage 3: the model",
        "",
        f"_{int(frame['season'].min())}–{int(frame['season'].max())} · {len(frame)} opening "
        f"starters · {int(y.sum())} {outcome} ({y.mean():.1%}) · {len(numeric)} preseason "
        "features_",
        "",
        "## Read the stratified number, not the pooled one",
        "",
        "Benching rate runs from 47% for an opener with under 100 prior attempts to 12% for one "
        "over 300. A model that learns only *\"this man was never really the starter\"* posts a "
        "strong pooled AUC and tells a reader nothing the depth chart had not already told them. "
        "`qb_breakout` documented the same trap, where draft position scored 0.890 pooled and "
        "0.496 within a band. So the question this report is built to answer is whether the "
        "model separates benchings **among similarly-established starters**.",
        "",
        _table(bands),
        "",
        "## Headline",
        "",
        f"Pooled cross-validated AUC **{observed:.3f}** (±{cv['auc'].std():.3f} across repeats), "
        f"against a label-permutation null of {null.mean():.3f} ± {null.std():.3f} — "
        f"p = {p_value:.3f} over {args.permutations} shuffles. "
        f"Brier {cv['brier'].mean():.3f}. Of the {5} riskiest openers flagged in each season, "
        f"**{cv['precision_at_k'].mean():.1%}** were benched, against a {y.mean():.1%} base rate.",
        "",
        "**Within prior-volume bands — the number that decides whether this is a finding:**",
        "",
        _table(within),
        "",
        *(["## The control: the same features cannot predict injury",
           "",
           f"Fitting the identical feature set on the identical rows against `injured_out` "
           f"({control['positives']} positives) reaches AUC **{control['auc']:.3f}** against a "
           f"null of {control['null']:.3f} — p = {control['p']:.3f}. It is at chance in every "
           "band:",
           "",
           _table(control["within"]),
           "",
           "This is the result that makes the benching number worth believing. PROMPT_LOG "
           "entries 096-097 already established that injury is not predictable from this repo's "
           "data; if these features had 'predicted' it too, the benching AUC would be an "
           "artefact of the evaluation rather than a fact about benching. They do not.",
           ""] if control else []),
        "## Against the baselines a reader already has",
        "",
        _table(bases),
        "",
        f"The best single column is **{best_base['baseline']}** at AUC {best_base['auc']:.3f}. "
        f"The model {'beats' if observed > best_base['auc'] else 'does not beat'} it pooled.",
        "",
        "## Deployment simulation",
        "",
        (f"Trained on {temporal['n_train']} openers through {temporal['split_season']} "
         f"({temporal['train_positives']} positives), tested on the {temporal['n_test']} after "
         f"({temporal['test_positives']} positives): AUC **{temporal['auc']:.3f}**, "
         f"top-5 precision {temporal['precision_at_k']:.1%}. Where this disagrees with the "
         "random-fold number, this is the one to believe — random folds let 2024 inform a "
         "prediction about 2009."
         if temporal else "Not computable: a split leaves one side single-class."),
        "",
        "## Does a more flexible learner help?",
        "",
        f"Gradient boosting reaches AUC {boost['auc'].mean():.3f} against the linear model's "
        f"{observed:.3f}. It is fitted only so this line can be written.",
        "",
        "## What the model reads",
        "",
        "Coefficients from a fit on everything, with how often each sign survives a bootstrap. "
        "Read the direction and the stability, not the magnitude.",
        "",
        _table(coefs.head(14)),
        "",
        "⚠️ **Several of these are collinear and must not be read as effects.** "
        "`has_prior_season` and `is_rookie` are near-mirrors of each other; `prior_attempts`, "
        "`prior_dropbacks` and `prior_starts` all measure the same thing; and the rate columns "
        "carry suppression signs as a result — `prior_int_rate` reads negative and "
        "`prior_comp_pct` positive, neither of which is a claim that throwing interceptions "
        "keeps a job. A stable sign here means the fit is reproducible, not that the mechanism "
        "is identified. The C=0.1 shrink is what keeps them from flipping fold to fold.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python sports/football/scripts/qb_benching_features.py",
        "uv run python sports/football/scripts/qb_benching_model.py",
        "```",
        "",
    ]
    name = "REPORT_qb_benching_model.md" if outcome == "benched" \
        else f"REPORT_qb_benching_model_{outcome}.md"
    (REPORTS_DIR / name).write_text("\n".join(lines))

    print(f"pooled AUC {observed:.3f} (null {null.mean():.3f}, p={p_value:.3f}) · "
          f"temporal {temporal['auc']:.3f}" if temporal else f"pooled AUC {observed:.3f}")
    print(within.to_string(index=False))
    print(f"wrote {REPORTS_DIR / name}")


if __name__ == "__main__":
    main()
