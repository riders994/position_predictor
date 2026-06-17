# Position Predictor — Project Plan

> Predict a pro sports player's **next-season fantasy rank** from prior-season data.
> One independent process per **sport**, one independent model per **position**.
> First target: **NFL — Running Back (RB)**.

This document is the canonical design for the project. It is intended to be read
top-to-bottom by a new contributor (or a future version of us) and fully explain
*what* we are building, *why*, and *how* it stays reproducible. Decisions here were
made collaboratively; see [`PROMPT_LOG.md`](./PROMPT_LOG.md) for the decision trail.

---

## 1. Problem statement

For a given position, use a player's historical NFL data through season *N* to
predict their **fantasy production in season *N+1***, then rank players by that
prediction. We evaluate how well our predicted ranking matches the actual
end-of-season ranking.

### 1.1 Locked decisions (RB v1)

| Decision | Choice | Rationale |
|---|---|---|
| Scoring format | **PPR** (1 pt / reception) | User's league format; receiving volume matters most here. |
| Target metric | **Next-season points-per-game (PPG)** → derived rank | Removes raw injury/availability noise from the *prediction* target; rank is computed from predicted PPG. |
| Ranking eligibility | Players who clear a **games-played / snap-share cutoff** | A high-PPG / low-games player still has value if they clear the bar. The cutoff is **derived analytically** (its own pipeline step), never hardcoded. |
| Player scope | **Returning players only** (≥1 prior NFL season) | Clean feature set from NFL history. Rookies (CFB/draft data) are **out of scope — not pursued**. |
| Data source | **nfl_data_py / nflverse** | Free, reproducible, deep (pbp, weekly, seasonal, rosters, snaps, draft, combine). |
| History | **Comparative 10 / 20 / 30-year training windows** | Measure recency bias per position rather than assume one window. |
| Market data (ADP/ECR) | **Benchmark only**, not a feature | Clean test of whether our stats-derived features add signal over the crowd. |

### 1.2 Non-goals (v1)

- Rookies / incoming players (no prior NFL season).
- In-season / weekly prediction (this is a preseason, full-season-ahead model).
- Other positions and sports — but the architecture is built so they slot in (§9).

---

## 2. Data

### 2.1 Recommended source — nflverse via `nfl_data_py`

| Dataset (`nfl_data_py` loader) | Contents | Coverage |
|---|---|---|
| `import_seasonal_data` | Season totals per player (rush/rec/TD/fantasy) | 1999– |
| `import_weekly_data` | Per-game box scores (used to compute PPG, games played) | 1999– |
| `import_seasonal_rosters` / `import_rosters` | Age, position, team, height/weight, experience | 1999– |
| `import_snap_counts` | Offensive snaps & snap share | **2012–** |
| `import_ngs_data` (Next Gen Stats) | Advanced rushing/receiving (efficiency over expected) | 2016– |
| `import_pbp_data` | Play-by-play → EPA, success rate, route/usage proxies, red-zone/goal-line | 1999– |
| `import_draft_picks` / `import_combine_data` | Draft capital, athletic profile | 1980s– |
| `import_ids` | Cross-source player ID crosswalk | — |

**Data-availability caveats (the comparative windows must respect these):**
- Snap counts begin **2012** → the snap-share component of the eligibility cutoff and any snap features only exist for 2012+.
- NGS begins **2016** → advanced-efficiency features only for 2016+; treated as optional/secondary feature block with explicit missingness handling.
- The **30-year window is data-limited to ~1999–present (~27 seasons)**; we document and accommodate the limit rather than fabricate older data.

### 2.2 Alternative sources (considered, not selected)

- **Pro-Football-Reference (scrape):** richer human-readable advanced stats (broken tackles, etc.) but scraping/TOS/rate-limit burden and weaker reproducibility. *Possible future enrichment source.*
- **Sleeper / FantasyPros APIs:** best for ADP / expert consensus rank → used **only** to build the benchmark (§7.4), not core stats. (Sleeper is also used for fantasy-position eligibility, §2/§4.)

### 2.3 Benchmark data

Prior-year ADP and/or expert consensus rank (ECR) for the test seasons, used purely
as a baseline ranking to beat. Stored in `data/external/`.

### 2.4 Reproducibility of data

- Raw pulls are cached to `data/raw/` as Parquet and **never edited in place**.
- Each pull records a manifest (source, loader, seasons, pull date, row counts, content hash) in `data/raw/_manifests/`.
- `data/` is git-ignored except for `external/` reference files and manifests.

---

## 3. Pipeline (reproducible experiment stages)

Each stage is a deterministic, config-driven script under `scripts/`, importing logic
from `src/position_predictor/`. Notebooks consume the same library code (no copy-paste
logic) so analysis and pipeline never diverge.

```
[1] fetch        scripts/fetch_data.py       nflverse → data/raw/*.parquet (+ manifest)
[2] build        scripts/build_dataset.py    join sources → player-season table (data/interim)
[3] target       (in build)                  next-season PPG + eligibility candidate labels
[4] features     scripts/build_features.py   engineered features → data/processed
[5] eligibility  scripts/run_eligibility.py  derive games/snap cutoff (analysis + chosen rule)
[6] eda          notebooks/.../01,02         coverage, distributions, target stability
[7] feat-select  notebooks/.../03 + module   correlation/VIF, importance, stability selection
[8] model        scripts/run_experiment.py   walk-forward CV × {10,20,30}yr windows × models
[9] report       scripts/make_report.py      metrics tables + figures → reports/
```

Orchestrated by a `Makefile` (e.g. `make fetch`, `make features`, `make experiment`)
so the whole thing runs end-to-end from a clean checkout.

---

## 4. Target & eligibility cutoff (the headline methodology)

### 4.1 Target

For each player-season row (player, season *N*), the label is their **season *N+1* PPR
PPG** = total PPR points in *N+1* ÷ games played in *N+1*. Modeling is **regression on
PPG**; ranking is derived by sorting predicted PPG among eligible players.

### 4.2 Eligibility cutoff — a first-class, data-driven step (every football model)

The cutoff defines the **evaluation universe**: who is "rankable" at season end. It is
derived, documented, and sensitivity-tested — not assumed.

**Candidate cutoff dimensions:** games played (`G`) and snap share (`snap%`, 2012+).

**Derivation methods (analyzed in notebook `02_eligibility_cutoff_analysis`):**
1. **Reliability / signal stabilization** — split-half and *k*-game vs full-season PPG
   correlation: find the minimum `G` where PPG becomes a *reliable* estimate of true
   per-game skill (variance stabilizes; low-game PPG is high-variance noise).
2. **Replacement / usable-starter level** — where snap share / opportunity separates
   fantasy-relevant backs from committee fringe (distribution elbow).
3. **Coverage vs purity trade-off** — how many real fantasy contributors a cutoff
   excludes vs how much noise it admits.

**Output:** a documented rule (e.g. `G ≥ g*` and/or `snap% ≥ s*`) **plus a sensitivity
grid** — all downstream rank metrics are reported across a small grid of cutoffs so
conclusions are shown to be robust to the exact threshold. The chosen rule is recorded
in the experiment config and the results.

### 4.3 Availability / injury model — required for every football model

Injuries materially shape fantasy value, so the football process includes a **second,
parallel supervised model** that predicts **season *N+1* availability** from numeric
history. It runs alongside the PPG model on the same feature table.

- **Label:** N+1 **games played** (count, 0–17), modeled as a count target
  (Poisson / negative-binomial gradient boosting or quantile GBM). A derived binary
  "**projected to clear the games cutoff**" is read off the count prediction.
- **Predictors (all numeric):** durability history (games played/missed per prior
  season, multi-year availability rate, consecutive-availability streaks), **workload /
  wear proxies** (career & recent-window touches, touches/game, snap load — high volume
  is an injury-risk proxy), and **age**. No injury-report text or qualitative tags.
- **Integration:** projected games gates the **projected eligibility universe** (who we
  expect to clear the cutoff in N+1) and is reported as a per-player **injury-risk** read.
  The primary ranking remains predicted **PPG**; availability determines who is rankable
  and surfaces reinjury risk. (A secondary projected-total board = projected PPG ×
  projected games can be reported but is not the primary objective.)
- **Evaluation:** games-played MAE; and classification metrics (AUC / precision-recall)
  for the "clears the cutoff" call, per fold and per history window.

---

## 5. Features (RB)

> **Quantitative-only principle.** Every feature must be a **number derived from box
> scores, play-by-play, snaps, rosters, or athletic testing**. No qualitative or
> hand-labeled inputs (e.g. "new coach", "scheme change", reputation tiers). Where a
> qualitative effect matters, we approximate it with a **measurable stat proxy** (e.g.
> a scheme/usage shift is captured by year-over-year changes in team run/pass rate,
> personnel/pace, and a back's snap- and route-share deltas — not a "coaching change"
> flag). This may cost some accuracy; that trade-off is accepted by design.

All features are computed **only from seasons ≤ N** (strict no-leakage rule, §6.3) — with one
deliberate, leakage-safe exception: the **`offseason` block** (added 2026-06-16) reads the *N+1*
**preseason** state (the April draft + the preseason roster), gated at a **Sept 1 cutoff** so it
predates any season-*N+1* game. This is information available **at draft time** — and exactly what
ECR has — so it is fair to use without "blending" the market in: `changed_team_next`, the draft
capital of a rookie RB the team added, and the proven workload of the *other* RBs now in the
backfield. It stays quantitative-only (every value is a number from `draft_picks`/`rosters`).
Blocks below; exact list maintained in `docs/data_dictionary.md`.

- **Volume / usage:** carries, targets, touches, snap share, route participation,
  target share, rush-attempt share, red-zone & goal-line touches, opportunity share,
  weighted opportunities.
- **Efficiency:** yards/carry, yards/touch, yards/route-run, catch rate, rush EPA/play,
  success rate, explosive-run rate; NGS efficiency-over-expected (2016+).
- **Production (prior):** PPR PPG, total PPR points, rush/rec yards & TDs, receptions,
  positional finish.
- **Team context (numeric proxies only):** team plays/game & pace, run/pass ratio, team
  rush success rate / rush EPA (O-line proxy), team passing efficiency (QB-quality proxy
  via passer numbers, not reputation), **scheme/usage-shift proxies** = YoY changes in
  team run/pass rate, personnel/pace, and the back's snap- & route-share deltas (this
  replaces any "coaching change" flag), backfield competition (teammate touch / snap
  share, RB-room churn measured by share concentration).
- **Player attributes:** **age** (RB age cliff — key), experience, height/weight, draft
  capital (pick number), combine athletic profile (numeric testing).
- **Trajectory:** YoY deltas and multi-year slopes of usage/efficiency/production;
  age-curve position.
- **Regression-to-mean signals:** TD rate vs expected TDs, YPC vs expected, receiving-TD
  luck — flag unsustainable production.
- **Availability / durability:** games played / missed and multi-year availability rate,
  workload/wear proxies (career & recent touches, touches/game, snap load). Used both as
  PPG features and as the **inputs to the availability model (§4.3)** — kept strictly
  distinct from the *label* season.

---

## 6. Validation & experimental design

### 6.1 Walk-forward (temporal) cross-validation

Train on seasons ≤ *N*, predict *N+1*. Expanding/rolling origin across the test seasons.
**No random K-fold** — that leaks the future. Metrics are computed per test season
(fold) and aggregated with dispersion (mean ± across folds).

### 6.2 The 10 / 20 / 30-year comparison (recency-bias study)

Hold the **test seasons fixed** (most recent *K* seasons). For each, train three models
that differ only in how much **history** the training set spans — 10, 20, 30 years back —
and compare metrics. If 10yr ≈/> 30yr, older data is stale (recency bias / regime drift);
if 30yr wins, long history helps. This is run for every position, every model family.

### 6.3 Era-separate modeling (feature-availability eras)

nflverse coverage changes over time (box scores 1999+, snaps 2012+, NGS 2016+), so a single
model trained across a long window faces **structurally** missing columns for older seasons.
Rather than impute features that did not exist, we partition seasons into **eras** — contiguous
spans sharing one available feature schema — and train **one model per era**:

| Era | Feature seasons | Adds | Feature blocks |
|---|---|---|---|
| `boxscore` | 1999–2011 | box scores, weekly EPA | volume, efficiency, production, player_attrs, trajectory, regression_mean, availability |
| `snaps` | 2012–2015 | snap-share usage | + snap_usage |
| `ngs` | 2016–present | Next Gen Stats over-expected | + ngs_efficiency |

The schemas are **nested** (each era only *adds* blocks). This is the key property: an
older-era model uses a strict subset of the newer eras' features, so it can score a modern
player by simply using that subset — no fabricated/imputed structural columns.

**An X-year window model = the composition of the era models whose seasons fall in the
window** (config `era_modeling`, code `src/position_predictor/eras.py`):

- A row is routed to an era model by its **feature season N** (`route_by: feature_season`).
- `era_segments_for_window(window_years, test_seasons)` returns, for a window, each era and the
  feature-season segment it trains on. A 30-yr window composes all three era models; a 10-yr
  window mostly the `ngs` model (plus the tail of `snaps`/`boxscore` it reaches).
- At prediction the target is a recent, full-feature season; **each** constituent era model
  scores it using its own feature subset, and the per-era predictions are combined
  (`era_modeling.combine`, default `val_weighted` — weight each era model by its walk-forward
  validation score so the data decides how much older eras help; `mean`/`recency_weighted` are
  also reported). This makes the §6.2 recency-bias comparison a direct test of whether
  older-era relationships still carry signal.

This replaces the earlier open question of "impute vs. mask vs. separate" for era-limited
features: **separate**, with nested schemas and window-level composition.

### 6.4 Leakage controls

- Features use only data through season *N*; the label is season *N+1*.
- Eligibility cutoff and any normalization/imputation are fit on training folds only.
- Training windows exclude the fixed test block; window bounds are computed in feature seasons
  with the horizon subtracted (`train_window_bounds`) so no label-season leaks into training.
- Player identity is consistent via the nflverse ID crosswalk.

---

## 7. Models

### 7.1 Baselines (must-beat floor)

- **Persistence:** next-season PPG = this-season PPG.
- **Smoothed history:** weighted average of last *n* seasons' PPG.
- **Mean reversion:** regress prior PPG toward positional mean.
- **Linear regression** on a small core feature set.

### 7.2 Candidate models

- **Regularized linear:** Ridge / Lasso / ElasticNet (interpretable, strong baseline).
- **Tree ensembles:** Random Forest, **gradient boosting (LightGBM / XGBoost / CatBoost)** — primary workhorse for tabular.
- **Learning-to-rank (stretch):** LightGBM LambdaMART directly optimizing rank, compared against the regress-then-sort approach.
- **Quantile / interval (stretch):** quantile GBM for floor–ceiling ranges.

### 7.3 Era ensemble (window composition, §6.3)

Each candidate family (§7.2) is fit **once per era** on that era's nested feature schema. For a
given history window, the per-era models are combined into a single window prediction by the
configured combiner (`val_weighted` default). Baselines (§7.1) are era-agnostic (they need only
prior PPG / games) and serve as the must-beat floor for the era ensemble at every window. The
combiner and its alternatives are reported side-by-side so the recency study (§6.2) shows both
*how much* history helps and *how best* to weight it.

**NGS-block ablation (required for the `ngs` era model).** Next Gen Stats is sparse even
post-2016 — coverage is ~73% among top-36 RBs but ~6% for fringe backs, and a top back lacking
NGS *rushing* is typically a receiving-profile RB. The missingness is therefore signal, exposed
as `has_ngs_rush` / `has_ngs_rec` flags (Stage 4). Rather than assume NGS helps, Stage 8 **runs
the `ngs` era model both with and without the `ngs_efficiency` block** and reports the delta on
**top-k precision** (Precision@12/24/36 — the tiers where NGS coverage is high), plus Spearman/
NDCG. Decision rule: keep the block if it improves top-k out-of-fold; otherwise the `ngs` era
collapses to the `snaps` schema. Handling within each model type: tree era models receive raw
NaN (native split handling); linear era models impute (median) + rely on the coverage flags.
This same ablation harness generalizes to any optional feature block.

### 7.4 Availability model (parallel, §4.3)

A **count model** (Poisson / negative-binomial or quantile GBM) predicting N+1 games
played, trained on the same folds/windows. Feeds projected eligibility and injury-risk
reporting. Baseline: prior-season (or multi-year average) games played.

### 7.4 Benchmark

Prior-year **ADP / ECR** ranking, scored with the same ranking metrics. The headline
result is whether our model beats the market benchmark out-of-sample.

---

## 8. Scoring / evaluation statistics

**Regression (on predicted PPG):** MAE, RMSE, R².

**Ranking quality (the real objective):**
- **Spearman ρ** and **Kendall τ** between predicted and actual positional rank.
- **NDCG@k** (top-of-board emphasis).
- **Precision@k / hit-rate** for tiers: top-12 (RB1), top-24 (RB2), top-36.
- **Tier accuracy / confusion** across fantasy tiers.

**Availability model (§4.3):** games-played **MAE / RMSE**; **AUC & PR-AUC** for the
"clears the games cutoff" classification.

**Reporting:** every model is scored against baselines and the ADP/ECR benchmark, per
test season and aggregated, for each of the 10/20/30-year windows and across the
eligibility-cutoff sensitivity grid. Results are written to `reports/results/` as tidy
tables keyed by (sport, position, model, window, cutoff, fold) for easy comparison.

---

## 9. Generalization to other positions & sports

The repo separates **shared core** from **per-(sport, position) configuration**:

- **Shared library** (`src/position_predictor/`): fetch, dataset build, CV, metrics,
  reporting — sport/position-agnostic.
- **Per-experiment config** (`config/*.yaml`): scoring formula, candidate eligibility
  cutoffs, feature set, history windows, models. Adding QB/WR/TE = new config + position
  feature module; adding a new sport = new data adapter + configs. Each position is
  modeled **separately**; each sport is an independent process.

**WR added (2026-06-17).** Generalising RB → WR confirmed the architecture: build / eras / CV /
metrics / report were already config-driven by `experiment.position`; the **only** code needing
parameterizing was the `offseason` block (now takes `position` + `features.offseason_workload_col`,
position-agnostic column names). A new `config/football_wr.yaml` (workload = targets, WR snap grid)
+ a re-derived eligibility cutoff was all else required. **TE is skipped** (too few
fantasy-relevant TEs/season for a stable per-season ranking). QB is next (needs passing features +
likely an `ngs_passing` pull).

---

## 10. Reproducibility checklist

- [ ] `uv` + `pyproject.toml` + `uv.lock` pin every dependency.
- [ ] `.python-version` pins the interpreter.
- [ ] Config-driven experiments (`config/*.yaml`); no magic numbers in code.
- [ ] Fixed random seeds; deterministic pipeline scripts.
- [ ] Raw data cached + manifested (source, seasons, date, hash).
- [ ] Notebooks import library code only (no divergent logic).
- [ ] Results saved with the config/hash that produced them.
- [ ] **`PROMPT_LOG.md`** records every prompt + response notes (decision provenance).
- [ ] `Makefile` runs the full pipeline from a clean checkout.

---

## 11. Deliverables

1. `data/` fetch pipeline (cached, manifested nflverse pulls).
2. EDA notebooks (coverage, distributions, **target stability/auto-correlation**, age curves).
3. Eligibility-cutoff analysis notebook + derived rule + sensitivity grid.
4. Feature engineering + feature-selection analysis.
5. Modeling + walk-forward results across 10/20/30-yr windows and model families.
6. Results report (metrics tables + figures) and README write-up of findings.

---

## 12. Decisions & parking lot

Resolved (2026-06-15):
- **Test block `k` = last 5 seasons** for the recency study (fixed across the 10/20/30-yr windows).
- **PFR-scraped advanced stats:** planned as a **later enrichment block**, not in v1.
- **Rookie model: dropped** — not pursued, not a future track. Project is returning-players-only.
- **Era-limited features → modeled as separate eras** (§6.3); **NGS kept** with coverage flags +
  a Stage-8 ablation (§7.3) rather than dropped or used to restrict the population.

Resolved (2026-06-16):
- **Eligibility games cutoff `g* = 4`** (Stage 5): the minimum games where split-half PPG
  reliability ≥ 0.70 (0.738 at G=4, stabilising ~0.80 by G=6–8); retains 98.5% of fantasy
  production. Snap-share reported as a secondary/sensitivity dimension; the primary rule is
  games-based for v1. Full grid + curve in `reports/results/eligibility_football_rb.*`.

Resolved (2026-06-16, Stage 8):
- **Headline (RB v1, test seasons 2020–2024, g\* = 4):** the era ensemble beats the must-beat
  baselines — best Spearman ≈ **0.741** (lightgbm/xgboost/ridge, 20–30-yr window) vs **0.720**
  for the best baseline (`linear`) and **0.691** for persistence; MAE ≈ 2.69 PPG vs 3.14 for
  persistence. Full tables in `reports/results/experiment_football_rb_*`.
- **Recency-bias read (§6.2):** 20 yr ≈ 30 yr (data-capped at 1999) and only marginally > 10 yr;
  the `val_weighted` combiner lands near-equal era weights — older-era relationships still predict
  modern RB outcomes about as well, consistent with the strong YoY PPG persistence (EDA, Stage 6).
- **NGS-block ablation (§7.3):** with-NGS does **not** beat without-NGS on Precision@12 out-of-fold
  (≈ equal; small Spearman edge), so per the decision rule the `ngs` era **collapses to the
  `snaps` schema** for v1. Coverage flags are retained. Revisit if receiving-back NGS coverage
  improves or with a top-tier-only model.
- **Availability model (§7.4):** the Poisson games model edges the prior-games baseline on
  clears-cutoff AUC (≈0.79 vs 0.78) but not on games MAE (≈4.6 vs 4.3) — prior games is a strong
  floor; flagged for tuning/wear-feature work in a later pass.

Resolved (2026-06-16, Stage 9):
- **Market benchmark acquired:** preseason FantasyPros ECR (DynastyProcess archive), latest
  preseason redraft-overall RB scrape per season, mapped to `gsis_id` via the nflverse crosswalk
  → `data/external/market_football_rb.parquet` (committed). Coverage of the eligible universe
  rises 72% (2020) → 99% (2024).
- **Do we beat the market? Not on overall rank — yet.** On the identical rows the market ranks,
  market Spearman ≈ **0.732** vs our best model ≈ **0.69** (all six candidates 0.686–0.690). The
  earlier 0.74 was on the *full* eligible universe (inflated by easy-to-rank fringe backs the
  market ignores); the fair head-to-head favours the crowd. **But** our model **matches/edges the
  market on Precision@12** (ridge 0.63 vs market 0.60) — the draftable top tier — using **no**
  market information. Reported in `reports/REPORT_football_rb.md` + `experiment_*_benchmark*`.
- **Eligibility-cutoff robustness:** the headline Spearman is stable from g*=4 (0.737) down to
  g=6 (0.731) and degrades only at g≥8 — the conclusion does not hinge on g*.
- **Reporting (`make report`)** built: `eval/report.py` renders the Markdown report + figures
  from the result tables.

Resolved (2026-06-16, offseason features):
- **Quantified offseason activity (Sept-1 cutoff) closes ~40% of the ECR gap.** Added the
  `offseason` block (§5) — changed-team, rookie-RB draft capital, and backfield competition for
  season N+1, all known by Sept 1. On the market head-to-head it lifted model Spearman **0.690 →
  0.706** (market 0.732; gap 0.042 → 0.026) and Precision@12 **0.633 → 0.650** (now well clear of
  market 0.60); full-universe Spearman 0.74 → **0.752**. The **top-weighted** score barely moved
  (weighted τ 0.662 → 0.663) — offseason signals re-rank the *middle* of the board (committee
  backs gaining/losing a job), not the elite. Most useful features: `backfield_prior_touches_next`
  and `rookie_rb_capital_next` (the continuous competition/capital signals, not the binary flags).
- **No blending, by design:** ECR stays a benchmark to compare against, never a feature.

Resolved (2026-06-17, WR v1):
- **WR model built** (`config/football_wr.yaml`). Eligibility cutoff re-derived: **g\* = 7 games**
  (vs RB's 4) — WR PPG split-half reliability never clears 0.70 (peaks 0.694 @ k=7), i.e. WR
  scoring is **noisier** (boom/bust, TD/big-play dependent) so it needs more games to stabilise
  and WR ranking is inherently harder.
- **Headline (test 2020–2024, g\*=7):** best model `ridge` Spearman **0.752** full-universe (beats
  linear baseline 0.732). Vs market on its ranked rows: market 0.767 vs model 0.748 (Spearman),
  market 0.733 vs 0.713 (weighted τ) — market leads overall rank, as with RB. **But the tree
  models beat the market on Precision@12 (0.583 vs 0.567)** — the draftable top tier.
- **NGS earns its place for WR** (ablation `keep=True`: P@12 0.53 vs 0.50) — unlike RB, where it
  collapsed. Receiving separation / air-yards over-expected is real WR signal.
- **pbp is out of scope** (user decision) — not pursuing red-zone/route features.

Still open:
- **QB** next (passing features + `ngs_passing`); **TE skipped** (thin position).
- Closing the remaining top-weighted gap vs market: more offseason signal (veteran FA
  competition, vacated targets), tuning.
