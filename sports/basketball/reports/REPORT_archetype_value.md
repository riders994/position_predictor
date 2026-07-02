# Phase 2 — Archetype Value Guide (draft intel)

**Which archetypes convert roster *value* into 9-cat wins — and which are traps.** Built by regressing
simulated `cat_win_rate` on each team's **value-weighted archetype exposure** (`Σ max(value,0)·membership`
over the roster — magnitude kept, *not* normalized shares). Corpus: 8640 simulated team-seasons
(60 leagues × 12 seasons). Standardized Ridge coefficient =
value→wins efficiency; **draft coef** uses draft-time (prior-season) value and is the actionable column,
**actual** is the post-hoc ceiling; `sign%` = season-bootstrap sign stability; `top-bot Δ` = extra raw
exposure of top- vs bottom-quartile teams.

## The guide

| archetype | draft coef | sign% | actual coef | top−bot Δexp | label |
| --- | --- | --- | --- | --- | --- |
| High-Usage Engine | +0.0204 | 100% | +0.035 | +2.33 | PRIORITIZE |
| Wing Shot-Creator | +0.0161 | 100% | +0.033 | +0.90 | PRIORITIZE |
| Rim-Protecting Big | +0.0115 | 100% | +0.031 | -0.10 | solid |
| Rebounding Forward | +0.0079 | 100% | +0.019 | +0.13 | solid |
| Stretch Forward | +0.0078 | 100% | +0.019 | +0.36 | solid |
| Lead Playmaker | +0.0070 | 99% | +0.025 | +0.19 | solid |
| Movement Shooter | +0.0069 | 100% | +0.018 | +0.26 | solid |
| Perimeter Stopper | +0.0067 | 100% | +0.014 | +0.34 | solid |
| Rim-Running Center | +0.0056 | 97% | +0.023 | -0.45 | solid |
| Foul-Drawing Iso Scorer | +0.0052 | 100% | +0.014 | +0.01 | solid |
| Non-Scoring Playmaker | +0.0006 | 71% | +0.009 | -0.02 | solid |
| Lead Scoring Guard | -0.0011 | 54% | +0.023 | -2.45 | trap |
| Low-Usage Wing | -0.0045 | 100% | +0.001 | -0.23 | avoid |

- **PRIORITIZE — High-Usage Engine, Wing Shot-Creator:** stably positive draft-time value→wins; winning rosters
  concentrate the most value here.
- **TRAP / avoid — Lead Scoring Guard, Low-Usage Wing:** either a stably negative draft-time weight, or (the
  classic *trap*) an archetype that pays off on *realized* value but whose draft-time weight is a wash —
  it looks valuable (raw scoring) yet its 9-cat category profile doesn't convert (e.g., volume scorers
  who bleed TOV / FG%).

## Why this is *intel*, not a draft engine

Round-by-round archetype **selection weights** were built and evaluated in-sim: a picker scoring players
by `projected_value × (weight · archetype)` with diminishing returns **lost to the field** (lift −0.02),
and even break-even static weighting only tied it. Balancing across archetypes sacrifices the
value/coverage **concentration** (punt builds) that actually wins 9-cat. So the draft **engine** stays
the category-coverage optimizer (`make optimize`); these coefficients are a companion read for *which
archetypes to spend on*, not a mechanical picker. The draft-time coefficients are also small in absolute
terms — the Phase-3 projection ceiling caps how much any draft-time signal can deliver.

---
*Method note:* value-weighted **exposure** (magnitude) reaches oof R²≈0.30 on actual value vs ≈0 for
equal-weighted shares — confirming archetype composition carries success signal only when weighted by
player value *and* left un-normalized. See REPORT_phase2_simulation.md for the representation shootout.
