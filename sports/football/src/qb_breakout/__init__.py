"""Late-breakout QB project — identify future NFL QB breakouts from pre-NFL evidence.

Sibling project to ``position_predictor`` inside ``sports/football``. Where that project
predicts next-season fantasy rank from **NFL** history for returning players, this one asks a
different question: given only what was knowable **before a QB played an NFL snap** (high-school
recruiting profile and college production), can we identify the QBs who will eventually break out
— especially the ones who break out *late*, after the league has written them off?

Layers:

- ``labels``   — NFL-side ground truth: per-season QB fantasy ranks, career breakout events, and
  the ``late`` / ``relocated`` flags that define the cohort of interest.
- ``data``     — pre-NFL evidence: ESPN recruiting (high school) and college production.
- ``eda``      — the descriptive analysis of who broke out late and what they had in common.

The NFL layer exists to *build the label and describe the archetypes*. Final models are restricted
to pre-NFL features by design (see ``docs/QB_BREAKOUT_PLAN.md`` §1).
"""
