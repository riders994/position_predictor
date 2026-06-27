# Basketball (NBA) — fantasy archetype project

A standalone project under `sports/basketball/` (own code, data, pipeline, reports). Goal: build a
winning **9-category** fantasy roster by thinking in play-style **archetypes** (the "11–14 real
positions" idea) rather than the traditional five positions.

**Status: planning.** Design of record in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md); full
decision trail in [`docs/PROMPT_LOG.md`](docs/PROMPT_LOG.md) (per-sport) and the repo-wide
[`../../docs/PROMPT_LOG.md`](../../docs/PROMPT_LOG.md).

Three phases:
1. **Archetype discovery** — unsupervised soft clustering of player play-style into ~11–14
   archetypes (modern game, 2013+).
2. **Archetype composition → fantasy success** — which archetype groupings build a winning 9-cat
   roster (labels from two public Fantrax sample leagues).
3. **Archetype predictor** — from a player's past data, predict next season's archetype(s)
   (leak-safe, returning players only).
