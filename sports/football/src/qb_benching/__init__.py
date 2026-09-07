"""Which quarterbacks who open a season as the starter lose the job to a *decision*.

The question is deliberately narrow. A team's Week-1 starter stops starting for three quite
different reasons — he gets hurt, he gets benched, or he is traded or released — and only the
second is a judgement about his play. Lumping them together produces a model of NFL injury
attrition, which this repo has already measured twice and found unpredictable (PROMPT_LOG
entries 096-097). So the label layer separates all three and the model targets benching alone,
carrying the injury arm alongside as the declared comparison.

The layers:

``labels``
    Who started each team-game, who opened the season, and what happened in every week the
    opener did not start. Produces the panel and the one-row-per-QB-season cohort.

What this project does **not** do:

* It does not feed the fantasy ranking models. Nothing here becomes a feature of a PPG
  projection, in the same way ``medstaff`` and ``qb_breakout`` stay out of that pipeline.
* It does not blend with the market. Preseason ECR/ADP is a benchmark — the question of whether
  the market already prices benching risk — and never an input.
* It is not an injury model. The injured arm exists to be compared against, not to be believed.
"""

__all__: list[str] = []
