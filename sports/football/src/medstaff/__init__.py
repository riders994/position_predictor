"""Grading NFL team availability systems from injury-report and roster-transaction data.

**What this grades, stated up front as a non-goal rather than a caveat.** The residual this
project measures is *not* identified as "the medical staff". It bundles the athletic training
staff, strength and conditioning, the sports-science group, the head coach's practice-intensity
choices, the general manager's taste for durable players, and the scheme. What is measurable
here is a **team availability system**; the training room is one input to it. No result in this
package should be described as grading a training staff.

Three components, in ascending order of how attributable they are:

``incidence``   do players get hurt — mostly conditioning, scheme, surface and luck
``duration``   how long they miss vs expected for that body part, severity, position and age
``recurrence``  whether the same body part returns after a return to play — the most
                attributable of the three, and the least contaminated by luck

Two things distinguish this from a leaderboard. Every component is **observed minus expected**
against a league-wide model fit with team identity held out, so a team can never explain away
its own residual. And the **cross-position-group signature** analysis asks whether a team's
excess in one body part travels across position groups that share nothing except the building —
which is the closest this data comes to evidence of a common cause.

See ``docs/MEDSTAFF_PLAN.md``. Conventions follow the sibling ``qb_breakout`` project; raw
caches are shared with ``position_predictor``.
"""

__all__: list[str] = []
