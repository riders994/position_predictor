"""Body-part normalisation: 170-odd free-text injury strings -> a small stable taxonomy.

The injury report's body-part field is typed by 32 different clubs with no controlled
vocabulary. Measured on 2021-2025 there are **172 distinct strings**, of which the top 45 cover
98.3% of rows — so a normaliser plus a keyword table works, but a plain lookup dict does not.
The mess, all of it real:

- laterality prefixes — ``right Shoulder``, ``left Knee``
- case variants — ``shoulder`` alongside ``Shoulder``
- singular/plural — ``Rib`` and ``Ribs``
- multi-part — ``Foot/Wrist/Hip``
- bracketed clauses — ``Ankle [Not Injury Related - Personal, Thursday Only]``
- whole free-text sentences — ``Andrew has cleared concussion protocol and does not have a
  game status.``
- non-injuries wearing the same field — ``Jury Duty``, ``Returning from Suspension``,
  ``Not injury related - resting player``

Two rules do most of the work:

**Non-injury is checked before anything else**, on the raw string. A row labelled "Not Injury
Related" is excluded even when it also names a body part, because the absence that week was not
medical.

**Among body-part matches, the earliest match in the string wins, longest match breaking ties.**
Earliest-wins encodes the convention that the primary injury is listed first, so ``Foot/Wrist/Hip``
is a foot injury. Longest-wins is what keeps ``hip flexor`` in the soft-tissue group instead of
collapsing to ``hip`` — and it does that structurally, rather than depending on the order rules
happen to be written in.

``FOCAL_GROUPS`` are the five with a plausible *common-cause* mechanism, and are the subject of
the cross-position-group signature analysis (plan §4.8). Shoulder is deliberately not focal:
contact-incidental injuries are individually driven, so a team-wide signature in one would have
no mechanism to point at.

**Hamstring is its own group, and is not focal — for two different reasons.** It is separated from
``soft_tissue_lower`` because it is 56% of that group on its own (2,210 of 3,980 report rows) and
because hamstring reinjury is the canonical rehab-quality marker in the sports-science literature,
which makes it the single most interesting category for the *recurrence* component. It is kept out
of ``FOCAL_GROUPS`` because the signature analysis' five parts were **preregistered** — adding a
sixth after seeing results is exactly the fishing that preregistering guards against. Hamstring
concordance can be reported, labelled post-hoc.
"""

from __future__ import annotations

import re

# The five parts whose excess could plausibly share a cause across position groups —
# surface, practice contact policy, S&C programme, medical protocol.
FOCAL_GROUPS = ("knee", "ankle", "back", "hip", "concussion")

GROUPS = (
    "concussion", "knee", "ankle", "foot", "lower_leg", "hamstring", "soft_tissue_lower",
    "hip",
    "back", "neck", "shoulder", "arm_hand", "torso", "head_face", "illness",
    "non_injury", "other",
)

# Checked first, against the raw lowered string, before any body part is considered.
NON_INJURY_PATTERNS = (
    r"not\s+injury\s+related", r"non[-\s]?injury", r"\bpersonal\b", r"\brest(ing|ed)?\b",
    r"load\s+management", r"\btravel\b", r"\bsuspension\b", r"suspended",
    r"jury\s+duty", r"\binactive\b", r"coach'?s?\s+decision", r"\bbereavement\b",
    r"birth\s+of", r"\bdiscipline\b", r"\bholdout\b", r"\bvisa\b",
)

# Severity words the clubs sometimes volunteer. Sparse by design — most rows are a bare body
# part — so this supplements, and never replaces, the practice-participation severity proxy.
SEVERE_PATTERNS = (
    r"\bacl\b", r"achilles", r"\btorn\b", r"\btear\b", r"ruptur", r"fractur",
    r"\bbroken\b", r"surger", r"dislocat", r"amputat", r"\bpcl\b",
)

# (group, pattern) in rough specificity order; ties are broken by match position then length,
# so this ordering is a readability aid rather than load-bearing logic.
_RULES: tuple[tuple[str, str], ...] = (
    ("concussion", r"concussion"),
    ("concussion", r"\bhead\b"),
    ("soft_tissue_lower", r"hip\s*flexor"),
    ("hamstring", r"hamstring"),
    ("soft_tissue_lower", r"quadricep|\bquad\b"),
    ("soft_tissue_lower", r"\bgroin\b"),
    ("soft_tissue_lower", r"\bthigh\b"),
    ("soft_tissue_lower", r"\bglute"),
    ("soft_tissue_lower", r"adductor"),
    ("knee", r"\bknee\b"),
    ("knee", r"\bacl\b|\bmcl\b|\bpcl\b|\blcl\b"),
    ("knee", r"meniscus|patella"),
    ("ankle", r"\bankle"),
    ("foot", r"\bfoot\b|midfoot|lisfranc"),
    ("foot", r"\btoe\b|turf\s*toe"),
    ("foot", r"\bheel\b|plantar"),
    ("lower_leg", r"\bcalf\b|\bcalves\b"),
    ("lower_leg", r"\bshin\b"),
    ("lower_leg", r"achilles"),
    ("lower_leg", r"fibula|tibia|lower\s+leg"),
    ("hip", r"\bhip\b"),
    ("back", r"\bback\b|lumbar|\bspine\b|sacrum|\bsi\s+joint\b|\bdisc\b"),
    ("neck", r"\bneck\b|stinger|\bburner\b|cervical|\btrap(ezius)?\b"),
    ("shoulder", r"shoulder|clavicle|collarbone|labrum|rotator|\bac\s+joint\b|scapula"),
    ("arm_hand", r"\belbow\b|\bwrist\b|\bhand\b|\bthumb\b|\bfinger|knuckle"),
    ("arm_hand", r"forearm|\bbicep|\btricep|\barm\b"),
    ("torso", r"\brib\b|\bribs\b|\bchest\b|pectoral|\bpec\b|sternum"),
    ("torso", r"abdom|\boblique\b|\bcore\b|hernia"),
    ("head_face", r"\beye\b|\bjaw\b|\bnose\b|\bear\b|\blips?\b|\bteeth\b|\btooth\b|\bface\b"),
    ("illness", r"illness|\bill\b|\bsick\b|\bflu\b|appendi|\bvirus\b|covid|\bcramp"),
)

_COMPILED = tuple((group, re.compile(pat, re.I)) for group, pat in _RULES)
_NON_INJURY = tuple(re.compile(p, re.I) for p in NON_INJURY_PATTERNS)
_SEVERE = tuple(re.compile(p, re.I) for p in SEVERE_PATTERNS)

_LATERALITY = re.compile(r"\b(left|right|l|r|bi?lateral)\b\s*", re.I)
_BRACKETED = re.compile(r"[\[\(][^\]\)]*[\]\)]")


def normalize(raw: str | None) -> str:
    """Lowercase, drop bracketed clauses and laterality, collapse whitespace."""
    if raw is None:
        return ""
    text = str(raw)
    if not text.strip():
        return ""
    text = _BRACKETED.sub(" ", text)
    text = _LATERALITY.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def is_non_injury(raw: str | None) -> bool:
    """True when the row records an absence that is not medical.

    Checked against the *raw* string, before bracket-stripping, so
    ``Ankle [Not Injury Related - Personal, Thursday Only]`` is correctly excluded rather than
    counted as an ankle injury.
    """
    if raw is None:
        return False
    text = str(raw)
    return any(p.search(text) for p in _NON_INJURY)


def body_part_group(raw: str | None) -> str | None:
    """Map a raw body-part string to a taxonomy group.

    Returns ``None`` for null/blank input, ``"non_injury"`` for non-medical absences, and
    ``"other"`` for a real string that matches no rule (audited by :func:`unmapped_report`).
    """
    if raw is None or not str(raw).strip():
        return None
    if is_non_injury(raw):
        return "non_injury"
    text = normalize(raw)
    if not text:
        return None

    best: tuple[int, int, int] | None = None
    winner = "other"
    for order, (group, pattern) in enumerate(_COMPILED):
        match = pattern.search(text)
        if match is None:
            continue
        # earliest match wins (the primary injury is listed first); longer match breaks a tie
        # so "hip flexor" beats "hip"; rule order is only the final tiebreak.
        key = (match.start(), -(match.end() - match.start()), order)
        if best is None or key < best:
            best, winner = key, group
    return winner


def is_severe(raw: str | None) -> bool:
    """True when the string volunteers a severity word (ACL, torn, fracture, ...)."""
    if raw is None:
        return False
    return any(p.search(str(raw)) for p in _SEVERE)


def is_focal(group: str | None) -> bool:
    """True for the five common-cause body parts used in the signature analysis."""
    return group in FOCAL_GROUPS


def coalesce_body_part(frame, *, out: str = "body_part_raw"):
    """Add the coalesced raw body part: report primary, else practice primary, else secondary.

    ``report_*`` is null for roughly half of all rows — those are practice-report-only weeks —
    and the null rate is not stable over time (it jumps at 2016 when the league dropped the
    "Probable" designation). The practice fields do not have that break, which is why they are
    part of the coalesce rather than a fallback of last resort.
    """
    import polars as pl

    candidates = [
        c for c in ("report_primary_injury", "practice_primary_injury",
                    "report_secondary_injury", "practice_secondary_injury")
        if c in frame.columns
    ]
    if not candidates:
        raise KeyError("no injury body-part columns present")
    return frame.with_columns(pl.coalesce([pl.col(c) for c in candidates]).alias(out))


def classify(frame, *, source: str = "body_part_raw"):
    """Attach ``body_part_group``, ``is_severe`` and ``body_part_known`` to a polars frame."""
    import polars as pl

    return frame.with_columns(
        pl.col(source).map_elements(body_part_group, return_dtype=pl.String)
        .alias("body_part_group"),
        pl.col(source).map_elements(is_severe, return_dtype=pl.Boolean).alias("is_severe"),
        pl.col(source).is_not_null().alias("body_part_known"),
    )


def unmapped_report(frame, *, source: str = "body_part_raw", min_share: float = 0.0):
    """Raw strings that fell through to ``other``, by frequency — the taxonomy audit.

    Any string above ~0.1% of rows appearing here means the taxonomy has a gap. This table goes
    into the stage-1 report so a gap is visible rather than silently pooled.
    """
    import polars as pl

    total = frame.height
    fell_through = (
        frame.filter(
            pl.col(source).is_not_null()
            & (pl.col(source).map_elements(body_part_group, return_dtype=pl.String) == "other")
        )
        .group_by(source)
        .len()
        .rename({source: "raw", "len": "n"})
        .with_columns((pl.col("n") / total).alias("share"))
        .sort("n", descending=True)
    )
    return fell_through.filter(pl.col("share") >= min_share) if min_share else fell_through


__all__ = [
    "FOCAL_GROUPS", "GROUPS", "NON_INJURY_PATTERNS", "SEVERE_PATTERNS",
    "body_part_group", "classify", "coalesce_body_part", "is_focal", "is_non_injury",
    "is_severe", "normalize", "unmapped_report",
]
