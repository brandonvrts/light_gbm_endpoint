"""Canonical marketing cohort vocabulary.

Faithful Python port of the marketing app's CohortVocab.cs (itself a byte-exact port
of signal-desk's api/ontology-lib.mjs: sizeBandOf / bandFromEmployeeRange / verticalOf /
cohortKey / cohortKeysFor). The cohort key is the LITERAL join key between a company,
a space, rea.DealPattern and mkt.CohortCatalog — any divergence makes joins silently
miss. Parity is pinned by a golden test against live rea.DealPattern.CohortKey values
(mirroring the marketing repo's own contract test).

NOTE: these size bands (25/100/500/2000) are NOT the space-demand endpoint's display
`_emp_band` (50/250/1000) — cohort joins must use THESE.

rea.DealPattern also contains a SEPARATE `sizesqft=<band>` key family (BackoffLevel 4,
signal-desk build-deal-patterns.mjs "A4") keyed on the DEAL's square-footage band, which
overloads the SizeBand column. That family is deal/space-grain, not a company cohort
dimension, and is intentionally not modeled here.
"""
from __future__ import annotations
import math
import re

SIZE_BANDS = ["micro", "small", "mid", "large", "enterprise"]

VERTICALS = [
    "finance-insurance", "healthcare", "technology", "manufacturing",
    "professional-services", "retail-hospitality", "logistics",
    "real-estate", "other",
]

# First match wins — order is significant (e.g. "accounting" resolves to
# finance-insurance, not professional-services), mirroring CohortVocab.VerticalOf.
_VERTICAL_RULES = [
    (re.compile(r"insur|financ|bank|wealth|invest|capital|account"), "finance-insurance"),
    (re.compile(r"health|medic|clinic|hospital|pharma|dental|care"), "healthcare"),
    (re.compile(r"software|saas|tech|data|cyber|it services|information"), "technology"),
    (re.compile(r"manufactur|industrial|fabricat|machin"), "manufacturing"),
    (re.compile(r"legal|law|consult|engineer|architect|account|professional"), "professional-services"),
    (re.compile(r"retail|restaurant|hospitality|hotel|food"), "retail-hospitality"),
    (re.compile(r"logist|transport|warehouse|distribution|freight"), "logistics"),
    (re.compile(r"real estate|property|construct|develop"), "real-estate"),
]


def size_band_of(employees) -> str:
    """Size band from an exact employee count (CohortVocab.SizeBandOf)."""
    if employees is None:
        return "unknown"
    try:
        n = float(employees)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(n) or n <= 0:
        return "unknown"
    if n < 25:
        return "micro"
    if n < 100:
        return "small"
    if n < 500:
        return "mid"
    if n < 2000:
        return "large"
    return "enterprise"


def band_from_employee_range(range_str: str | None) -> str:
    """Size band from a ZI EmployeeRange string, banded by the LOWER bound
    (CohortVocab.BandFromEmployeeRange). NOTE the thresholds intentionally differ
    from size_band_of — this mirrors the upstream split exactly."""
    s = (range_str or "").strip()
    if not s:
        return "unknown"
    if re.search(r"over\s*10", s, re.IGNORECASE):
        return "enterprise"
    m = re.search(r"[\d,]+", s)
    if not m:
        return "unknown"
    try:
        first = int(m.group(0).replace(",", ""))
    except ValueError:
        return "unknown"
    if first <= 0:
        return "unknown"
    if first < 20:
        return "micro"
    if first < 100:
        return "small"
    if first < 500:
        return "mid"
    if first < 5000:
        return "large"
    return "enterprise"


def vertical_of(industry: str | None) -> str | None:
    """Coarse vertical from an industry string (CohortVocab.VerticalOf).
    Returns None for empty input (no vertical dimension on the key)."""
    s = (industry or "").lower()
    if not s:
        return None
    for rx, vertical in _VERTICAL_RULES:
        if rx.search(s):
            return vertical
    return "other"


def cohort_key(size_band: str | None, vertical: str | None) -> str:
    """Build the canonical key (CohortVocab.CohortKey). Empty dims -> 'global'."""
    parts = []
    if size_band:
        parts.append(f"size={size_band}")
    if vertical:
        parts.append(f"vertical={vertical}")
    return "|".join(parts) if parts else "global"


def cohort_keys_for(size_band: str | None, vertical: str | None) -> list[tuple[int, str]]:
    """Ordered most-specific -> least-specific (level, key) backoff list
    (CohortVocab.CohortKeysFor). 'unknown' dims are dropped before keying.
    Levels: 0=size+vertical, 1=size, 2=vertical, 3=global."""
    sb = size_band if size_band and size_band != "unknown" else None
    v = vertical if vertical and vertical != "unknown" else None
    keys: list[tuple[int, str]] = []
    if sb and v:
        keys.append((0, cohort_key(sb, v)))
    if sb:
        keys.append((1, cohort_key(sb, None)))
    if v:
        keys.append((2, cohort_key(None, v)))
    keys.append((3, "global"))
    return keys
