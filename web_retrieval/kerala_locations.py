"""
Shared list of known Kerala place names, used both by
web_retrieval.course_merger (to recognize a trailing location suffix
on an institute name) and web_retrieval.course_extractor (to validate
that an extracted "location" field value is an actual place, not
arbitrary text from a marketing paragraph). Kept as its own tiny
module so both can import the exact same list instead of drifting.
"""

import re

KERALA_LOCATIONS = (
    "calicut", "kozhikode", "kochi", "cochin", "ernakulam",
    "trivandrum", "thiruvananthapuram", "kollam", "kottayam",
    "thrissur", "trichur", "palakkad", "palghat", "kannur",
    "cannanore", "alappuzha", "alleppey", "malappuram", "idukki",
    "wayanad", "kasaragod", "pathanamthitta",
)


# ============================================================
# ALIAS NORMALIZATION (Step 5B)
# ============================================================
# Several Kerala cities have two common English spellings/names that
# refer to the SAME place -- "Kochi"/"Cochin", "Kozhikode"/"Calicut",
# "Thiruvananthapuram"/"Trivandrum", "Thrissur"/"Trichur",
# "Kannur"/"Cannanore", "Alappuzha"/"Alleppey", "Palakkad"/"Palghat" --
# a course page and a user's query can each independently pick either
# spelling, so a plain string-equality/substring check between an
# extracted "location" field and a query's requested_location was
# missing an obvious match ("Cochin" on the page, "Kochi" in the
# query). "Ernakulam" is deliberately NOT folded into "Kochi" despite
# colloquial overlap -- it's also a distinct district name, and the
# two aren't always used interchangeably, so treating them as
# identical would be a guess this module isn't in a position to make.
#
# Every alias maps to a single canonical lowercase form (arbitrarily,
# whichever spelling KERALA_LOCATIONS lists first in each pair) --
# used ONLY for matching/ranking (see web_retrieval.query_intent and
# web_retrieval.course_ranker). Extracted display text is never
# rewritten to the canonical form; a page that said "Cochin" still
# shows "Cochin" to the user.
KERALA_LOCATION_ALIASES = {
    "calicut": "kozhikode",
    "kozhikode": "kozhikode",
    "kochi": "kochi",
    "cochin": "kochi",
    "ernakulam": "ernakulam",
    "trivandrum": "thiruvananthapuram",
    "thiruvananthapuram": "thiruvananthapuram",
    "kollam": "kollam",
    "kottayam": "kottayam",
    "thrissur": "thrissur",
    "trichur": "thrissur",
    "palakkad": "palakkad",
    "palghat": "palakkad",
    "kannur": "kannur",
    "cannanore": "kannur",
    "alappuzha": "alappuzha",
    "alleppey": "alappuzha",
    "malappuram": "malappuram",
    "idukki": "idukki",
    "wayanad": "wayanad",
    "kasaragod": "kasaragod",
    "pathanamthitta": "pathanamthitta",
}

_LOCATION_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in KERALA_LOCATION_ALIASES) + r")\b",
    re.IGNORECASE
)


def canonical_location(raw_value):
    """
    Returns the canonical Kerala place name found in RAW_VALUE (a
    whole-word match against KERALA_LOCATION_ALIASES), or None if no
    known Kerala location appears in it at all. "Kochi" and "Cochin"
    both return "kochi"; unrelated/empty text returns None -- never
    guessed from partial or unrelated text.
    """

    if not raw_value:
        return None

    match = _LOCATION_ALIAS_PATTERN.search(raw_value)

    if not match:
        return None

    return KERALA_LOCATION_ALIASES[match.group(1).lower()]
