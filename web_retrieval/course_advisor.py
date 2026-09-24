"""
Deterministic Course Advisor -- Advisor Step 2.

Turns an advisory-style query ("I'm a beginner and I want to learn
Data Science in Kochi, budget below Rs 50,000, online classes") into a
structured preference dict and re-ranks the already-deduplicated
structured course_records (see web_retrieval.course_merger and
web_retrieval.course_ranker) against those preferences -- entirely
deterministically, no LLM call.

This module never duplicates the topic/level/location matching logic
that already lives in web_retrieval.course_ranker: preferences here
are a strict SUPERSET of web_retrieval.query_intent.parse_query_intent's
own dict, and ranking here calls course_ranker.build_rank_key directly
and only APPENDS three new tiers (budget/mode/placement) -- see
build_advisor_rank_key. There is exactly one ranking mechanism in this
project; this module extends it, it does not replace it.

Every dimension follows the same tri-state pattern already established
by course_ranker's level/location ranking: MATCH < UNSPECIFIED <
CONFLICT. Missing information is never treated as a conflict, and the
Advisor never activates for an ordinary factual/search query -- see
is_advisor_query.
"""

import re

from web_retrieval.course_extractor import NOT_AVAILABLE
from web_retrieval.query_intent import parse_query_intent
from web_retrieval.course_ranker import (
    build_rank_key,
    _topic_rank,
    _level_rank,
    _location_rank,
    _TOPIC_STRONG,
    _TOPIC_COMPOUND,
    _TOPIC_RELATED,
    _TOPIC_UNKNOWN,
    _LEVEL_MATCH,
    _LEVEL_UNSPECIFIED,
    _LOCATION_MATCH,
    _LOCATION_UNSPECIFIED,
)


# ============================================================
# ADVISOR ACTIVATION
# ============================================================
# See the module docstring's "smallest safe architecture" reasoning
# (Advisor Step 1B): Branch A fires on an explicit, unambiguous
# advisory/recommendation phrase alone. Branch B fires on first-person
# preference language ONLY when at least one real preference signal
# was also detected -- this is what stops a vague "I'm looking for
# information about Codeme Hub" (no topic/level/location/budget/mode/
# placement signal at all) from misfiring, while still catching "I'm
# looking for a Data Science course" (a real topic signal is present).

_ADVISOR_PHRASES_A = (
    "recommend", "recommendation", "recommended",
    "suggest", "suggestion",
    "suitable for me",
    "should i choose", "should i pick", "should i select",
    "help me choose", "help me pick",
    "best for me",
    "course for me",
    "advise me", "advice",
)

_ADVISOR_PHRASES_B = (
    "i'm a beginner", "i am a beginner",
    "i prefer",
    "my budget",
    "i want to learn",
    "i want",
    "i need a course", "i need a",
    "i'm looking for", "i am looking for",
    "help me",
)


def _has_real_preference_signal(preferences):

    return bool(
        preferences.get("requested_topics")
        or preferences.get("requested_level")
        or preferences.get("requested_location")
        or preferences.get("budget_max") is not None
        or preferences.get("mode")
        or preferences.get("placement_preference")
    )


def is_advisor_query(query, preferences=None):
    """
    Deterministic activation rule (Advisor Step 1B, corrected):

        Branch A: an explicit advisory/recommendation phrase is
            present -- fires on its own, no other signal required.
        Branch B: first-person/preference language is present AND at
            least one real preference signal (topic/level/location/
            budget/mode/placement) was actually detected -- prevents
            vague first-person phrasing with nothing else to go on
            from activating the Advisor.

    preferences, if not already computed by the caller, is derived via
    parse_advisor_preferences(query) -- callers that already have it
    (e.g. mcp_server/course_search_server.py) should pass it in to
    avoid re-parsing the same query twice.
    """

    if not query or not query.strip():
        return False

    query_lower = query.lower()

    if preferences is None:
        preferences = parse_advisor_preferences(query)

    if any(phrase in query_lower for phrase in _ADVISOR_PHRASES_A):
        return True

    if any(phrase in query_lower for phrase in _ADVISOR_PHRASES_B):
        return _has_real_preference_signal(preferences)

    return False


# ============================================================
# BUDGET PARSING
# ============================================================
# Deliberately REQUIRES a ceiling cue word ("under"/"below"/"less
# than"/"within"/"up to"/"budget of"/"max(imum) of") before the
# amount -- a bare "Rs 50,000" with no cue word is ambiguous (it could
# be an exact quoted price mention, not necessarily what the USER is
# asking for as a ceiling), so it deliberately returns None rather
# than guessing which direction it constrains the search.

_AMOUNT_SUFFIX_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "l": 100_000,
    "lakh": 100_000,
    "lakhs": 100_000,
}

_BUDGET_PATTERN = re.compile(
    r"\b(?:under|below|less\s+than|within|up\s+to|budget\s+of|"
    r"max(?:imum)?\s+of|maximum)\s*"
    r"(?:₹|rs\.?|inr)?\s*"
    r"([\d,]+(?:\.\d+)?)\s*"
    r"(k|thousand|lakhs?|l)?\b",
    re.IGNORECASE
)


def _parse_amount(number_str, suffix):

    try:
        value = float(number_str.replace(",", ""))

    except (TypeError, ValueError):
        return None

    multiplier = _AMOUNT_SUFFIX_MULTIPLIERS.get((suffix or "").lower(), 1)

    return int(value * multiplier)


def _parse_budget_max(query_lower):

    match = _BUDGET_PATTERN.search(query_lower)

    if not match:
        return None

    return _parse_amount(match.group(1), match.group(2))


# The "fees" field (see web_retrieval.course_extractor._extract_fees)
# only ever gets populated when a currency symbol/Rs/INR prefix was
# already found, so re-parsing it here for a numeric value doesn't
# need the same ceiling-cue requirement as the query-side parser above
# -- it's already a validated, narrow field, not arbitrary free text.
_FEE_AMOUNT_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|lakhs?|l)?",
    re.IGNORECASE
)


def _parse_fee_amount(fees_text):

    if not fees_text or fees_text in (NOT_AVAILABLE, "Varies by source"):
        return None

    match = _FEE_AMOUNT_PATTERN.search(fees_text)

    if not match:
        return None

    return _parse_amount(match.group(1), match.group(2))


# ============================================================
# MODE PARSING
# ============================================================
# Normalized, where a direct equivalent already exists, to the SAME
# labels course_extractor._MODE_RULES produces on real records
# ("Online", "Offline", "LMS / Self-paced") so comparison in
# _mode_rank is a simple, explainable check rather than fuzzy text
# matching. "Hybrid" has no single-word equivalent in the existing
# vocabulary (course records normalize a hybrid mention to "Online &
# Offline") -- kept as its own explicit label here, with _mode_rank
# below bridging the two forms when comparing against a record.

_MODE_KEYWORD_RULES = (
    (re.compile(r"\bself[\s-]?paced\b", re.IGNORECASE), "LMS / Self-paced"),
    (re.compile(r"\bhybrid\b", re.IGNORECASE), "Hybrid"),
    (re.compile(r"\bonline\b", re.IGNORECASE), "Online"),
    (
        re.compile(r"\boffline\b|\bclassroom\b|\bin[\s-]?person\b", re.IGNORECASE),
        "Offline",
    ),
)


def _parse_mode(query_lower):

    for pattern, label in _MODE_KEYWORD_RULES:

        if pattern.search(query_lower):
            return label

    return None


# ============================================================
# PLACEMENT PREFERENCE
# ============================================================
# Conservative, whole-phrase matching only -- a bare "job" or "career"
# is NOT enough ("I want a job in Data Science" is a career goal, not
# a request for the COURSE to include placement support). Only fires
# on a phrase that specifically ties "placement"/"career support" to
# the course itself.

_PLACEMENT_PREFERENCE_PHRASES = (
    "placement assistance",
    "placement support",
    "placement available",
    "job placement",
    "career support",
    "career assistance",
    "placement preferred",
    "with placement",
    "placement guarantee",
)


def _has_placement_preference(query_lower):

    return any(phrase in query_lower for phrase in _PLACEMENT_PREFERENCE_PHRASES)


# ============================================================
# PREFERENCE SCHEMA
# ============================================================

def parse_advisor_preferences(query: str) -> dict:
    """
    Returns a strict superset of web_retrieval.query_intent.parse_query_intent's
    dict -- requested_topics/requested_fields/requested_level/
    requested_location/comparison_intent are reused VERBATIM (never
    re-derived), plus three new Advisor-specific fields:
    budget_max (int or None), mode (str or None),
    placement_preference (bool). Every field stays optional; a query
    with nothing detectable returns a safe, all-empty/neutral dict,
    never an error.
    """

    intent = parse_query_intent(query)

    query_lower = query.lower() if query else ""

    return {
        **intent,
        "budget_max": _parse_budget_max(query_lower),
        "mode": _parse_mode(query_lower),
        "placement_preference": _has_placement_preference(query_lower),
    }


# ============================================================
# ADVISOR-SPECIFIC DIMENSION RANKING (tri-state, same pattern as
# course_ranker's level/location: MATCH < UNSPECIFIED < CONFLICT)
# ============================================================

_BUDGET_MATCH = 0
_BUDGET_UNSPECIFIED = 1
_BUDGET_CONFLICT = 2


def _budget_rank(course, budget_max):

    if budget_max is None:
        return _BUDGET_MATCH

    fee_amount = _parse_fee_amount(course.get("fees"))

    if fee_amount is None:
        return _BUDGET_UNSPECIFIED

    return _BUDGET_MATCH if fee_amount <= budget_max else _BUDGET_CONFLICT


_MODE_MATCH = 0
_MODE_UNSPECIFIED = 1
_MODE_CONFLICT = 2

# A course record's learning_mode value of "Online & Offline"
# explicitly offers both delivery types, so it satisfies an Online, an
# Offline, or a Hybrid request equally -- it's real evidence of both,
# not a guess.
_BOTH_MODES_LABEL = "online & offline"


def _mode_rank(course, requested_mode):

    if not requested_mode:
        return _MODE_MATCH

    course_mode = course.get("learning_mode")

    if not course_mode or course_mode == NOT_AVAILABLE:
        return _MODE_UNSPECIFIED

    course_mode_lower = course_mode.strip().lower()
    requested_lower = requested_mode.strip().lower()

    if _BOTH_MODES_LABEL in course_mode_lower:
        return _MODE_MATCH

    if requested_lower == course_mode_lower:
        return _MODE_MATCH

    if requested_lower == "online" and "self-paced" in course_mode_lower:
        return _MODE_MATCH

    return _MODE_CONFLICT


_PLACEMENT_MATCH = 0
_PLACEMENT_UNSPECIFIED = 1
_PLACEMENT_CONFLICT = 2

_PLACEMENT_NEGATION_CUES = (
    "no placement", "not provided", "does not provide", "doesn't provide",
    "without placement", "no career support", "not offer", "doesn't offer",
)

# Advisor Step 2B (Problem 4): a populated placement_information field
# is NOT, by itself, strong enough evidence to claim a MATCH for a
# user who specifically asked about placement support -- confirmed
# against a real live response where a course's placement_information
# text was "You leave with a portfolio that shows you can ship, not
# just train." (a project/portfolio claim, extracted because the
# SOURCE PARAGRAPH's corroboration check in course_extractor.py only
# requires an evidence word to appear SOMEWHERE in that paragraph, not
# specifically in the sentence that ends up here) and the Advisor
# treated that as a placement match. Rather than touching
# course_extractor.py's extraction/corroboration logic (out of scope
# for this fix, and would affect every other caller of that field),
# _placement_rank now requires the EXTRACTED TEXT ITSELF to contain a
# genuinely strong, explicit placement/career-support phrase before
# calling it a match -- a portfolio/project/employability mention
# alone is deliberately NOT enough and now falls through to
# UNSPECIFIED (present, but not clear enough to claim a match),
# never a false MATCH and never a false CONFLICT.
_STRONG_PLACEMENT_EVIDENCE_PHRASES = (
    "placement assistance",
    "placement support",
    "placement services",
    "placement cell",
    "placement training",
    "placement guarantee",
    "job placement",
    "career support",
    "career assistance",
    "recruitment assistance",
    "interview support",
    "job placement support",
    "interview and job placement",
)


def _placement_rank(course, placement_preference):

    if not placement_preference:
        # Neutral for every record -- the user never asked, so this
        # dimension must not influence ranking at all.
        return _PLACEMENT_MATCH

    placement_text = course.get("placement_information")

    if not placement_text or placement_text == NOT_AVAILABLE:
        return _PLACEMENT_UNSPECIFIED

    lowered = placement_text.lower()

    if any(cue in lowered for cue in _PLACEMENT_NEGATION_CUES):
        return _PLACEMENT_CONFLICT

    if any(phrase in lowered for phrase in _STRONG_PLACEMENT_EVIDENCE_PHRASES):
        return _PLACEMENT_MATCH

    # Present, but not a strong enough phrase to confidently call it a
    # match (e.g. a portfolio/project/employability mention) -- never
    # claimed as a match, never penalized as a conflict either.
    return _PLACEMENT_UNSPECIFIED


# ============================================================
# RANKING -- extends course_ranker.build_rank_key, never duplicates it
# ============================================================

def build_advisor_rank_key(course, preferences, query_lower):
    """
    The existing five-dimension base key (institute/topic/field-
    coverage/level/location -- see course_ranker.build_rank_key,
    UNCHANGED behavior) with three more tiers appended: budget, mode,
    placement. One tuple, one deterministic sort -- never two
    competing ranking passes.
    """

    base_key = build_rank_key(course, preferences, query_lower)

    return base_key + (
        _budget_rank(course, preferences.get("budget_max")),
        _mode_rank(course, preferences.get("mode")),
        _placement_rank(course, preferences.get("placement_preference")),
    )


def advisor_rank_courses(course_records, preferences, query: str = ""):
    """
    Reorders (never drops) course_records using the extended 8-tier
    Advisor rank key. When preferences carries no Advisor-specific
    values (budget_max/mode/placement_preference all empty), every
    record ties on the three appended tiers and the result is
    identical to course_ranker.filter_and_rank_courses's own ordering
    for the same base query_intent fields.
    """

    if not course_records:
        return course_records

    preferences = preferences or {}
    query_lower = (query or "").lower()

    return sorted(
        course_records,
        key=lambda course: build_advisor_rank_key(course, preferences, query_lower)
    )


# ============================================================
# EXPLANATION (deterministic, no LLM)
# ============================================================

def _compute_dimension_results(course, preferences, query_lower):

    return {
        "topic": _topic_rank(course, preferences.get("requested_topics") or []),
        "level": _level_rank(course, preferences.get("requested_level")),
        "location": _location_rank(course, preferences.get("requested_location")),
        "budget": _budget_rank(course, preferences.get("budget_max")),
        "mode": _mode_rank(course, preferences.get("mode")),
        "placement": _placement_rank(course, preferences.get("placement_preference")),
    }


def explain_match(course, preferences, dimension_results=None) -> str:
    """
    Turns the SAME dimension results the ranking already computed into
    a short, human-readable explanation -- never a second, independent
    evaluation, and never a claim the source data doesn't support. A
    dimension the user didn't ask about is left out of the explanation
    entirely (nothing to explain); one the user asked about but that
    has no data on this record produces a neutral "not available"
    sentence, never a checkmark.
    """

    preferences = preferences or {}

    if dimension_results is None:
        dimension_results = _compute_dimension_results(course, preferences, "")

    lines = []

    requested_topics = preferences.get("requested_topics") or []

    if requested_topics:

        topic_result = dimension_results.get("topic")
        course_name = course.get("course_name", NOT_AVAILABLE)

        if topic_result in (_TOPIC_STRONG, _TOPIC_COMPOUND):
            lines.append(f"✓ {course_name} matches your requested course")

        elif topic_result == _TOPIC_RELATED:
            lines.append(
                f"~ {course_name} is related to your requested topic, "
                "but is not an exact match"
            )

        elif topic_result == _TOPIC_UNKNOWN:
            lines.append("Course subject is not available.")

        else:
            lines.append(f"✗ {course_name} does not match your requested course")

    requested_level = preferences.get("requested_level")

    if requested_level:

        level_result = dimension_results.get("level")

        if level_result == _LEVEL_MATCH:
            lines.append(f"✓ {requested_level} level matches your preference")

        elif level_result == _LEVEL_UNSPECIFIED:
            lines.append("Level information is not available.")

        else:
            lines.append(
                f"✗ Course level is {course.get('level')}, "
                f"while you requested {requested_level}."
            )

    requested_location = preferences.get("requested_location")

    if requested_location:

        location_result = dimension_results.get("location")

        if location_result == _LOCATION_MATCH:
            lines.append(f"✓ Located in {course.get('location')}")

        elif location_result == _LOCATION_UNSPECIFIED:
            lines.append("Location information is not available.")

        else:
            lines.append(
                f"✗ Course is located in {course.get('location')}, "
                f"not {requested_location}."
            )

    budget_max = preferences.get("budget_max")

    if budget_max is not None:

        budget_result = dimension_results.get("budget")

        if budget_result == _BUDGET_MATCH:
            lines.append("✓ Fee is within your budget")

        elif budget_result == _BUDGET_UNSPECIFIED:
            lines.append("Fee information is not available.")

        else:
            lines.append(
                f"✗ Fee is {course.get('fees')}, "
                f"above your budget of ₹{budget_max}."
            )

    mode = preferences.get("mode")

    if mode:

        mode_result = dimension_results.get("mode")

        if mode_result == _MODE_MATCH:
            lines.append(f"✓ {mode} mode matches your preference")

        elif mode_result == _MODE_UNSPECIFIED:
            lines.append("Mode information is not available.")

        else:
            lines.append(
                f"✗ Course mode is {course.get('learning_mode')}, "
                f"while you requested {mode}."
            )

    if preferences.get("placement_preference"):

        placement_result = dimension_results.get("placement")

        if placement_result == _PLACEMENT_MATCH:
            lines.append("✓ Placement/career support is mentioned")

        elif placement_result == _PLACEMENT_UNSPECIFIED:
            lines.append("Placement information is not available.")

        else:
            lines.append(
                "✗ Placement support is explicitly stated as not provided."
            )

    return "\n".join(lines)
