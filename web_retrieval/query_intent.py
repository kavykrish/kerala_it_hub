"""
Deterministic query-intent parsing -- Step 5 Part 2.

Turns a user's raw question into a small structured intent dict, using
ONLY vocabulary that already exists elsewhere in the project (no new
taxonomy, no LLM call): IMPORTANT_TOPICS and FIELD_LABEL_CATEGORIES
from web_retrieval.retriever, _LEVEL_LABELS from
web_retrieval.course_extractor, and canonical_location (built on
KERALA_LOCATIONS/KERALA_LOCATION_ALIASES) from
web_retrieval.kerala_locations.

This never invents a topic/field/level/location that isn't literally
present (as a substring) in the query text -- an intent field is left
empty/None rather than guessed. A query with nothing detectable
returns a safe, entirely empty intent, never an error.
"""

from web_retrieval.retriever import IMPORTANT_TOPICS, FIELD_LABEL_CATEGORIES
from web_retrieval.course_extractor import _LEVEL_LABELS
from web_retrieval.kerala_locations import canonical_location


# Field-aware retrieval's FIELD_LABEL_CATEGORIES also maps
# "course name"/"program name"/"course title"/"category"/"course
# category" -- those identify WHICH course the user means, not a
# DETAIL of it the way duration/fees/eligibility/etc. do, so they're
# excluded from requested_fields here (a query naming a course by
# title isn't "asking about" its own course_name field).
_IDENTITY_FIELDS = {"course_name", "course_category"}

_COMPARISON_PHRASES = (
    "compare", "comparison", " vs ", " vs.", "versus",
    "which is better", "which one is better",
    "which is best", "which one is best",
)


def _find_requested_topics(query_lower):

    matches = []

    for topic in IMPORTANT_TOPICS:

        if topic in query_lower and topic not in matches:
            matches.append(topic)

    return matches


def _find_requested_fields(query_lower):

    fields = []

    for label, field in FIELD_LABEL_CATEGORIES.items():

        if field in _IDENTITY_FIELDS:
            continue

        if label in query_lower and field not in fields:
            fields.append(field)

    return fields


def _find_requested_level(query_lower):

    for keyword, label in _LEVEL_LABELS.items():

        if keyword in query_lower:
            return label

    return None


def _find_requested_location(query_lower):
    """
    Returns the CANONICAL Kerala place name (Step 5B) -- "Kochi" and
    "Cochin" in the query both resolve to the same "kochi" so they
    match a course record regardless of which spelling its own source
    page used (see web_retrieval.kerala_locations.canonical_location
    and web_retrieval.course_ranker's location ranking).
    """

    return canonical_location(query_lower)


def _has_comparison_intent(query_lower):

    return any(phrase in query_lower for phrase in _COMPARISON_PHRASES)


def _empty_intent():

    return {
        "requested_topics": [],
        "requested_fields": [],
        "requested_level": None,
        "requested_location": None,
        "comparison_intent": False,
    }


def parse_query_intent(query: str) -> dict:
    """
    Deterministically extracts:

        requested_topics: known IT topics (see IMPORTANT_TOPICS)
            literally mentioned in the query, e.g. ["data science"].
        requested_fields: structured Course fields (duration, fees,
            eligibility, ...) the query is explicitly asking about,
            e.g. ["eligibility", "duration"].
        requested_level: "Beginner" / "Intermediate" / "Advanced", or
            None if the query doesn't mention a level.
        requested_location: a Kerala location literally named in the
            query, or None.
        comparison_intent: True if the query is asking to compare
            multiple courses/institutes.

    Never raises and never guesses -- an undetectable dimension is
    left empty/None rather than inferred.
    """

    if not query or not query.strip():
        return _empty_intent()

    query_lower = query.lower()

    return {
        "requested_topics": _find_requested_topics(query_lower),
        "requested_fields": _find_requested_fields(query_lower),
        "requested_level": _find_requested_level(query_lower),
        "requested_location": _find_requested_location(query_lower),
        "comparison_intent": _has_comparison_intent(query_lower),
    }
