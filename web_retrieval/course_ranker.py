"""
Deterministic, query-aware ranking of deduplicated Course records --
Step 5 Parts 3-7.

Takes the already-deduplicated/merged records from
web_retrieval.course_merger.deduplicate_courses and a query_intent
dict (see web_retrieval.query_intent.parse_query_intent) and returns
the SAME records, REORDERED so the ones that actually match what the
user asked about (named institute, topic, requested fields, level,
location) surface first.

This never drops a record and never invents a relationship the source
data doesn't support:

- A course whose identity doesn't match the requested topic is never
  removed -- it's simply ranked below the ones that do, so an LLM (or
  a human) reading the ranked list can still see it and say "these
  other courses were also found, but aren't Data Science".
- A related-but-different course (Machine Learning, AI) is NEVER
  treated as if its course_name/course_category WERE the requested
  topic. It can rank above a totally unrelated course when its own
  curriculum explicitly mentions the requested topic, but it can never
  reach the same rank as a course whose own identity actually IS the
  requested topic.
- Level and location are SOFT signals only: a record with no stated
  level/location is treated as neutral (still eligible, just not
  boosted), never penalized as if it conflicted.

The internal rank numbers below are ordering keys only -- lower always
means "shown first" -- and are never exposed outside this module (not
stored on the record, not sent to the LLM or the API response).
"""

import re

from web_retrieval.course_extractor import NOT_AVAILABLE
from web_retrieval.kerala_locations import canonical_location


# ============================================================
# TOPIC MATCHING
# ============================================================

_TOPIC_STRONG = 0     # course_name/category's identity IS the topic
_TOPIC_COMPOUND = 1   # topic is part of a compound identity
_TOPIC_RELATED = 2    # not in the identity, but curriculum backs it
_TOPIC_UNKNOWN = 3    # no course_name/category to judge at all
_TOPIC_UNRELATED = 4  # a genuinely different, unrelated course


def _topic_rank(course, requested_topics):
    """
    See module docstring. Only ever looks at structured fields already
    on the record (course_name, course_category, curriculum) -- never
    re-scans raw retrieved text to manufacture a match.
    """

    if not requested_topics:
        return _TOPIC_STRONG

    course_name = (course.get("course_name") or "").strip()
    course_category = (course.get("course_category") or "").strip()

    has_identity = (
        course_name and course_name != NOT_AVAILABLE
    ) or (
        course_category and course_category != NOT_AVAILABLE
    )

    if not has_identity:
        return _TOPIC_UNKNOWN

    identity_parts = [
        text.lower()
        for text in (course_name, course_category)
        if text and text != NOT_AVAILABLE
    ]

    for topic in requested_topics:

        for identity_text in identity_parts:

            if identity_text == topic:
                return _TOPIC_STRONG

            if re.search(r"\b" + re.escape(topic) + r"\b", identity_text):
                return _TOPIC_COMPOUND

    curriculum_text = " ".join(course.get("curriculum") or []).lower()

    for topic in requested_topics:

        if topic in curriculum_text:
            return _TOPIC_RELATED

    return _TOPIC_UNRELATED


# ============================================================
# REQUESTED-FIELD COVERAGE
# ============================================================

def _field_coverage_rank(course, requested_fields):
    """
    Lower = more of the specifically-requested fields have a real
    value on this record. Never used to drop a record -- only to
    prioritize the ones that can actually answer what was asked.
    """

    if not requested_fields:
        return 0

    missing = sum(
        1 for field in requested_fields
        if not course.get(field) or course.get(field) == NOT_AVAILABLE
    )

    return missing


# ============================================================
# LEVEL (SOFT RANKING -- NEVER A HARD FILTER)
# ============================================================

_LEVEL_MATCH = 0
_LEVEL_UNSPECIFIED = 1
_LEVEL_CONFLICT = 2


def _level_rank(course, requested_level):

    if not requested_level:
        return _LEVEL_MATCH

    course_level = course.get("level")

    if not course_level or course_level == NOT_AVAILABLE:
        return _LEVEL_UNSPECIFIED

    if course_level.strip().lower() == requested_level.strip().lower():
        return _LEVEL_MATCH

    return _LEVEL_CONFLICT


# ============================================================
# LOCATION (SOFT RANKING)
# ============================================================

_LOCATION_MATCH = 0
_LOCATION_UNSPECIFIED = 1
_LOCATION_DIFFERENT = 2


def _location_rank(course, requested_location):
    """
    Compares CANONICAL place names (Step 5B), not raw substrings, so a
    course whose page said "Cochin" still matches a query that asked
    for "Kochi" -- see web_retrieval.kerala_locations.canonical_location.
    Still a purely soft signal: a record with no recognizable location
    at all is neutral (_LOCATION_UNSPECIFIED), never penalized as if
    it conflicted.
    """

    if not requested_location:
        return _LOCATION_MATCH

    course_location = course.get("location")

    if not course_location or course_location == NOT_AVAILABLE:
        return _LOCATION_UNSPECIFIED

    canonical_course_location = canonical_location(course_location)
    canonical_requested_location = canonical_location(requested_location)

    if not canonical_course_location or not canonical_requested_location:
        return _LOCATION_UNSPECIFIED

    if canonical_course_location == canonical_requested_location:
        return _LOCATION_MATCH

    return _LOCATION_DIFFERENT


# ============================================================
# NAMED INSTITUTE (Part 7 -- specific institute/course query)
# ============================================================

def _institute_rank(course, query_lower):
    """
    Grounded only in the record's OWN already-extracted institute_name
    -- never a separate, invented list of known institute names. If
    the user's query literally names this record's institute, it's
    almost certainly the one thing they're asking about.
    """

    institute = course.get("institute_name")

    if not institute or institute == NOT_AVAILABLE or not query_lower:
        return 1

    return 0 if institute.strip().lower() in query_lower else 1


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def build_rank_key(course, query_intent, query_lower):
    """
    The five-dimension base ranking key (Step 5 Parts 3-7): named
    institute > topic > requested-field coverage > level > location.
    Exposed as its own function (Step 5B/Advisor Step 2 -- previously
    a local closure inside filter_and_rank_courses) purely so
    web_retrieval.course_advisor can EXTEND this exact same computation
    with its own additional tiers (budget/mode/placement) instead of
    re-implementing topic/level/location matching a second time. This
    is a behavior-preserving extraction only -- the five dimensions
    compute identically to before, in the same order, via the same
    five functions above.

    query_intent (or any dict that carries the same
    "requested_topics"/"requested_fields"/"requested_level"/
    "requested_location" keys, e.g. course_advisor's preferences dict,
    which is a strict superset of query_intent) supplies what each
    dimension is being judged against.
    """

    requested_topics = query_intent.get("requested_topics") or []
    requested_fields = query_intent.get("requested_fields") or []
    requested_level = query_intent.get("requested_level")
    requested_location = query_intent.get("requested_location")

    return (
        _institute_rank(course, query_lower),
        _topic_rank(course, requested_topics),
        _field_coverage_rank(course, requested_fields),
        _level_rank(course, requested_level),
        _location_rank(course, requested_location),
    )


def filter_and_rank_courses(course_records, query_intent=None, query: str = ""):
    """
    Reorders (never drops or mutates the count of) course_records by
    how well each one matches the deterministic query_intent (see
    web_retrieval.query_intent.parse_query_intent) extracted from the
    user's question. "filter" in the name reflects the Step 5 spec's
    own naming -- this function only ever sorts, it never removes a
    record; every record the caller passed in is present in the
    returned list.
    """

    if not course_records:
        return course_records

    query_intent = query_intent or {}
    query_lower = (query or "").lower()

    return sorted(
        course_records,
        key=lambda course: build_rank_key(course, query_intent, query_lower)
    )
