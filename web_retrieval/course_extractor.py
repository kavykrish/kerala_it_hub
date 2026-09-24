"""
Structured course extraction -- foundation layer.

Turns the raw, field-labelled chunks retrieved for ONE source page
(see web_retrieval.retriever.retrieve_field_aware_chunks) into a single
structured Course record for that page, using the schema already
defined in data/sources.json's "course_information_fields" (extended
with a couple of fields -- level, placement_information, source_title
-- that the schema didn't have yet, but no existing field names were
changed).

Extraction here is entirely deterministic (regex/keyword-based against
text that is already known to exist in the retrieved chunks) -- no LLM
call is made. This keeps the fix free of any extra Groq usage (the
account's tokens-per-minute budget is already tight, see
backend/rag_generator.py's retry logic) and satisfies the project's own
rule to prefer deterministic extraction when the retrieved text already
contains what's needed. parse_llm_course_json is provided as a
validated entry point for an LLM-assisted fallback, but nothing in the
current pipeline calls it yet.

This module does NOT merge records across sources/pages -- each
Course record stays tied to exactly the one source_url it came from.
Cross-source institute matching/merging is the next step, not this one.
"""

import json
import os
import re

from web_retrieval.chunker import HEADINGS
from web_retrieval.retriever import detect_field_category


# ============================================================
# COURSE SCHEMA
# ============================================================
# Matches data/sources.json's course_information_fields where the name
# already existed there (institute_name, course_name, course_category,
# duration, eligibility, fees, location, learning_mode, curriculum,
# certification, admission_information, source_url, source_priority).
# "level", "placement_information" and "source_title" are new -- they
# didn't exist in sources.json's field list, so nothing there needed to
# change. sources.json also has "course_status", "official_website" and
# "source_type", which aren't produced here since they're not part of
# the extraction this step covers.

NOT_AVAILABLE = "Not available"

COURSE_FIELDS = (
    "institute_name",
    "course_name",
    "course_category",
    "level",
    "duration",
    "eligibility",
    "fees",
    "location",
    "learning_mode",
    "curriculum",
    "certification",
    "admission_information",
    "placement_information",
    "source_url",
    "source_title",
    "source_priority",
)

# Fields whose value is a list (currently just curriculum) rather than
# a plain string.
_LIST_FIELDS = ("curriculum",)

# Which structured field each field-aware retrieval category (see
# web_retrieval/retriever.py's FIELD_LABEL_CATEGORIES) fills.
_CATEGORY_TO_FIELD = {
    "duration": "duration",
    "eligibility": "eligibility",
    "fees": "fees",
    "learning_mode": "learning_mode",
    "location": "location",
    "certification": "certification",
    "admission_information": "admission_information",
    "placement_information": "placement_information",
    "curriculum": "curriculum",
}

# Cap on how much text a single string field can hold -- guards against
# a mis-split chunk dumping an entire page section into one field.
_FIELD_VALUE_CHAR_LIMIT = 400


def empty_course_record(
    source_url: str = "",
    source_title: str = "",
    source_priority: str = "unknown"
):
    """
    A fully-shaped Course record with every field set to
    "Not available" (or [] for list fields) except source metadata.
    Never omits a key -- callers can always safely index every field
    in COURSE_FIELDS.
    """

    record = {field: NOT_AVAILABLE for field in COURSE_FIELDS}

    for field in _LIST_FIELDS:
        record[field] = []

    record["source_url"] = source_url or ""
    record["source_title"] = source_title or NOT_AVAILABLE
    record["source_priority"] = source_priority or "unknown"

    return record


# ============================================================
# SOURCE PRIORITY
# ============================================================
# Reuses the priority scale already defined in data/sources.json's
# source_policy (government=1, university=2,
# government_skill_development=3, technical_education=4,
# official_institute=5, trusted_education_platform=6) rather than
# inventing a new one. sources.json's own entries are human-readable
# category descriptions with no url/domain field to match against --
# it explicitly says official institute sources are "discovered
# dynamically" -- so this is a best-effort heuristic bridge from a
# real URL to that existing scale, not a literal lookup table.

_SOURCES_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sources.json"
)


def _load_source_policy():

    try:
        with open(_SOURCES_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)

    except (OSError, json.JSONDecodeError):
        return {}


SOURCE_POLICY = _load_source_policy()

_GOV_DOMAIN_HINTS = (".gov.in", ".nic.in")
_SKILL_DEV_DOMAIN_HINTS = ("asapkerala", "kite.kerala.gov.in")
_TECHNICAL_EDU_DOMAIN_HINTS = ("sbte", "dtekerala", "itikerala")
_UNIVERSITY_DOMAIN_HINTS = (
    "keralauniversity", "cusat.ac.in", "mgu", "uoc.ac.in",
    "calicutuniversity", "kannuruniversity", "duk.ac.in",
    "cukerala.ac.in", "ktu.edu.in", ".ac.in"
)
_THIRD_PARTY_PLATFORM_HINTS = (
    "coursera.org", "udemy.com", "simplilearn.com", "edx.org",
    "analyticsvidhya.com", "geeksforgeeks.org", "shiksha.com",
    "collegedunia.com", "careers360.com", "naukri.com",
    "upgrad.com", "greatlearning.in", "internshala.com"
)


def determine_source_priority(url: str) -> str:
    """
    Best-effort classification of a source URL against the priority
    scale already defined in data/sources.json, so a later merge step
    can prefer a higher-authority source when two sources disagree on
    a field. Falls back to priority "5" (official_institute_sources,
    the JSON's own catch-all for a dynamically-discovered institute
    site) since most search results for a specific course ARE an
    institute's own page, not a government/university/aggregator one.
    """

    if not url:
        return "unknown"

    lowered = url.lower()

    # Specific known categories are checked before the generic .gov.in
    # / .ac.in catch-alls, since e.g. ASAP Kerala's own domain IS a
    # .gov.in domain but sources.json specifically categorizes it under
    # government_skill_development (priority 3), not plain government
    # (priority 1) -- the specific match should win.
    if any(hint in lowered for hint in _SKILL_DEV_DOMAIN_HINTS):
        return "3"

    if any(hint in lowered for hint in _TECHNICAL_EDU_DOMAIN_HINTS):
        return "4"

    if any(hint in lowered for hint in _UNIVERSITY_DOMAIN_HINTS):
        return "2"

    if any(hint in lowered for hint in _GOV_DOMAIN_HINTS):
        return "1"

    if any(hint in lowered for hint in _THIRD_PARTY_PLATFORM_HINTS):
        return "6"

    return "5"


# ============================================================
# COURSE CATEGORY / LEVEL KEYWORDS
# ============================================================
# course_category is detected from data/course_keywords.json, which
# (per the audit) existed but was never actually loaded by any code --
# this is the first thing that puts it to use. Matching is a literal
# substring check against phrases that are already in the JSON, so a
# match is always something that literally appears in the retrieved
# text, never an invented category.

_COURSE_KEYWORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "course_keywords.json"
)


def _load_course_keywords():

    try:
        with open(_COURSE_KEYWORDS_PATH, encoding="utf-8") as f:
            return json.load(f)

    except (OSError, json.JSONDecodeError):
        return {}


COURSE_KEYWORDS = _load_course_keywords()

_LEVEL_LABELS = {
    "beginner": "Beginner",
    "fresher": "Beginner",
    "freshers": "Beginner",
    "no prior experience": "Beginner",
    "no coding experience": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
    "professionals": "Advanced",
    "working professionals": "Advanced",
    "experienced": "Advanced",
}


def _guess_course_category(chunks):
    """
    Longest literal keyword match (from data/course_keywords.json)
    found anywhere in the page's retrieved chunks, so "Data Science" is
    preferred over a shorter, more generic overlapping match.
    """

    combined = " ".join(chunk for chunk in chunks if chunk).lower()

    best_match = None
    best_length = 0

    for phrases in COURSE_KEYWORDS.values():

        for phrase in phrases:

            if phrase.lower() in combined and len(phrase) > best_length:
                best_match = phrase
                best_length = len(phrase)

    return best_match


def _guess_level(chunks):

    combined = " ".join(chunk for chunk in chunks if chunk).lower()

    for keyword, label in _LEVEL_LABELS.items():

        if keyword in combined:
            return label

    return None


def _guess_institute_name(source_title):
    """
    Page/search-result titles are very commonly
    "<Course/Page Name> | <Institute Name>" or "... - <Institute
    Name>" -- the institute/brand name conventionally trails the
    separator. Returns None (never a guess) if there's no such
    separator to go on.
    """

    if not source_title:
        return None

    for sep in (" | ", " – ", " — ", " - "):

        if sep in source_title:

            parts = [p.strip() for p in source_title.split(sep) if p.strip()]

            if len(parts) >= 2:
                return parts[-1]

    return None


def _guess_course_name(general_chunks, source_title):
    """
    The first short, title-like line among the chunks NOT already
    claimed by a field heading is usually the page's own course
    heading. Falls back to the leading segment of the source title.
    Never fabricates -- returns None if nothing plausible is found.
    """

    for chunk in general_chunks:

        stripped = chunk.strip()

        if not stripped:
            continue

        first_line = stripped.splitlines()[0].strip()

        if first_line and len(first_line) <= 120 and not first_line.endswith("."):
            return first_line

    if source_title:

        for sep in (" | ", " – ", " — ", " - "):

            if sep in source_title:
                return source_title.split(sep)[0].strip()

        return source_title.strip()

    return None


# ============================================================
# FIELD VALUE EXTRACTION
# ============================================================

_HEADING_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in HEADINGS) + r")\b\s*[:\-–]?\s*",
    re.IGNORECASE
)


def strip_heading_prefix(chunk_text: str) -> str:
    """
    Removes the leading field heading from a chunk (however it was
    written -- a standalone heading line, or inline like "Duration: 3
    months"), leaving just the field's actual value text. Uses the
    exact same heading vocabulary the chunker split on
    (web_retrieval.chunker.HEADINGS), so this is guaranteed consistent
    with how the chunk was produced rather than guessing separately.
    """

    if not chunk_text or not chunk_text.strip():
        return ""

    lines = [line for line in chunk_text.strip().splitlines() if line.strip()]

    if not lines:
        return ""

    first_line = lines[0]
    match = _HEADING_PATTERN.match(first_line)

    if not match:
        return chunk_text.strip()

    remainder = first_line[match.end():].strip()
    rest_lines = [line.strip() for line in lines[1:]]

    body = ([remainder] if remainder else []) + rest_lines

    return "\n".join(body).strip()


def _clean_field_value(text: str) -> str:

    collapsed = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

    return collapsed[:_FIELD_VALUE_CHAR_LIMIT]


# ============================================================
# EXTRACTION
# ============================================================

def extract_course_record(retrieved_chunks, source_metadata=None):
    """
    Build one structured Course record for a single source page from
    its retrieved chunks (see
    web_retrieval.retriever.retrieve_field_aware_chunks).

    Args:
        retrieved_chunks: list of chunk text strings retrieved for ONE
            source page. Accepts either plain strings or the
            (final_score, ..., chunk) tuples semantic_retrieve_chunks /
            retrieve_field_aware_chunks return -- the chunk text is
            pulled out of either shape automatically.
        source_metadata: dict with "source_url" and optionally
            "source_title" / "source_priority". If source_priority
            isn't given, it's derived from source_url via
            determine_source_priority.

    Never invents a value: any field not explicitly present in the
    retrieved chunks is left as "Not available" (or [] for
    curriculum). Never raises -- malformed input degrades to an
    all-"Not available" record instead of crashing the caller.
    """

    source_metadata = source_metadata or {}

    source_url = source_metadata.get("source_url") or ""
    source_title = source_metadata.get("source_title") or ""
    source_priority = source_metadata.get("source_priority")

    if not source_priority:
        source_priority = determine_source_priority(source_url)

    record = empty_course_record(
        source_url=source_url,
        source_title=source_title,
        source_priority=source_priority
    )

    # Accept either plain chunk strings or (score, ..., chunk) tuples.
    chunk_texts = []

    for item in (retrieved_chunks or []):

        if isinstance(item, str):
            chunk_texts.append(item)

        elif isinstance(item, (list, tuple)) and item:
            chunk_texts.append(item[-1])

    chunk_texts = [c for c in chunk_texts if c and c.strip()]

    if not chunk_texts:
        return validate_course_record(record)

    field_values = {}
    general_chunks = []

    for chunk in chunk_texts:

        category = detect_field_category(chunk)

        if category:
            value = strip_heading_prefix(chunk)

            if value:
                field_values.setdefault(category, []).append(value)

        else:
            general_chunks.append(chunk)

    for category, field_name in _CATEGORY_TO_FIELD.items():

        values = field_values.get(category)

        if not values:
            continue

        if field_name in _LIST_FIELDS:

            items = []

            for value in values:
                for line in value.splitlines():

                    cleaned = line.strip(" -•\t")

                    if cleaned:
                        items.append(cleaned)

            if items:
                record[field_name] = items

        else:

            combined = " ".join(v for v in values if v)
            cleaned = _clean_field_value(combined)

            if cleaned:
                record[field_name] = cleaned

    guessed_institute = _guess_institute_name(source_title)

    if guessed_institute:
        record["institute_name"] = guessed_institute

    guessed_course = _guess_course_name(general_chunks, source_title)

    if guessed_course:
        record["course_name"] = guessed_course

    guessed_category = _guess_course_category(chunk_texts)

    if guessed_category:
        record["course_category"] = guessed_category

    guessed_level = _guess_level(chunk_texts)

    if guessed_level:
        record["level"] = guessed_level

    return validate_course_record(record)


# ============================================================
# VALIDATION
# ============================================================

def validate_course_record(record):
    """
    Defensively normalizes any dict-like input into a fully-shaped,
    safe Course record -- every key in COURSE_FIELDS guaranteed
    present, string fields guaranteed to be non-empty strings (else
    "Not available"), curriculum guaranteed to be a list of non-empty
    strings. Never raises, regardless of how malformed the input is
    (wrong types, missing keys, extra keys, not even a dict).
    """

    if not isinstance(record, dict):
        return empty_course_record()

    safe = empty_course_record(
        source_url=(
            record.get("source_url")
            if isinstance(record.get("source_url"), str)
            else ""
        ),
        source_title=(
            record.get("source_title")
            if isinstance(record.get("source_title"), str)
            else ""
        ),
        source_priority=(
            record.get("source_priority")
            if isinstance(record.get("source_priority"), str)
            else "unknown"
        ),
    )

    for field in COURSE_FIELDS:

        if field in ("source_url", "source_title", "source_priority"):
            continue

        value = record.get(field, NOT_AVAILABLE)

        if field in _LIST_FIELDS:

            if isinstance(value, list):
                safe[field] = [
                    str(item).strip()
                    for item in value
                    if str(item).strip()
                ]
            else:
                safe[field] = []

        else:

            if isinstance(value, str) and value.strip():
                safe[field] = value.strip()
            else:
                safe[field] = NOT_AVAILABLE

    return safe


def parse_llm_course_json(raw_text, source_metadata=None):
    """
    Parse and validate a JSON Course record that an LLM-assisted
    extraction fallback *would* return, if one is added later -- not
    currently called anywhere in the pipeline (extraction is fully
    deterministic for now, see this module's docstring).

    Never trusts the input blindly: a parse failure, a non-dict
    payload, unknown keys, or wrong-typed values all degrade to a safe
    "Not available" record via validate_course_record rather than
    raising or letting the model's own claims about source_url /
    source_title / source_priority override the caller-supplied
    metadata, which is always authoritative.
    """

    try:
        data = json.loads(raw_text)

    except (TypeError, json.JSONDecodeError):
        data = None

    if not isinstance(data, dict):
        data = {}

    record = validate_course_record(data)

    meta = source_metadata or {}

    if meta.get("source_url"):
        record["source_url"] = meta["source_url"]

    if meta.get("source_title"):
        record["source_title"] = meta["source_title"]

    if meta.get("source_priority"):
        record["source_priority"] = meta["source_priority"]
    elif record["source_priority"] == "unknown" and record["source_url"]:
        record["source_priority"] = determine_source_priority(record["source_url"])

    return record
