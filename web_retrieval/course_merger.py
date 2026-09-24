"""
Course/institute deduplication and merging -- Step 3.

Takes the per-source Course records produced by
web_retrieval.course_extractor.extract_course_record (one record per
source page, never merged) and combines the ones that represent the
SAME real-world course/institute offering into a single clean record,
while keeping genuinely different courses/institutes separate.

Design notes:

- No new dependency was added for fuzzy matching. rapidfuzz is not
  currently installed in this project (confirmed before writing this
  module), and the string comparisons needed here are short
  institute/course names, not large-scale search -- the stdlib
  difflib.SequenceMatcher is accurate enough for that and avoids
  growing requirements.txt / the Render build for a capstone-scale
  project. If retrieval volume grows enough that this becomes a
  bottleneck, rapidfuzz would be the natural upgrade.

- Two records are only ever considered the same course when BOTH their
  normalized institute name AND normalized course name match (exactly
  or above a strict similarity threshold) -- matching on institute
  alone, or course alone, is never enough, so "ABC Institute / Data
  Science" and "ABC Institute / Python Development" always stay
  separate even though they share an institute.

- A record whose institute or course name is "Not available" is never
  merged with anything on that basis -- there's no real identity
  signal to compare, so merging would be a guess, not a fact.
"""

import difflib
import re

from web_retrieval.course_extractor import (
    NOT_AVAILABLE,
    validate_course_record,
)


# ============================================================
# NAME NORMALIZATION
# ============================================================

_DASH_VARIANTS = re.compile(r"[‐‑‒–—―]")

# Kept intentionally small and Kerala-specific (matching this
# project's scope) -- only a suffix that matches one of these known
# place names is stripped from the CANONICAL institute key. The
# original display name is never touched.
_KNOWN_LOCATIONS = (
    "calicut", "kozhikode", "kochi", "cochin", "ernakulam",
    "trivandrum", "thiruvananthapuram", "kollam", "kottayam",
    "thrissur", "trichur", "palakkad", "palghat", "kannur",
    "cannanore", "alappuzha", "alleppey", "malappuram", "idukki",
    "wayanad", "kasaragod", "pathanamthitta",
)

_LOCATION_SUFFIX_PATTERN = re.compile(
    r"[-,]\s*(" + "|".join(_KNOWN_LOCATIONS) + r")\s*$"
)


def normalize_institute_name(raw_name):
    """
    Returns {"canonical": ..., "display": ...}.

    canonical is a lowercased, punctuation-normalized key with
    parenthetical branding and a known trailing location dropped, so
    "Codeme Hub", "Codeme Hub (Tech Learning)", "CODEME HUB" and
    "Codeme Hub - Calicut" all normalize to the same canonical key.
    display always preserves the original text unchanged (for showing
    the user / keeping the source's own wording).
    """

    if not raw_name or raw_name == NOT_AVAILABLE:
        return {"canonical": "", "display": raw_name or NOT_AVAILABLE}

    display = re.sub(r"\s+", " ", raw_name).strip()

    text = display.lower()
    text = _DASH_VARIANTS.sub("-", text)
    text = re.sub(r"\([^)]*\)", " ", text)  # drop any parenthetical branding
    text = _LOCATION_SUFFIX_PATTERN.sub("", text)
    text = re.sub(r"[^a-z0-9\s&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return {"canonical": text, "display": display}


# Parenthetical qualifiers that only indicate LEVEL, not a distinct
# course -- safe to fold into the same canonical key. Any other
# parenthetical (e.g. "(Weekend Batch)", "(Part Time)") is left as-is,
# since it could plausibly distinguish a genuinely different offering.
_COURSE_LEVEL_PARENS = {
    "beginner", "beginners", "basic", "basics",
    "intro", "introduction", "fresher", "freshers",
}

# Generic trailing words that don't change what the course actually
# is ("Data Science Course" == "Data Science").
_COURSE_GENERIC_SUFFIXES = {
    "course", "program", "programme", "training", "certification",
}


def _strip_level_paren(match):

    inner = match.group(1).strip().lower()

    return "" if inner in _COURSE_LEVEL_PARENS else match.group(0)


def normalize_course_name(raw_name):
    """
    Returns {"canonical": ..., "display": ...}.

    Only strips parenthetical LEVEL qualifiers ("(Beginners)",
    "(Basics)", ...) and generic trailing filler words ("Course",
    "Program", ...) -- never a leading/adjacent qualifying word, so
    "Advanced Data Science" and "Data Science" deliberately stay
    distinct canonical keys and are never merged by this function
    alone.
    """

    if not raw_name or raw_name == NOT_AVAILABLE:
        return {"canonical": "", "display": raw_name or NOT_AVAILABLE}

    display = re.sub(r"\s+", " ", raw_name).strip()

    text = display.lower()
    text = _DASH_VARIANTS.sub("-", text)
    text = re.sub(r"\(([^)]*)\)", _strip_level_paren, text)

    words = text.split()

    while words and words[-1].strip(".,") in _COURSE_GENERIC_SUFFIXES:
        words.pop()

    text = " ".join(words)
    text = re.sub(r"[^a-z0-9\s&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return {"canonical": text, "display": display}


def _normalize_plain(value):
    """Simple lowercase/whitespace normalization for location comparison."""

    if not value or value == NOT_AVAILABLE:
        return ""

    return re.sub(r"\s+", " ", value.strip().lower())


def _fuzzy_ratio(a: str, b: str) -> float:

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio() * 100


# Strict on purpose -- this is the guard against "aggressively merging
# unrelated institutes/courses" the spec calls out repeatedly.
FUZZY_MATCH_THRESHOLD = 90.0


# ============================================================
# SOURCE PRIORITY RANKING
# ============================================================
# Reuses the priority values web_retrieval.course_extractor already
# derives from data/sources.json (strings "1".."6", or "unknown") --
# no new ranking system. Lower rank number = higher authority, matching
# sources.json's own scale; "unknown" ranks below every known category.

_PRIORITY_RANK = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
_UNKNOWN_RANK = 99


def _priority_rank(priority_value) -> int:

    return _PRIORITY_RANK.get(str(priority_value), _UNKNOWN_RANK)


# ============================================================
# IDENTITY MATCHING
# ============================================================

def _is_same_course(candidate, existing) -> bool:
    """
    candidate/existing are the internal annotated dicts built in
    deduplicate_courses (see below), each holding "record", "inst",
    "course".
    """

    inst_a = candidate["inst"]["canonical"]
    inst_b = existing["inst"]["canonical"]

    course_a = candidate["course"]["canonical"]
    course_b = existing["course"]["canonical"]

    # Never merge on a missing identity signal -- both sides need a
    # real institute name AND a real course name to compare.
    if not inst_a or not inst_b or not course_a or not course_b:
        return False

    inst_match = (
        inst_a == inst_b
        or _fuzzy_ratio(inst_a, inst_b) >= FUZZY_MATCH_THRESHOLD
    )

    if not inst_match:
        return False

    course_match = (
        course_a == course_b
        or _fuzzy_ratio(course_a, course_b) >= FUZZY_MATCH_THRESHOLD
    )

    if not course_match:
        return False

    # If both sides actually state a location and it differs, treat
    # them as different offerings (e.g. different branches) rather
    # than assuming they're the same -- only applies when BOTH sides
    # have a real location to compare against.
    loc_a = _normalize_plain(candidate["record"].get("location"))
    loc_b = _normalize_plain(existing["record"].get("location"))

    if loc_a and loc_b and loc_a != loc_b:
        return False

    return True


# ============================================================
# FIELD MERGING
# ============================================================

_MERGE_STRING_FIELDS = (
    "duration",
    "eligibility",
    "fees",
    "location",
    "learning_mode",
    "certification",
    "admission_information",
    "placement_information",
    "course_category",
    "level",
)


def _pick_best_display(records, field_name):
    """
    Among records that actually have a value for field_name, prefer
    the one from the highest-authority source; break ties by
    preferring the shortest (plainest / least sub-branded) text, e.g.
    "Codeme Hub" over "Codeme Hub (Tech Learning)" when both sources
    are otherwise equally authoritative.
    """

    candidates = [
        (record.get(field_name), _priority_rank(record.get("source_priority")))
        for record in records
        if record.get(field_name) and record.get(field_name) != NOT_AVAILABLE
    ]

    if not candidates:
        return NOT_AVAILABLE

    best_rank = min(rank for _, rank in candidates)
    best_tier = [name for name, rank in candidates if rank == best_rank]

    return min(best_tier, key=len)


def _merge_field(field_name, records):
    """
    Returns (value, conflict_info_or_None).

    - No source has this field -> ("Not available", None).
    - Every source that has it agrees (case/whitespace-insensitive) ->
      (that value, None). This also covers the "only one source had
      it" case -- Rule 7 (never let "Not available" overwrite a real
      value) falls straight out of only ever looking at candidates
      that actually have a real value.
    - Sources disagree, but one distinct value is backed by a source
      whose priority is strictly better than every source backing any
      other distinct value -> (that value, conflict record noting how
      it was resolved).
    - Otherwise (ties, or none of the disagreeing sources have a
      "known" priority to prefer) -> ("Varies by source", conflict
      record with every candidate preserved, unresolved).
    """

    candidates = []

    for record in records:

        value = record.get(field_name)

        if value and value != NOT_AVAILABLE:
            candidates.append((
                value,
                _priority_rank(record.get("source_priority")),
                record.get("source_url", "")
            ))

    if not candidates:
        return NOT_AVAILABLE, None

    grouped = {}

    for value, rank, url in candidates:

        key = re.sub(r"\s+", " ", value.strip().lower())
        grouped.setdefault(key, {"value": value, "best_rank": rank, "urls": []})
        grouped[key]["urls"].append(url)
        grouped[key]["best_rank"] = min(grouped[key]["best_rank"], rank)

    if len(grouped) == 1:
        only = next(iter(grouped.values()))
        return only["value"], None

    # Genuine conflict: two or more distinct values.
    ranks_present = sorted({info["best_rank"] for info in grouped.values()})
    best_rank = ranks_present[0]
    leaders = [info for info in grouped.values() if info["best_rank"] == best_rank]

    all_candidates_report = [
        {"value": info["value"], "source_urls": info["urls"], "best_rank": info["best_rank"]}
        for info in grouped.values()
    ]

    if len(leaders) == 1 and best_rank < _UNKNOWN_RANK:

        resolved_value = leaders[0]["value"]

        return resolved_value, {
            "resolution": "source_priority",
            "resolved_value": resolved_value,
            "candidates": all_candidates_report,
        }

    return "Varies by source", {
        "resolution": "unresolved_conflict",
        "resolved_value": "Varies by source",
        "candidates": all_candidates_report,
    }


def _merge_curriculum(records):

    seen = set()
    merged = []

    for record in records:

        for item in record.get("curriculum") or []:

            key = item.strip().lower()

            if key and key not in seen:
                seen.add(key)
                merged.append(item.strip())

    return merged


def _collect_sources(records):
    """
    Deduplicated list of {"source_url", "source_title",
    "source_priority"}, in first-seen order, one entry per distinct
    source_url -- never drops a source just because it was processed
    later.
    """

    sources = []
    seen_urls = set()

    for record in records:

        url = record.get("source_url", "")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        sources.append({
            "source_url": url,
            "source_title": record.get("source_title", NOT_AVAILABLE),
            "source_priority": record.get("source_priority", "unknown"),
        })

    return sources


def _primary_source(sources):
    """The highest-authority source among a merged record's sources."""

    if not sources:
        return {
            "source_url": "",
            "source_title": NOT_AVAILABLE,
            "source_priority": "unknown",
        }

    return min(sources, key=lambda s: _priority_rank(s["source_priority"]))


def _build_merged_record(records):

    merged = {
        "institute_name": _pick_best_display(records, "institute_name"),
        "course_name": _pick_best_display(records, "course_name"),
        "curriculum": _merge_curriculum(records),
    }

    conflicts = {}

    for field in _MERGE_STRING_FIELDS:

        value, conflict = _merge_field(field, records)
        merged[field] = value

        if conflict:
            conflicts[field] = conflict

    sources = _collect_sources(records)
    primary = _primary_source(sources)

    merged["source_url"] = primary["source_url"]
    merged["source_title"] = primary["source_title"]
    merged["source_priority"] = primary["source_priority"]
    merged["source_urls"] = [s["source_url"] for s in sources]
    merged["sources"] = sources

    merged = validate_course_record(merged)

    # validate_course_record doesn't know about the extra multi-source
    # fields -- restore them after validation normalizes everything
    # else.
    merged["source_urls"] = [s["source_url"] for s in sources]
    merged["sources"] = sources

    return merged, conflicts


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def deduplicate_courses(course_records):
    """
    Groups the given Course records (see
    web_retrieval.course_extractor.extract_course_record) into
    distinct real-world courses, merging every record that's
    identified as the same course/institute offering into one clean
    record, and leaving every genuinely different course/institute
    untouched as its own record.

    Returns:
        {
            "courses": [Course record, ...],  # deduplicated/merged
            "merge_report": {
                "original_count": int,
                "final_count": int,
                "merges_performed": int,
                "merges": [ {..per-merge detail..}, ... ],
                "log_lines": [str, ...],  # human-readable, for
                                          # logging/debugging only --
                                          # never shown to end users
            }
        }

    Never raises: malformed items in course_records (wrong type,
    missing keys) are normalized via validate_course_record before
    any comparison happens.
    """

    safe_records = [validate_course_record(r) for r in (course_records or [])]

    if not safe_records:
        return {
            "courses": [],
            "merge_report": {
                "original_count": 0,
                "final_count": 0,
                "merges_performed": 0,
                "merges": [],
                "log_lines": ["Original course records: 0", "After deduplication: 0"],
            },
        }

    annotated = [
        {
            "record": record,
            "inst": normalize_institute_name(record.get("institute_name")),
            "course": normalize_course_name(record.get("course_name")),
        }
        for record in safe_records
    ]

    groups = []

    for item in annotated:

        placed = False

        for group in groups:

            if _is_same_course(item, group["members"][0]):
                group["members"].append(item)
                placed = True
                break

        if not placed:
            groups.append({"members": [item]})

    courses = []
    merges = []
    log_lines = [
        f"Original course records: {len(safe_records)}",
    ]

    for group in groups:

        members = [m["record"] for m in group["members"]]

        if len(members) == 1:

            merged, _conflicts = _build_merged_record(members)
            courses.append(merged)
            continue

        merged, conflicts = _build_merged_record(members)
        courses.append(merged)

        detail = {
            "display_institute": merged["institute_name"],
            "display_course": merged["course_name"],
            "source_count": len(members),
            "merged_source_urls": merged["source_urls"],
            "original_institute_names": [m.get("institute_name") for m in members],
            "original_course_names": [m.get("course_name") for m in members],
            "conflicts": conflicts,
        }

        merges.append(detail)

        original_names = " + ".join(
            f'"{name}"' for name in dict.fromkeys(detail["original_institute_names"])
        )

        log_lines.append(
            f"MERGE: {original_names} -> {merged['course_name']} "
            f"({len(members)} source records merged)"
        )

    log_lines.append(f"After deduplication: {len(courses)}")
    log_lines.append(f"Merged groups: {len(merges)}")

    return {
        "courses": courses,
        "merge_report": {
            "original_count": len(safe_records),
            "final_count": len(courses),
            "merges_performed": len(merges),
            "merges": merges,
            "log_lines": log_lines,
        },
    }
