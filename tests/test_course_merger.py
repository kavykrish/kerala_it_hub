"""
Tests for Step 3 -- course/institute deduplication and merging
(web_retrieval/course_merger.py).

Pure unit/fixture tests -- no live web search, no embedding model, no
Groq call. Course records are built directly (as
extract_course_record itself already returns them, per the Step 2
tests) so these tests focus purely on normalization/dedup/merge logic.
"""

from web_retrieval.course_extractor import NOT_AVAILABLE, empty_course_record
from web_retrieval.course_merger import (
    deduplicate_courses,
    normalize_course_name,
    normalize_institute_name,
)


def make_record(**overrides):
    record = empty_course_record(
        source_url=overrides.pop("source_url", "https://example.com/course"),
        source_title=overrides.pop("source_title", "Example Course"),
        source_priority=overrides.pop("source_priority", "5"),
    )
    record.update(overrides)
    return record


# ============================================================
# Normalization
# ============================================================

def test_normalize_institute_name_variants_match():

    variants = [
        "Codeme Hub",
        "Codeme Hub (Tech Learning)",
        "CODEME HUB",
        "Codeme Hub – Calicut",
    ]

    canonicals = {normalize_institute_name(v)["canonical"] for v in variants}

    assert canonicals == {"codeme hub"}


def test_normalize_institute_name_preserves_display():

    result = normalize_institute_name("Codeme Hub (Tech Learning)")

    assert result["display"] == "Codeme Hub (Tech Learning)"
    assert result["canonical"] == "codeme hub"


def test_normalize_institute_name_not_available():

    result = normalize_institute_name(NOT_AVAILABLE)

    assert result["canonical"] == ""
    assert result["display"] == NOT_AVAILABLE


def test_normalize_course_name_level_variants_match():

    variants = ["Data Science", "Data Science (Beginners)", "DATA SCIENCE", "Data Science Course"]

    canonicals = {normalize_course_name(v)["canonical"] for v in variants}

    assert canonicals == {"data science"}


def test_normalize_course_name_advanced_stays_distinct():

    base = normalize_course_name("Data Science")["canonical"]
    advanced = normalize_course_name("Advanced Data Science")["canonical"]

    assert base != advanced


# ============================================================
# 1. Two identical institute/course records -> one result
# ============================================================

def test_identical_records_merge_into_one():

    a = make_record(institute_name="Codeme Hub", course_name="Data Science", duration="6 Months")
    b = make_record(institute_name="Codeme Hub", course_name="Data Science", duration="6 Months")

    result = deduplicate_courses([a, b])

    assert len(result["courses"]) == 1
    assert result["merge_report"]["original_count"] == 2
    assert result["merge_report"]["final_count"] == 1
    assert result["merge_report"]["merges_performed"] == 1


# ============================================================
# 2. Same institute, slightly different names -> one result
# ============================================================

def test_slightly_different_institute_names_merge():

    a = make_record(institute_name="Codeme Hub", course_name="Data Science")
    b = make_record(institute_name="Codeme Hub (Tech Learning)", course_name="Data Science")

    result = deduplicate_courses([a, b])

    assert len(result["courses"]) == 1


# ============================================================
# 3. Same course, different source URLs -> one result
# ============================================================

def test_same_course_different_urls_merge():

    a = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        source_url="https://codemehub.com/data-science"
    )
    b = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        source_url="https://some-directory.com/codeme-hub-ds"
    )

    result = deduplicate_courses([a, b])

    assert len(result["courses"]) == 1
    assert set(result["courses"][0]["source_urls"]) == {
        "https://codemehub.com/data-science",
        "https://some-directory.com/codeme-hub-ds",
    }


# ============================================================
# 4. Complementary fields are merged
# ============================================================

def test_complementary_fields_are_merged():

    a = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        duration="9 Months", fees=NOT_AVAILABLE,
        eligibility=NOT_AVAILABLE, learning_mode="Online & Offline",
        source_url="source-A"
    )
    b = make_record(
        institute_name="Codeme Hub (Tech Learning)", course_name="Data Science (Beginners)",
        duration=NOT_AVAILABLE, fees="₹45,000",
        eligibility="Any graduate", learning_mode=NOT_AVAILABLE,
        source_url="source-B"
    )

    result = deduplicate_courses([a, b])

    assert len(result["courses"]) == 1

    merged = result["courses"][0]

    assert merged["duration"] == "9 Months"
    assert merged["fees"] == "₹45,000"
    assert merged["eligibility"] == "Any graduate"
    assert merged["learning_mode"] == "Online & Offline"


# ============================================================
# 5. "Not available" must never overwrite real information
# ============================================================

def test_not_available_never_overwrites_real_value():

    a = make_record(institute_name="Codeme Hub", course_name="Data Science", fees="₹50,000")
    b = make_record(institute_name="Codeme Hub", course_name="Data Science", fees=NOT_AVAILABLE)

    result = deduplicate_courses([a, b])

    assert result["courses"][0]["fees"] == "₹50,000"

    # order shouldn't matter
    result_reversed = deduplicate_courses([b, a])

    assert result_reversed["courses"][0]["fees"] == "₹50,000"


# ============================================================
# 6. Duplicate source URLs are removed
# ============================================================

def test_duplicate_source_urls_are_removed():

    a = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        source_url="https://codemehub.com/ds"
    )
    b = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        source_url="https://codemehub.com/ds"
    )

    result = deduplicate_courses([a, b])

    assert result["courses"][0]["source_urls"] == ["https://codemehub.com/ds"]


# ============================================================
# 7. Different courses at the same institute remain separate
# ============================================================

def test_different_courses_same_institute_stay_separate():

    a = make_record(institute_name="ABC Institute", course_name="Data Science")
    b = make_record(institute_name="ABC Institute", course_name="Python Development")

    result = deduplicate_courses([a, b])

    assert len(result["courses"]) == 2


# ============================================================
# 8. Advanced Data Science must not merge with Data Science
# ============================================================

def test_advanced_variant_does_not_merge():

    a = make_record(institute_name="XYZ Academy", course_name="Data Science")
    b = make_record(institute_name="XYZ Academy", course_name="Advanced Data Science")

    result = deduplicate_courses([a, b])

    assert len(result["courses"]) == 2


# ============================================================
# 9. Conflicting duration values
# ============================================================

def test_conflicting_values_with_no_priority_signal_vary_by_source():

    a = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        duration="6 Months", source_priority="5", source_url="source-A"
    )
    b = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        duration="9 Months", source_priority="5", source_url="source-B"
    )

    result = deduplicate_courses([a, b])

    merged = result["courses"][0]

    assert merged["duration"] == "Varies by source"

    merge_detail = result["merge_report"]["merges"][0]
    assert "duration" in merge_detail["conflicts"]
    assert merge_detail["conflicts"]["duration"]["resolution"] == "unresolved_conflict"


# ============================================================
# 10. Source priority conflict resolution
# ============================================================

def test_conflicting_values_resolved_by_source_priority():

    a = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        duration="6 Months", source_priority="1", source_url="gov-source"
    )
    b = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        duration="9 Months", source_priority="5", source_url="institute-site"
    )

    result = deduplicate_courses([a, b])

    merged = result["courses"][0]

    assert merged["duration"] == "6 Months"

    merge_detail = result["merge_report"]["merges"][0]
    assert merge_detail["conflicts"]["duration"]["resolution"] == "source_priority"
    assert merge_detail["conflicts"]["duration"]["resolved_value"] == "6 Months"

    # the losing candidate must still be preserved, not discarded
    candidate_values = {
        c["value"] for c in merge_detail["conflicts"]["duration"]["candidates"]
    }
    assert candidate_values == {"6 Months", "9 Months"}


# ============================================================
# 11. Empty course list
# ============================================================

def test_empty_course_list():

    result = deduplicate_courses([])

    assert result["courses"] == []
    assert result["merge_report"]["original_count"] == 0
    assert result["merge_report"]["final_count"] == 0


def test_none_course_list():

    result = deduplicate_courses(None)

    assert result["courses"] == []


# ============================================================
# 12. Malformed course records
# ============================================================

def test_malformed_course_records_do_not_crash():

    result = deduplicate_courses([
        None,
        "not a record",
        {"institute_name": 123, "course_name": ["not", "a", "string"]},
        {},
    ])

    # never raises; every malformed item becomes a safe, mostly
    # "Not available" record via validate_course_record, and since
    # none of them have a real institute+course identity they can't
    # be merged with each other -- four separate empty-ish records.
    assert len(result["courses"]) == 4

    for course in result["courses"]:
        assert course["institute_name"] == NOT_AVAILABLE


# ============================================================
# 13. Multiple source URLs are preserved
# ============================================================

def test_multiple_source_urls_preserved_in_order():

    a = make_record(institute_name="Codeme Hub", course_name="Data Science", source_url="url-1")
    b = make_record(institute_name="Codeme Hub", course_name="Data Science", source_url="url-2")
    c = make_record(institute_name="Codeme Hub", course_name="Data Science", source_url="url-3")

    result = deduplicate_courses([a, b, c])

    merged = result["courses"][0]

    assert merged["source_urls"] == ["url-1", "url-2", "url-3"]
    assert len(merged["sources"]) == 3
