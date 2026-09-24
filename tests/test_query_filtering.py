"""
Tests for deterministic query-aware ranking (Step 5 Parts 3-7,
web_retrieval/course_ranker.py).

Pure unit/fixture tests -- no live web search, no embedding model, no
Groq call. Course records are built directly (mirrors
tests/test_course_merger.py's own make_record helper).
"""

from web_retrieval.course_extractor import NOT_AVAILABLE, empty_course_record
from web_retrieval.course_merger import deduplicate_courses
from web_retrieval.course_ranker import filter_and_rank_courses


def make_record(**overrides):
    record = empty_course_record(
        source_url=overrides.pop("source_url", "https://example.com/course"),
        source_title=overrides.pop("source_title", "Example Course"),
        source_priority=overrides.pop("source_priority", "5"),
    )
    record.update(overrides)
    return record


def names(ranked):
    return [r["course_name"] for r in ranked]


# ============================================================
# 1-4. Topic matching
# ============================================================

def test_data_science_exact_topic_match_ranks_first():

    ds = make_record(institute_name="Alpha", course_name="Data Science")
    unrelated = make_record(institute_name="Beta", course_name="Web Development")

    ranked = filter_and_rank_courses(
        [unrelated, ds],
        {"requested_topics": ["data science"]}
    )

    assert ranked[0]["course_name"] == "Data Science"


def test_data_science_and_machine_learning_compound_match():

    compound = make_record(institute_name="Alpha", course_name="Data Science & Machine Learning")
    unrelated = make_record(institute_name="Beta", course_name="Web Development")

    ranked = filter_and_rank_courses(
        [unrelated, compound],
        {"requested_topics": ["data science"]}
    )

    assert ranked[0]["course_name"] == "Data Science & Machine Learning"
    assert ranked[1]["course_name"] == "Web Development"


def test_pure_machine_learning_is_not_treated_as_data_science():

    ds = make_record(institute_name="Alpha", course_name="Data Science")
    ml = make_record(institute_name="Beta", course_name="Machine Learning")

    ranked = filter_and_rank_courses(
        [ml, ds],
        {"requested_topics": ["data science"]}
    )

    # The real Data Science course ranks strictly above the pure ML
    # one -- ML is never promoted to look like a Data Science match.
    assert ranked[0]["course_name"] == "Data Science"
    assert ranked[-1]["course_name"] == "Machine Learning"

    # Both records are still present -- nothing was dropped.
    assert len(ranked) == 2


def test_pure_ai_is_not_treated_as_data_science():

    ds = make_record(institute_name="Alpha", course_name="Data Science")
    ai = make_record(institute_name="Beta", course_name="Artificial Intelligence")

    ranked = filter_and_rank_courses(
        [ai, ds],
        {"requested_topics": ["data science"]}
    )

    assert ranked[0]["course_name"] == "Data Science"
    assert ranked[-1]["course_name"] == "Artificial Intelligence"
    assert len(ranked) == 2


def test_related_course_with_curriculum_evidence_ranks_above_unrelated():

    ml_with_evidence = make_record(
        institute_name="Beta",
        course_name="Machine Learning",
        curriculum=["Data Science fundamentals", "Model deployment"],
    )
    unrelated = make_record(institute_name="Gamma", course_name="Web Development")

    ranked = filter_and_rank_courses(
        [unrelated, ml_with_evidence],
        {"requested_topics": ["data science"]}
    )

    assert ranked[0]["course_name"] == "Machine Learning"
    assert ranked[1]["course_name"] == "Web Development"


# ============================================================
# 5-7. Level soft ranking
# ============================================================

def test_beginner_soft_ranking():

    beginner = make_record(institute_name="Alpha", course_name="Data Science", level="Beginner")
    advanced = make_record(institute_name="Beta", course_name="Data Science", level="Advanced")

    ranked = filter_and_rank_courses(
        [advanced, beginner],
        {"requested_level": "Beginner"}
    )

    assert ranked[0]["institute_name"] == "Alpha"


def test_advanced_soft_ranking():

    beginner = make_record(institute_name="Alpha", course_name="Data Science", level="Beginner")
    advanced = make_record(institute_name="Beta", course_name="Data Science", level="Advanced")

    ranked = filter_and_rank_courses(
        [beginner, advanced],
        {"requested_level": "Advanced"}
    )

    assert ranked[0]["institute_name"] == "Beta"


def test_missing_level_remains_eligible_and_not_penalized_like_conflict():

    unspecified = make_record(institute_name="Alpha", course_name="Data Science")
    conflicting = make_record(institute_name="Beta", course_name="Data Science", level="Advanced")

    ranked = filter_and_rank_courses(
        [conflicting, unspecified],
        {"requested_level": "Beginner"}
    )

    # Unspecified ranks above an explicit conflict, and both records
    # are still present -- missing level is never a reason to drop.
    assert ranked[0]["institute_name"] == "Alpha"
    assert len(ranked) == 2


# ============================================================
# 8-11. Requested-field-aware ranking
# ============================================================

def test_both_requested_fields_present_ranks_first():

    both = make_record(
        institute_name="Alpha", course_name="Data Science",
        eligibility="Any graduate", duration="6 months",
    )
    neither = make_record(institute_name="Beta", course_name="Data Science")

    ranked = filter_and_rank_courses(
        [neither, both],
        {"requested_fields": ["eligibility", "duration"]}
    )

    assert ranked[0]["institute_name"] == "Alpha"


def test_one_requested_field_present_ranks_between():

    both = make_record(
        institute_name="Alpha", course_name="Data Science",
        eligibility="Any graduate", duration="6 months",
    )
    one = make_record(
        institute_name="Beta", course_name="Data Science",
        eligibility="Any graduate",
    )
    neither = make_record(institute_name="Gamma", course_name="Data Science")

    ranked = filter_and_rank_courses(
        [neither, one, both],
        {"requested_fields": ["eligibility", "duration"]}
    )

    assert [r["institute_name"] for r in ranked] == ["Alpha", "Beta", "Gamma"]


def test_neither_requested_field_present_still_included():

    neither = make_record(institute_name="Gamma", course_name="Data Science")

    ranked = filter_and_rank_courses(
        [neither],
        {"requested_fields": ["eligibility", "duration"]}
    )

    # Never deleted for missing requested fields -- the answer can
    # still say "this course didn't provide those details".
    assert len(ranked) == 1
    assert ranked[0]["institute_name"] == "Gamma"


def test_unrelated_fields_present_but_requested_fields_absent_ranks_low():

    has_unrelated_fields = make_record(
        institute_name="Alpha", course_name="Data Science",
        certification="Certificate provided", placement_information="Placement support",
    )
    has_requested_field = make_record(
        institute_name="Beta", course_name="Data Science",
        duration="6 months",
    )

    ranked = filter_and_rank_courses(
        [has_unrelated_fields, has_requested_field],
        {"requested_fields": ["duration"]}
    )

    assert ranked[0]["institute_name"] == "Beta"


# ============================================================
# 12. Location-aware ranking
# ============================================================

def test_kochi_location_ranking():

    kochi = make_record(institute_name="Alpha", course_name="Data Science", location="Kochi, Kerala")
    trivandrum = make_record(institute_name="Beta", course_name="Data Science", location="Trivandrum, Kerala")
    unspecified = make_record(institute_name="Gamma", course_name="Data Science")

    ranked = filter_and_rank_courses(
        [trivandrum, unspecified, kochi],
        {"requested_location": "kochi"}
    )

    assert ranked[0]["institute_name"] == "Alpha"
    # Unspecified location is still eligible, ranked above an explicit
    # different location.
    assert ranked[1]["institute_name"] == "Gamma"
    assert ranked[2]["institute_name"] == "Beta"


# ============================================================
# 13. Specific institute/course query
# ============================================================

def test_codeme_hub_specific_query_ranks_named_institute_first():

    codeme = make_record(institute_name="Codeme Hub", course_name="Data Science")
    other = make_record(institute_name="Luminar Technolab", course_name="Data Science")

    ranked = filter_and_rank_courses(
        [other, codeme],
        {"requested_topics": ["data science"]},
        query="tell me about the data science course at Codeme Hub",
    )

    assert ranked[0]["institute_name"] == "Codeme Hub"


# ============================================================
# 14. Empty/unknown intent
# ============================================================

def test_empty_intent_does_not_crash_and_preserves_all_records():

    a = make_record(institute_name="Alpha", course_name="Data Science")
    b = make_record(institute_name="Beta", course_name="Web Development")

    ranked = filter_and_rank_courses([a, b], {})

    assert len(ranked) == 2
    assert {r["institute_name"] for r in ranked} == {"Alpha", "Beta"}


def test_none_intent_does_not_crash():

    a = make_record(institute_name="Alpha", course_name="Data Science")

    ranked = filter_and_rank_courses([a], None)

    assert len(ranked) == 1


def test_empty_course_records_returns_empty():

    assert filter_and_rank_courses([], {"requested_topics": ["data science"]}) == []


# ============================================================
# 15. No regression in existing deduplication
# ============================================================

def test_ranking_runs_cleanly_after_deduplication():

    page_a = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        duration="9 Months", source_url="https://codemehub.com/a",
    )
    page_b = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        eligibility="Any graduate", source_url="https://codemehub.com/b",
    )
    other = make_record(
        institute_name="Luminar Technolab", course_name="Machine Learning",
        source_url="https://luminar.com/x",
    )

    dedup_result = deduplicate_courses([page_a, page_b, other])

    assert dedup_result["merge_report"]["original_count"] == 3
    assert dedup_result["merge_report"]["final_count"] == 2

    ranked = filter_and_rank_courses(
        dedup_result["courses"],
        {"requested_topics": ["data science"]}
    )

    assert len(ranked) == 2
    assert ranked[0]["institute_name"] == "Codeme Hub"
    # The merged Codeme record still has both fields from its two
    # source pages -- ranking didn't disturb the merge.
    assert ranked[0]["duration"] == "9 Months"
    assert ranked[0]["eligibility"] == "Any graduate"
