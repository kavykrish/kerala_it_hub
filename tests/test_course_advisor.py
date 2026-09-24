"""
Tests for the deterministic Course Advisor (Advisor Step 2,
web_retrieval/course_advisor.py).

Pure unit/fixture tests -- no live web search, no embedding model, no
Groq call. Course records are built directly (mirrors
tests/test_course_merger.py's own make_record helper).
"""

from web_retrieval.course_extractor import NOT_AVAILABLE, empty_course_record
from web_retrieval.course_ranker import build_rank_key, filter_and_rank_courses
from web_retrieval.course_advisor import (
    parse_advisor_preferences,
    is_advisor_query,
    _parse_budget_max,
    _parse_mode,
    _has_placement_preference,
    _budget_rank,
    _mode_rank,
    _placement_rank,
    _BUDGET_MATCH,
    _BUDGET_UNSPECIFIED,
    _BUDGET_CONFLICT,
    _MODE_MATCH,
    _MODE_UNSPECIFIED,
    _MODE_CONFLICT,
    _PLACEMENT_MATCH,
    _PLACEMENT_UNSPECIFIED,
    _PLACEMENT_CONFLICT,
    build_advisor_rank_key,
    advisor_rank_courses,
    explain_match,
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
# Advisor activation (1-9)
# ============================================================

def test_1_beginner_data_science_kochi_activates():

    assert is_advisor_query("I'm a beginner and I want to learn Data Science in Kochi") is True


def test_2_which_course_should_i_choose_activates():

    assert is_advisor_query("Which Data Science course should I choose as a beginner?") is True


def test_3_suggest_a_course_activates():

    assert is_advisor_query("Suggest a Data Science course for me") is True


def test_4_looking_for_course_activates():

    assert is_advisor_query("I'm looking for a Data Science course in Kochi") is True


def test_5_online_under_budget_activates():

    assert is_advisor_query("I want an online Data Science course under ₹50,000") is True


def test_6_plain_topic_location_query_does_not_activate():

    assert is_advisor_query("Data Science courses in Kochi") is False


def test_7_factual_institute_query_does_not_activate():

    assert is_advisor_query(
        "What is the duration of the Data Science course at Codeme Hub?"
    ) is False


def test_8_looking_for_with_no_signal_does_not_activate():

    assert is_advisor_query("I'm looking for information about Codeme Hub") is False


def test_9_vague_first_person_with_no_signal_does_not_activate():

    assert is_advisor_query("I want to say thank you") is False


# ============================================================
# Budget parsing (10-13)
# ============================================================

def test_10_rupee_symbol_amount():

    assert _parse_budget_max("under ₹50,000") == 50000


def test_11_k_suffix():

    assert _parse_budget_max("under 50k") == 50000


def test_12_lakh_suffix():

    assert _parse_budget_max("below 1 lakh") == 100000


def test_13_ambiguous_budget_returns_none():

    # No ceiling cue word ("under"/"below"/...) -- ambiguous, not a
    # guessed ceiling.
    assert _parse_budget_max("the course costs ₹50,000") is None


# ============================================================
# Mode parsing (14-16)
# ============================================================

def test_14_online_mode():

    assert _parse_mode("online classes") == "Online"


def test_15_classroom_maps_to_offline():

    assert _parse_mode("classroom training") == "Offline"


def test_16_hybrid_mode():

    assert _parse_mode("hybrid course") == "Hybrid"


# ============================================================
# Placement preference (17-19)
# ============================================================

def test_17_placement_assistance_phrase():

    assert _has_placement_preference("course with placement assistance") is True


def test_18_placement_support_phrase():

    assert _has_placement_preference("placement support") is True


def test_19_bare_job_does_not_activate_placement():

    assert _has_placement_preference("i want a job") is False


# ============================================================
# Ranking -- tri-state dimension ordering (20-22)
# ============================================================

def test_20_budget_match_beats_unspecified_beats_conflict():

    match = make_record(fees="₹40,000")
    unspecified = make_record(fees=NOT_AVAILABLE)
    conflict = make_record(fees="₹90,000")

    assert _budget_rank(match, 50000) == _BUDGET_MATCH
    assert _budget_rank(unspecified, 50000) == _BUDGET_UNSPECIFIED
    assert _budget_rank(conflict, 50000) == _BUDGET_CONFLICT
    assert _BUDGET_MATCH < _BUDGET_UNSPECIFIED < _BUDGET_CONFLICT


def test_21_mode_match_beats_unspecified_beats_conflict():

    match = make_record(learning_mode="Online")
    unspecified = make_record(learning_mode=NOT_AVAILABLE)
    conflict = make_record(learning_mode="Offline")

    assert _mode_rank(match, "Online") == _MODE_MATCH
    assert _mode_rank(unspecified, "Online") == _MODE_UNSPECIFIED
    assert _mode_rank(conflict, "Online") == _MODE_CONFLICT
    assert _MODE_MATCH < _MODE_UNSPECIFIED < _MODE_CONFLICT


def test_22_placement_match_beats_unspecified_beats_conflict():

    match = make_record(placement_information="Placement assistance is provided.")
    unspecified = make_record(placement_information=NOT_AVAILABLE)
    conflict = make_record(placement_information="No placement support is provided.")

    assert _placement_rank(match, True) == _PLACEMENT_MATCH
    assert _placement_rank(unspecified, True) == _PLACEMENT_UNSPECIFIED
    assert _placement_rank(conflict, True) == _PLACEMENT_CONFLICT
    assert _PLACEMENT_MATCH < _PLACEMENT_UNSPECIFIED < _PLACEMENT_CONFLICT


# ============================================================
# Ranking integration (23-28)
# ============================================================

def test_23_advisor_key_preserves_existing_five_dimensions():

    course = make_record(
        institute_name="Codeme Hub", course_name="Data Science", level="Beginner"
    )
    preferences = {
        "requested_topics": ["data science"],
        "requested_fields": [],
        "requested_level": "Beginner",
        "requested_location": None,
        "budget_max": 50000,
        "mode": "Online",
        "placement_preference": True,
    }

    base_key = build_rank_key(course, preferences, "")
    advisor_key = build_advisor_rank_key(course, preferences, "")

    assert advisor_key[:5] == base_key
    assert len(advisor_key) == 8


def test_24_advisor_off_matches_existing_step5b_order():

    ds = make_record(institute_name="Alpha", course_name="Data Science")
    ml = make_record(institute_name="Beta", course_name="Machine Learning")

    query_intent = {"requested_topics": ["data science"]}

    existing_order = filter_and_rank_courses([ml, ds], query_intent)

    assert [c["institute_name"] for c in existing_order] == ["Alpha", "Beta"]


def test_25_empty_advisor_signals_do_not_alter_base_ranking():

    ds = make_record(institute_name="Alpha", course_name="Data Science")
    ml = make_record(institute_name="Beta", course_name="Machine Learning")

    query_intent = {"requested_topics": ["data science"]}
    preferences = {**query_intent, "budget_max": None, "mode": None, "placement_preference": False}

    existing_order = filter_and_rank_courses([ml, ds], query_intent)
    advisor_order = advisor_rank_courses([ml, ds], preferences)

    assert (
        [c["institute_name"] for c in existing_order]
        == [c["institute_name"] for c in advisor_order]
    )


def test_26_missing_fees_does_not_cause_budget_mismatch():

    unspecified = make_record(institute_name="Alpha", fees=NOT_AVAILABLE)
    conflict = make_record(institute_name="Beta", fees="₹200,000")

    ranked = advisor_rank_courses(
        [conflict, unspecified],
        {"budget_max": 50000}
    )

    assert ranked[0]["institute_name"] == "Alpha"
    assert len(ranked) == 2


def test_27_missing_mode_does_not_cause_mode_mismatch():

    unspecified = make_record(institute_name="Alpha", learning_mode=NOT_AVAILABLE)
    conflict = make_record(institute_name="Beta", learning_mode="Offline")

    ranked = advisor_rank_courses(
        [conflict, unspecified],
        {"mode": "Online"}
    )

    assert ranked[0]["institute_name"] == "Alpha"
    assert len(ranked) == 2


def test_28_missing_placement_does_not_cause_placement_mismatch():

    unspecified = make_record(institute_name="Alpha", placement_information=NOT_AVAILABLE)
    conflict = make_record(
        institute_name="Beta",
        placement_information="No placement support is provided."
    )

    ranked = advisor_rank_courses(
        [conflict, unspecified],
        {"placement_preference": True}
    )

    assert ranked[0]["institute_name"] == "Alpha"
    assert len(ranked) == 2


def test_advisor_never_drops_a_record():

    a = make_record(institute_name="Alpha")
    b = make_record(institute_name="Beta")
    c = make_record(institute_name="Gamma")

    ranked = advisor_rank_courses(
        [a, b, c],
        {"requested_topics": ["data science"], "budget_max": 50000, "mode": "Online", "placement_preference": True}
    )

    assert len(ranked) == 3
    assert {r["institute_name"] for r in ranked} == {"Alpha", "Beta", "Gamma"}


# ============================================================
# Explanation (29-32)
# ============================================================

def test_29_matching_explanation_contains_positive_statements():

    course = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        level="Beginner", location="Kochi, Kerala",
    )
    preferences = {
        "requested_topics": ["data science"],
        "requested_level": "Beginner",
        "requested_location": "kochi",
    }

    explanation = explain_match(course, preferences)

    assert "✓" in explanation
    assert "Data Science" in explanation
    assert "Beginner" in explanation


def test_30_missing_fields_produce_neutral_statements():

    course = make_record(institute_name="Codeme Hub", course_name="Data Science")
    preferences = {"requested_topics": ["data science"], "budget_max": 50000}

    explanation = explain_match(course, preferences)

    assert "Fee information is not available." in explanation


def test_31_conflicts_produce_explicit_conflict_statements():

    course = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        learning_mode="Offline",
    )
    preferences = {"requested_topics": ["data science"], "mode": "Online"}

    explanation = explain_match(course, preferences)

    assert "✗" in explanation
    assert "Offline" in explanation
    assert "Online" in explanation


def test_32_explanation_never_invents_facts():
    """
    A dimension the user never asked about must not appear in the
    explanation at all -- nothing to claim, so nothing is said.
    """

    course = make_record(institute_name="Codeme Hub", course_name="Data Science")
    preferences = {"requested_topics": ["data science"]}

    explanation = explain_match(course, preferences)

    assert "budget" not in explanation.lower()
    assert "placement" not in explanation.lower()
    assert "location" not in explanation.lower()
    assert "mode" not in explanation.lower()


# ============================================================
# Preference schema shape
# ============================================================

def test_preferences_schema_is_superset_of_query_intent():

    preferences = parse_advisor_preferences(
        "I'm a beginner and I want to learn Data Science in Kochi. "
        "My budget is below ₹50,000 and I prefer online classes."
    )

    assert preferences["requested_topics"] == ["data science"]
    assert preferences["requested_level"] == "Beginner"
    assert preferences["requested_location"] == "kochi"
    assert preferences["budget_max"] == 50000
    assert preferences["mode"] == "Online"
    assert set(preferences.keys()) == {
        "requested_topics", "requested_fields", "requested_level",
        "requested_location", "comparison_intent",
        "budget_max", "mode", "placement_preference",
    }


def test_empty_query_returns_safe_neutral_preferences():

    preferences = parse_advisor_preferences("")

    assert preferences["requested_topics"] == []
    assert preferences["budget_max"] is None
    assert preferences["mode"] is None
    assert preferences["placement_preference"] is False
    assert is_advisor_query("") is False
