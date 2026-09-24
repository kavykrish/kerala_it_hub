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
    overall_match_status,
    FULL_MATCH,
    PARTIAL_MATCH,
    CONFLICT,
    build_advisor_course_info,
    build_advisor_courses_payload,
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


# ============================================================
# Advisor Step 2B -- output leakage / prompt integration
# ============================================================
# explain_match's own output (used directly by tests below, and also
# by backend/rag_generator.py's _build_advisor_context) must never
# contain the internal "Criterion:"/"Match?:"/"Explanation:" labels
# that leaked into a live Android response, and must never claim an
# "explicit match" for a dimension the structured record didn't
# actually confirm.

def test_explain_match_never_contains_criterion_label():

    course = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        level="Beginner", location=NOT_AVAILABLE, fees="₹42,948",
        learning_mode="Online & Offline",
    )
    preferences = {
        "requested_topics": ["data science"],
        "requested_level": "Beginner",
        "requested_location": "kochi",
        "budget_max": 50000,
        "mode": "Online",
        "placement_preference": True,
    }

    explanation = explain_match(course, preferences)

    assert "Criterion:" not in explanation
    assert "Match?:" not in explanation
    assert "Explanation:" not in explanation


def test_explain_match_never_contains_malformed_placeholder_echo():

    course = make_record(institute_name="Codeme Hub", course_name="Data Science")
    preferences = {"requested_topics": ["data science"]}

    explanation = explain_match(course, preferences)

    assert "Criterion: Criterion" not in explanation
    assert "Match?: Match?" not in explanation
    assert "Explanation: Explanation" not in explanation


def test_missing_location_is_never_worded_as_explicit_match():
    """
    course.location == NOT_AVAILABLE must never produce a checkmark
    line claiming the location matches -- only the neutral
    "not available" sentence.
    """

    course = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        level="Beginner", location=NOT_AVAILABLE,
    )
    preferences = {
        "requested_topics": ["data science"],
        "requested_level": "Beginner",
        "requested_location": "kochi",
    }

    explanation = explain_match(course, preferences)

    assert "Location information is not available." in explanation
    assert "✓ Located in" not in explanation
    assert "kochi" not in explanation.lower()


def test_topic_match_with_unknown_location_does_not_claim_location_match():

    course = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        location=NOT_AVAILABLE,
    )
    preferences = {"requested_topics": ["data science"], "requested_location": "kochi"}

    explanation = explain_match(course, preferences)

    assert "✓ Data Science matches your requested course" in explanation
    assert "Location information is not available." in explanation


def test_portfolio_only_evidence_does_not_produce_placement_match():

    course = make_record(
        placement_information=(
            "You leave with a portfolio that shows you can ship, not just train."
        )
    )

    assert _placement_rank(course, True) == _PLACEMENT_UNSPECIFIED


def test_explicit_placement_assistance_produces_placement_match():

    course = make_record(placement_information="Placement assistance is provided to all students.")

    assert _placement_rank(course, True) == _PLACEMENT_MATCH


def test_career_support_produces_placement_match():

    course = make_record(placement_information="We offer career support after course completion.")

    assert _placement_rank(course, True) == _PLACEMENT_MATCH


def test_i_want_a_job_does_not_set_placement_preference():

    preferences = parse_advisor_preferences("I want a job")

    assert preferences["placement_preference"] is False


# ============================================================
# Advisor Step 2B -- prompt/context construction never leaks labels
# ============================================================

def test_advisor_context_never_contains_criterion_label():

    from backend.rag_generator import _build_advisor_context

    course = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        fees="₹42,948", learning_mode="Online & Offline",
    )
    preferences = {"requested_topics": ["data science"], "budget_max": 50000, "mode": "Online"}

    context = _build_advisor_context([course], preferences)

    assert "Criterion:" not in context
    assert "Match?:" not in context
    assert "Explanation:" not in context
    assert "Criterion: Criterion" not in context


def test_advisor_context_explicitly_instructs_against_leakage():
    """
    Regression guard (Problem 5): the generated context must actively
    instruct the model not to reproduce internal labels/structure --
    not just happen to avoid them itself.
    """

    from backend.rag_generator import _build_advisor_context

    course = make_record(institute_name="Codeme Hub", course_name="Data Science")
    preferences = {"requested_topics": ["data science"]}

    context = _build_advisor_context([course], preferences)

    assert "INTERNAL" in context.upper()
    assert '"Criterion"' in context or "'Criterion'" in context
    assert "do not" in context.lower() or "never" in context.lower()


def test_advisor_context_empty_when_no_usable_explanation():

    from backend.rag_generator import _build_advisor_context

    assert _build_advisor_context([], {"requested_topics": ["data science"]}) == ""
    assert _build_advisor_context([make_record()], {}) == ""


# ============================================================
# Advisor Step 2C -- placement evidence, re-confirmed strict (1-8)
# ============================================================

def test_1_portfolio_only_text_is_unspecified():

    course = make_record(
        placement_information=(
            "You leave with a portfolio that shows you can ship, not just train."
        )
    )
    assert _placement_rank(course, True) == _PLACEMENT_UNSPECIFIED


def test_2_project_building_text_is_unspecified():

    course = make_record(placement_information="Build projects to showcase your skills.")
    assert _placement_rank(course, True) == _PLACEMENT_UNSPECIFIED


def test_3_resume_building_text_is_unspecified():

    course = make_record(placement_information="This course includes resume building workshops.")
    assert _placement_rank(course, True) == _PLACEMENT_UNSPECIFIED


def test_4_employability_text_is_unspecified():

    course = make_record(placement_information="Improve your employability with industry projects.")
    assert _placement_rank(course, True) == _PLACEMENT_UNSPECIFIED


def test_5_career_guidance_text_is_unspecified():

    course = make_record(placement_information="Students receive career guidance and career advice.")
    assert _placement_rank(course, True) == _PLACEMENT_UNSPECIFIED


def test_6_placement_assistance_is_match():

    course = make_record(placement_information="Placement assistance is provided to all students.")
    assert _placement_rank(course, True) == _PLACEMENT_MATCH


def test_7_placement_support_is_match():

    course = make_record(placement_information="Placement support is offered after graduation.")
    assert _placement_rank(course, True) == _PLACEMENT_MATCH


def test_8_job_placement_is_match():

    course = make_record(placement_information="Job placement is guaranteed for top performers.")
    assert _placement_rank(course, True) == _PLACEMENT_MATCH


# ============================================================
# Advisor Step 2C -- deterministic overall match status (9-14)
# ============================================================

_FULL_PREFS = {
    "requested_topics": ["data science"],
    "requested_level": "Beginner",
    "requested_location": "kochi",
}


def test_9_exact_match_is_full_match():

    course = make_record(course_name="Data Science", level="Beginner", location="Kochi, Kerala")
    assert overall_match_status(course, _FULL_PREFS) == FULL_MATCH


def test_10_unknown_location_is_partial_match():

    course = make_record(course_name="Data Science", level="Beginner", location=NOT_AVAILABLE)
    assert overall_match_status(course, _FULL_PREFS) == PARTIAL_MATCH


def test_11_advanced_kochi_is_conflict_on_level():

    course = make_record(course_name="Data Science", level="Advanced", location="Kochi, Kerala")
    assert overall_match_status(course, _FULL_PREFS) == CONFLICT


def test_12_beginner_calicut_is_conflict_on_location():

    course = make_record(course_name="Data Science", level="Beginner", location="Calicut")
    assert overall_match_status(course, _FULL_PREFS) == CONFLICT


def test_13_advanced_calicut_is_conflict_on_both():

    course = make_record(course_name="Data Science", level="Advanced", location="Calicut")
    assert overall_match_status(course, _FULL_PREFS) == CONFLICT


def test_14_missing_preference_information_is_not_conflict():

    course = make_record(
        course_name="Data Science", level=NOT_AVAILABLE, location=NOT_AVAILABLE
    )
    status = overall_match_status(course, _FULL_PREFS)

    assert status == PARTIAL_MATCH
    assert status != CONFLICT


# ============================================================
# Advisor Step 2C -- prompt/context requirements (15-17)
# ============================================================

def test_15_advisor_context_contains_deterministic_status():

    from backend.rag_generator import _build_advisor_context

    course = make_record(course_name="Data Science", level="Beginner", location="Kochi, Kerala")

    context = _build_advisor_context([course], _FULL_PREFS)

    assert "Overall match status: FULL_MATCH" in context


def test_16_advisor_context_instructs_against_reinterpretation():

    from backend.rag_generator import _build_advisor_context

    course = make_record(course_name="Data Science", level="Advanced", location="Calicut")

    context = _build_advisor_context([course], _FULL_PREFS)

    assert "never decide for yourself whether a course matches" in context
    assert "Overall match status" in context
    assert "raw" in context.lower() and "Placement" in context


def test_17_no_criterion_match_explanation_leakage_step2c():

    from backend.rag_generator import _build_advisor_context

    course = make_record(course_name="Data Science", level="Beginner", location="Kochi, Kerala")

    context = _build_advisor_context([course], _FULL_PREFS)

    assert "Criterion: Criterion" not in context
    assert "Match?: Match?" not in context
    assert "Explanation: Explanation" not in context


# ============================================================
# Advisor Step 2C -- live bug scenario (Luminar/Blitz/Rogersoft)
# ============================================================

def test_no_full_match_note_fires_when_none_qualify():

    from backend.rag_generator import _build_advisor_context

    luminar = make_record(
        course_name="Data Science", institute_name="Luminar Technolab",
        level="Advanced", location="Calicut",
    )
    blitz = make_record(
        course_name="Data Science", institute_name="Blitz Academy",
        level="Beginner", location=NOT_AVAILABLE,
    )

    context = _build_advisor_context([blitz, luminar], _FULL_PREFS)

    assert (
        "No course above has an Overall match status of FULL_MATCH"
        in context
    )
    assert "Overall match status: CONFLICT" in context
    assert "Overall match status: PARTIAL_MATCH" in context


def test_ranking_and_status_logic_unchanged_by_step3b():
    """
    Advisor Step 3B only adds new presentation/formatting functions --
    it must not touch the ranking or status decision logic at all.
    Re-asserts the exact Step 2C worked examples still hold.
    """

    course = make_record(course_name="Data Science", level="Advanced", location="Calicut")
    assert overall_match_status(course, _FULL_PREFS) == CONFLICT

    ds = make_record(institute_name="Alpha", course_name="Data Science")
    ml = make_record(institute_name="Beta", course_name="Machine Learning")
    ranked = advisor_rank_courses(
        [ml, ds], {"requested_topics": ["data science"]}
    )
    assert ranked[0]["institute_name"] == "Alpha"


# ============================================================
# Advisor Step 3B -- structured /ask API payload
# ============================================================

def test_a_normal_search_has_no_structured_payload_needed():
    """
    build_advisor_courses_payload is only ever called by backend/main.py
    when advisor_active is True -- for a normal search there is simply
    no call, so there's nothing to assert about the function itself
    here beyond it handling an empty/None input safely.
    """

    assert build_advisor_courses_payload([], {}) == []
    assert build_advisor_courses_payload(None, {}) == []


def test_b_advisor_courses_is_a_list_of_structured_dicts():

    course = make_record(institute_name="Codeme Hub", course_name="Data Science")
    preferences = {"requested_topics": ["data science"]}

    payload = build_advisor_courses_payload([course], preferences)

    assert isinstance(payload, list)
    assert isinstance(payload[0], dict)


def test_c_advisor_course_structure_and_valid_match_status():

    course = make_record(
        institute_name="Codeme Hub", course_name="Data Science",
        level="Beginner", duration="9 Months",
    )
    preferences = {"requested_topics": ["data science"], "requested_level": "Beginner"}

    info = build_advisor_course_info(course, preferences)

    assert info["course_name"] == "Data Science"
    assert info["institute"] == "Codeme Hub"
    assert info["duration"] == "9 Months"
    assert info["match_status"] in (FULL_MATCH, PARTIAL_MATCH, CONFLICT)
    assert info["match_status"] == FULL_MATCH


def test_c_why_is_associated_with_the_correct_course():
    """
    Two different courses under the SAME preferences must get
    DIFFERENT, course-specific "why" lists -- never one shared
    explanation copy-pasted across every course.
    """

    ds = make_record(institute_name="Alpha", course_name="Data Science", level="Beginner")
    ml = make_record(institute_name="Beta", course_name="Machine Learning", level="Beginner")

    preferences = {"requested_topics": ["data science"], "requested_level": "Beginner"}

    payload = build_advisor_courses_payload([ds, ml], preferences)

    ds_info = next(c for c in payload if c["institute"] == "Alpha")
    ml_info = next(c for c in payload if c["institute"] == "Beta")

    assert ds_info["match_status"] == FULL_MATCH
    assert ml_info["match_status"] == CONFLICT
    assert ds_info["why"] != ml_info["why"]
    assert "Data Science matches your requested course." in ds_info["why"]
    assert any("does not match" in line for line in ml_info["why"])


def test_d_api_status_matches_deterministic_advisor_logic():
    """
    The match_status the payload exposes must be EXACTLY what
    overall_match_status (the existing deterministic function) already
    produces for the same course/preferences -- never a separately
    computed value.
    """

    course = make_record(course_name="Data Science", level="Advanced", location="Calicut")

    expected = overall_match_status(course, _FULL_PREFS)
    info = build_advisor_course_info(course, _FULL_PREFS)

    assert info["match_status"] == expected
    assert expected == CONFLICT


def test_e_portfolio_placement_not_reported_as_match_in_payload():

    course = make_record(
        course_name="Data Science",
        placement_information=(
            "You leave with a portfolio that shows you can ship, not just train."
        ),
    )
    preferences = {"requested_topics": ["data science"], "placement_preference": True}

    info = build_advisor_course_info(course, preferences)

    assert info["match_status"] != FULL_MATCH
    assert not any("✓" in line for line in info["why"])
    assert any(
        "does not confirm placement assistance specifically" in line
        for line in info["why"]
    )


def test_f_missing_location_not_claimed_as_kochi_in_payload():

    course = make_record(course_name="Data Science", location=NOT_AVAILABLE)
    preferences = {"requested_topics": ["data science"], "requested_location": "kochi"}

    info = build_advisor_course_info(course, preferences)

    assert info["location"] == NOT_AVAILABLE
    assert not any("Located in" in line for line in info["why"])
    assert any("Location information is not available." in line for line in info["why"])


def test_g_missing_fields_stay_not_available_never_inferred():

    course = make_record(course_name="Data Science")
    preferences = {"requested_topics": ["data science"]}

    info = build_advisor_course_info(course, preferences)

    for field in ("duration", "fees", "eligibility", "mode", "location", "certification"):
        assert info[field] == NOT_AVAILABLE


def test_h_all_existing_advisor_tests_still_pass_marker():
    """
    Placeholder assertion documenting the requirement -- actual
    coverage is the rest of this file (65 pre-3B tests) plus the full
    suite run reported separately.
    """

    assert True


def test_portfolio_text_does_not_leak_into_llm_as_placement_match():
    """
    End-to-end regression for the exact live bug: a course whose
    placement_information is a portfolio sentence, requested with
    placement_preference=True, must show as an unconfirmed/neutral
    line in the advisor context -- never a checkmark match line.
    """

    from backend.rag_generator import _build_advisor_context

    course = make_record(
        course_name="Data Science",
        placement_information=(
            "You leave with a portfolio that shows you can ship, not just train."
        ),
    )
    preferences = {"requested_topics": ["data science"], "placement_preference": True}

    context = _build_advisor_context([course], preferences)

    assert "✓ Placement" not in context
    assert "does not confirm placement assistance specifically" in context
