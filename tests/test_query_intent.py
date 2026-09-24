"""
Tests for deterministic query-intent parsing (Step 5 Part 2,
web_retrieval/query_intent.py).

Pure unit tests -- no network, no embedding model, no LLM call.
"""

from web_retrieval.query_intent import parse_query_intent


def test_beginner_topic_query():

    intent = parse_query_intent("Data science courses for beginners in Kerala")

    assert intent["requested_topics"] == ["data science"]
    assert intent["requested_level"] == "Beginner"
    assert intent["requested_fields"] == []
    assert intent["requested_location"] is None
    assert intent["comparison_intent"] is False


def test_requested_fields_query():

    intent = parse_query_intent(
        "what is the eligibility and duration for data science courses in Kerala?"
    )

    assert intent["requested_topics"] == ["data science"]
    assert set(intent["requested_fields"]) == {"eligibility", "duration"}


def test_specific_institute_query_has_no_field_restriction():

    intent = parse_query_intent("tell me about the data science course at Codeme Hub")

    assert intent["requested_topics"] == ["data science"]
    assert intent["requested_fields"] == []


def test_comparison_intent_detected():

    intent = parse_query_intent("compare data science courses")

    assert intent["comparison_intent"] is True


def test_advanced_level_detected():

    intent = parse_query_intent("advanced data science courses")

    assert intent["requested_level"] == "Advanced"


def test_location_detected():

    intent = parse_query_intent("data science courses in Kochi")

    assert intent["requested_location"] == "kochi"


def test_empty_query_returns_safe_empty_intent():

    intent = parse_query_intent("")

    assert intent == {
        "requested_topics": [],
        "requested_fields": [],
        "requested_level": None,
        "requested_location": None,
        "comparison_intent": False,
    }


def test_unrecognized_query_returns_safe_empty_intent():

    intent = parse_query_intent("random unrelated gibberish text")

    assert intent["requested_topics"] == []
    assert intent["requested_fields"] == []
    assert intent["requested_level"] is None
    assert intent["requested_location"] is None
    assert intent["comparison_intent"] is False


def test_course_name_and_category_labels_are_not_requested_fields():
    """
    "course name"/"category" identify WHICH course, not a detail of
    it -- must never appear in requested_fields.
    """

    intent = parse_query_intent("what is the course name and category for this program?")

    assert "course_name" not in intent["requested_fields"]
    assert "course_category" not in intent["requested_fields"]
