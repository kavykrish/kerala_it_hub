"""
Tests for Step 3's wiring into backend/rag_generator.py: the new
_build_structured_context helper, and generate_rag_answer's fallback
behaviour when course_records is empty/None.

The Groq call itself is mocked (via call_groq_with_retry) -- these
tests must never make a real API call, both to stay fast/offline and
to avoid spending the project's own Groq quota (the earlier 500-error
incident was specifically caused by unnecessary local calls against
the shared account).
"""

from unittest.mock import patch, MagicMock

from web_retrieval.course_extractor import NOT_AVAILABLE, empty_course_record
from backend.rag_generator import _build_structured_context, generate_rag_answer


def make_course(**overrides):
    course = empty_course_record(
        source_url=overrides.pop("source_url", "https://example.com/course"),
        source_title=overrides.pop("source_title", "Example Course"),
        source_priority=overrides.pop("source_priority", "5"),
    )
    course.update(overrides)
    course.setdefault("source_urls", [course["source_url"]])
    course.setdefault("sources", [{
        "source_url": course["source_url"],
        "source_title": course["source_title"],
        "source_priority": course["source_priority"],
    }])
    return course


def _mock_groq_response(text):

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


# ============================================================
# _build_structured_context
# ============================================================

def test_build_structured_context_includes_known_fields():

    course = make_course(
        institute_name="Codeme Hub",
        course_name="Data Science",
        duration="9 Months",
        fees="Rs. 45,000",
    )

    context, sources = _build_structured_context([course], char_budget=15000)

    assert "COURSE 1" in context
    assert "Institute: Codeme Hub" in context
    assert "Course: Data Science" in context
    assert "Duration: 9 Months" in context
    assert "Fees: Rs. 45,000" in context
    assert sources == [{"title": "Example Course", "url": "https://example.com/course"}]


def test_build_structured_context_skips_empty_records():

    empty = make_course(institute_name=NOT_AVAILABLE, course_name=NOT_AVAILABLE)

    context, sources = _build_structured_context([empty], char_budget=15000)

    assert context == ""
    assert sources == []


def test_build_structured_context_respects_char_budget():

    courses = [
        make_course(
            institute_name=f"Institute {i}",
            course_name="Data Science",
            source_url=f"https://example.com/{i}"
        )
        for i in range(50)
    ]

    context, _sources = _build_structured_context(courses, char_budget=500)

    assert len(context) < 2000  # well below what 50 full blocks would produce


# ============================================================
# generate_rag_answer -- structured path vs. fallback
# ============================================================

@patch("backend.rag_generator.call_groq_with_retry")
def test_generate_rag_answer_uses_structured_context_when_available(mock_groq):

    mock_groq.return_value = _mock_groq_response("The answer.\n\nSources\n1. Example - https://example.com/course")

    course = make_course(
        institute_name="Codeme Hub",
        course_name="Data Science",
        duration="9 Months",
    )

    result = generate_rag_answer(
        query="data science courses",
        retrieved_results=[{"content": "irrelevant raw chunk", "source_url": "x", "source_title": "x"}],
        history=[],
        course_records=[course],
    )

    assert result["answer"] == "The answer.\n\nSources\n1. Example - https://example.com/course"

    # the prompt actually sent to Groq should be the clean COURSE
    # block, not the raw chunk text.
    sent_messages = mock_groq.call_args.kwargs["messages"]
    user_message = sent_messages[1]["content"]

    assert "COURSE 1" in user_message
    assert "Codeme Hub" in user_message
    assert "irrelevant raw chunk" not in user_message


@patch("backend.rag_generator.call_groq_with_retry")
def test_generate_rag_answer_falls_back_when_course_records_empty(mock_groq):

    mock_groq.return_value = _mock_groq_response("Fallback answer.")

    result = generate_rag_answer(
        query="data science courses",
        retrieved_results=[{
            "content": "Duration\n6 Months",
            "source_url": "https://example.com/x",
            "source_title": "Example"
        }],
        history=[],
        course_records=[],
    )

    assert result["answer"] == "Fallback answer."

    sent_messages = mock_groq.call_args.kwargs["messages"]
    user_message = sent_messages[1]["content"]

    # falls back to the original raw-chunk SOURCE block format
    assert "SOURCE 1" in user_message
    assert "Duration" in user_message


@patch("backend.rag_generator.call_groq_with_retry")
def test_generate_rag_answer_falls_back_when_course_records_none(mock_groq):

    mock_groq.return_value = _mock_groq_response("Fallback answer.")

    result = generate_rag_answer(
        query="data science courses",
        retrieved_results=[{
            "content": "Duration\n6 Months",
            "source_url": "https://example.com/x",
            "source_title": "Example"
        }],
        history=[],
    )

    assert result["answer"] == "Fallback answer."


@patch("backend.rag_generator.call_groq_with_retry")
def test_generate_rag_answer_falls_back_when_all_course_records_are_empty(mock_groq):

    mock_groq.return_value = _mock_groq_response("Fallback answer.")

    empty_course = make_course(institute_name=NOT_AVAILABLE, course_name=NOT_AVAILABLE)

    result = generate_rag_answer(
        query="data science courses",
        retrieved_results=[{
            "content": "Duration\n6 Months",
            "source_url": "https://example.com/x",
            "source_title": "Example"
        }],
        history=[],
        course_records=[empty_course],
    )

    sent_messages = mock_groq.call_args.kwargs["messages"]
    user_message = sent_messages[1]["content"]

    assert "SOURCE 1" in user_message
