"""
Tests for the structured course extraction foundation
(web_retrieval/course_extractor.py).

These are pure unit/fixture tests -- no live web search, no network
calls, and no LLM calls (extraction is fully deterministic). Chunks are
produced by running realistic synthetic page text through the actual
chunker (web_retrieval.chunker.section_chunk_text), so these tests
exercise chunker + extractor together the same way the real pipeline
does, rather than hand-typing already-clean "chunks" that might not
resemble what the chunker really produces.
"""

from web_retrieval.chunker import section_chunk_text
from web_retrieval.text_cleaner import clean_text
from web_retrieval.course_extractor import (
    NOT_AVAILABLE,
    determine_source_priority,
    empty_course_record,
    extract_course_record,
    parse_llm_course_json,
    validate_course_record,
)


def _chunks_for(page_text):
    # Mirrors the real pipeline's order (mcp_server/course_search_server.py
    # Step 3): clean_text runs before chunking, which is also what
    # strips this test module's own source-indentation from the inline
    # triple-quoted fixtures below so the chunker's heading anchor
    # regex (which requires no leading whitespace) can match.
    cleaned = clean_text(page_text)
    return section_chunk_text(cleaned, max_chunk_size=1000, overlap_lines=2)


COMPLETE_PAGE = """
Data Science and AI Course

Duration
6 Months, weekdays and weekends available

Eligibility
Open to all graduates and diploma holders

Fees
Rs. 75,000 (installments available)

Mode of Training
Online and Offline (Hybrid)

Location
Kochi, Kerala

Certification
Certificate of completion issued jointly with an industry partner

Curriculum
Python Programming
Statistics for Data Science
Machine Learning
Deep Learning

Admission Process
Apply online, followed by a short screening interview

Placement Assistance
Dedicated placement support with resume building and mock interviews
"""

SOURCE_META = {
    "source_url": "https://www.example-institute.com/data-science-course",
    "source_title": "Data Science Course in Kochi | Example Institute",
}


# ============================================================
# 1. Complete course information
# ============================================================

def test_complete_course_information_fills_every_field():

    chunks = _chunks_for(COMPLETE_PAGE)

    record = extract_course_record(chunks, SOURCE_META)

    assert record["institute_name"] == "Example Institute"
    assert record["duration"] != NOT_AVAILABLE
    assert "6 Months" in record["duration"]
    assert record["eligibility"] != NOT_AVAILABLE
    assert record["fees"] != NOT_AVAILABLE
    assert "75,000" in record["fees"]
    assert record["learning_mode"] != NOT_AVAILABLE
    assert record["location"] != NOT_AVAILABLE
    assert "Kochi" in record["location"]
    assert record["certification"] != NOT_AVAILABLE
    assert record["admission_information"] != NOT_AVAILABLE
    assert record["placement_information"] != NOT_AVAILABLE
    assert isinstance(record["curriculum"], list)
    assert len(record["curriculum"]) >= 2
    assert record["source_url"] == SOURCE_META["source_url"]


# ============================================================
# 2. Course with only duration and mode
# ============================================================

def test_course_with_only_duration_and_mode():

    page = """
    Short Course Listing

    Duration
    3 Months

    Mode
    Online only
    """

    chunks = _chunks_for(page)

    record = extract_course_record(chunks, SOURCE_META)

    assert record["duration"] != NOT_AVAILABLE
    assert "3 Months" in record["duration"]
    assert record["learning_mode"] != NOT_AVAILABLE
    assert "Online" in record["learning_mode"]

    # everything else genuinely wasn't on the page -- must stay
    # "Not available", never guessed.
    assert record["fees"] == NOT_AVAILABLE
    assert record["eligibility"] == NOT_AVAILABLE
    assert record["certification"] == NOT_AVAILABLE
    assert record["curriculum"] == []


# ============================================================
# 3. Course with missing fees
# ============================================================

def test_course_with_missing_fees_only():

    page = """
    Cloud Computing Course

    Duration
    4 Months

    Eligibility
    Any graduate

    Mode
    Offline

    Location
    Trivandrum
    """

    chunks = _chunks_for(page)

    record = extract_course_record(chunks, SOURCE_META)

    assert record["fees"] == NOT_AVAILABLE
    assert record["duration"] != NOT_AVAILABLE
    assert record["eligibility"] != NOT_AVAILABLE
    assert record["learning_mode"] != NOT_AVAILABLE
    assert record["location"] != NOT_AVAILABLE


# ============================================================
# 4. Course with missing eligibility
# ============================================================

def test_course_with_missing_eligibility_only():

    page = """
    Cybersecurity Course

    Duration
    5 Months

    Fees
    Rs. 60,000

    Certification
    Industry-recognized certificate
    """

    chunks = _chunks_for(page)

    record = extract_course_record(chunks, SOURCE_META)

    assert record["eligibility"] == NOT_AVAILABLE
    assert record["duration"] != NOT_AVAILABLE
    assert record["fees"] != NOT_AVAILABLE
    assert record["certification"] != NOT_AVAILABLE


# ============================================================
# 5. Course with missing duration
# ============================================================

def test_course_with_missing_duration_only():

    page = """
    Full Stack Development Course

    Eligibility
    12th pass or graduate

    Fees
    Rs. 50,000

    Placement Assistance
    Placement support provided after course completion
    """

    chunks = _chunks_for(page)

    record = extract_course_record(chunks, SOURCE_META)

    assert record["duration"] == NOT_AVAILABLE
    assert record["eligibility"] != NOT_AVAILABLE
    assert record["fees"] != NOT_AVAILABLE
    assert record["placement_information"] != NOT_AVAILABLE


# ============================================================
# 6. Empty retrieved content
# ============================================================

def test_empty_retrieved_content_does_not_crash():

    record = extract_course_record([], SOURCE_META)

    assert record["source_url"] == SOURCE_META["source_url"]

    for field, value in record.items():
        if field in ("source_url", "source_title", "source_priority"):
            continue
        if field == "curriculum":
            assert value == []
        else:
            assert value == NOT_AVAILABLE


def test_none_retrieved_content_does_not_crash():

    record = extract_course_record(None, SOURCE_META)

    assert record["duration"] == NOT_AVAILABLE
    assert record["curriculum"] == []


def test_no_source_metadata_does_not_crash():

    record = extract_course_record(["Duration\n3 months"], None)

    assert record["source_url"] == ""
    assert record["source_priority"] == "unknown"


# ============================================================
# 7. Malformed extraction output
# ============================================================

def test_malformed_json_string_degrades_safely():

    record = parse_llm_course_json("this is not json at all", SOURCE_META)

    assert record["duration"] == NOT_AVAILABLE
    assert record["source_url"] == SOURCE_META["source_url"]


def test_json_array_instead_of_object_degrades_safely():

    record = parse_llm_course_json("[1, 2, 3]", SOURCE_META)

    assert record["fees"] == NOT_AVAILABLE
    assert record["curriculum"] == []


def test_wrong_typed_fields_degrade_safely():

    # fees given as a number, curriculum given as a string instead of
    # a list -- both invalid shapes for their field.
    raw_json = '{"fees": 50000, "curriculum": "Python, ML", "duration": "3 months"}'

    record = parse_llm_course_json(raw_json, SOURCE_META)

    assert record["fees"] == NOT_AVAILABLE
    assert record["curriculum"] == []
    assert record["duration"] == "3 months"


def test_validate_course_record_rejects_non_dict():

    assert validate_course_record(None) == empty_course_record()
    assert validate_course_record("not a dict") == empty_course_record()
    assert validate_course_record([1, 2, 3]) == empty_course_record()


def test_page_that_is_not_actually_a_course_page():

    page = "Contact us\nAbout our company\nHome\nBlog\n"

    chunks = _chunks_for(page)

    record = extract_course_record(chunks, SOURCE_META)

    assert record["duration"] == NOT_AVAILABLE
    assert record["fees"] == NOT_AVAILABLE
    assert record["curriculum"] == []


# ============================================================
# 8. Source URL preservation
# ============================================================

def test_source_url_is_preserved_exactly():

    chunks = _chunks_for(COMPLETE_PAGE)

    meta = {
        "source_url": "https://institute.example.com/course?utm_source=x",
        "source_title": "Some Title",
    }

    record = extract_course_record(chunks, meta)

    assert record["source_url"] == meta["source_url"]


def test_source_title_is_preserved_when_present():

    chunks = _chunks_for(COMPLETE_PAGE)

    record = extract_course_record(chunks, SOURCE_META)

    assert record["source_title"] == SOURCE_META["source_title"]


def test_source_title_defaults_to_not_available_when_missing():

    chunks = _chunks_for(COMPLETE_PAGE)

    record = extract_course_record(
        chunks,
        {"source_url": "https://example.com/course"}
    )

    assert record["source_title"] == NOT_AVAILABLE


# ============================================================
# 9. Source priority preservation
# ============================================================

def test_source_priority_explicit_value_is_preserved_not_recalculated():

    chunks = _chunks_for(COMPLETE_PAGE)

    record = extract_course_record(
        chunks,
        {
            "source_url": "https://some-random-blog.example.com/course",
            "source_priority": "2",
        }
    )

    # Even though the URL doesn't look like a university domain, an
    # explicitly-provided priority must win over the heuristic.
    assert record["source_priority"] == "2"


def test_source_priority_derived_when_not_given():

    chunks = _chunks_for(COMPLETE_PAGE)

    record = extract_course_record(
        chunks,
        {"source_url": "https://asapkerala.gov.in/course-page"}
    )

    assert record["source_priority"] == determine_source_priority(
        "https://asapkerala.gov.in/course-page"
    )


def test_determine_source_priority_heuristics():

    assert determine_source_priority("https://something.gov.in/x") == "1"
    assert determine_source_priority("https://asapkerala.gov.in/x") == "3"
    assert determine_source_priority("https://sbtekerala.gov.in/x") == "4"
    assert determine_source_priority("https://cusat.ac.in/x") == "2"
    assert determine_source_priority("https://www.coursera.org/x") == "6"
    assert determine_source_priority("https://some-institute.com/x") == "5"
    assert determine_source_priority("") == "unknown"
    assert determine_source_priority(None) == "unknown"
