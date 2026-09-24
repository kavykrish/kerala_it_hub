"""
Step 3B -- real-web extraction quality tests.

These target the specific failure modes found by testing the real,
deployed pipeline against real search results: page/article titles
and SEO headlines being mistaken for institute_name/course_name, and
free-text fields (duration, fees, learning_mode, location) capturing
whole marketing paragraphs instead of the actual short value.

Synthetic fixtures only -- no live web search, no embedding model, no
Groq call.
"""

from web_retrieval.chunker import section_chunk_text
from web_retrieval.text_cleaner import clean_text
from web_retrieval.course_extractor import NOT_AVAILABLE, extract_course_record


def _extract(page_text, source_url="https://example.com/course", source_title=""):

    cleaned = clean_text(page_text)
    chunks = section_chunk_text(cleaned, max_chunk_size=1000, overlap_lines=2)

    # Mirrors mcp_server/course_search_server.py's Step 3/4 exactly:
    # page_context is a bounded prefix of the page's own cleaned text,
    # in original order -- distinct from the (possibly reordered)
    # chunks passed as retrieved_chunks.
    return extract_course_record(
        chunks,
        {"source_url": source_url, "source_title": source_title},
        page_context=cleaned[:500]
    )


# ============================================================
# Item 10 -- the four example fixtures from the spec
# ============================================================

def test_example_a_seo_title_with_clean_content():

    page = """
Data Science

Duration
6 Months

Eligibility
Any graduate
"""

    record = _extract(
        page,
        source_url="https://example-institute.com/course",
        source_title="Best Data Science Course in Kerala"
    )

    assert record["source_title"] == "Best Data Science Course in Kerala"
    assert record["course_name"] == "Data Science"
    assert record["duration"] == "6 Months"
    assert record["eligibility"] == "Any graduate"


def test_example_b_course_name_is_not_the_seo_headline():

    title = "No.1 Best Data Science Course in Calicut, Kerala and UAE"

    page = """
Data Science Course

Duration
9 Months

Mode
Online & Offline Live Training
"""

    record = _extract(page, source_url="https://codemehub.com/x", source_title=title)

    assert record["course_name"] != title
    assert "No.1" not in record["course_name"]
    assert "Best" not in record["course_name"]
    assert "UAE" not in record["course_name"]
    assert record["course_name"] == "Data Science"
    assert record["duration"] == "9 Months"


def test_example_c_vague_duration_text_is_not_available():

    page = """
Data Science

Duration
Flexible schedule based on offline & online classes
"""

    record = _extract(page)

    assert record["duration"] == NOT_AVAILABLE

    # The "offline & online" wording lives inside the DURATION section
    # only -- there's no separate Mode heading on this page at all, so
    # it must not leak into learning_mode either (see item 7: don't
    # let one field's chunk contaminate another field).
    assert record["learning_mode"] == NOT_AVAILABLE


def test_example_d_lms_access_is_not_live_online_training():

    page = """
Data Science

Mode
24x7 LMS Access
"""

    record = _extract(page)

    assert record["learning_mode"] == "LMS / Self-paced"
    assert record["learning_mode"] != "Online Live Training"
    assert "Live" not in record["learning_mode"]


# ============================================================
# 1. Page title is not automatically institute name
# ============================================================

def test_page_title_is_not_institute_name():

    record = _extract(
        "Data Science\n\nDuration\n6 Months\n",
        source_url="https://example.com/data-science",
        source_title="Scope, Skills & Jobs 2026"
    )

    assert record["institute_name"] != "Scope, Skills & Jobs 2026"


# ============================================================
# 2. Page title is not automatically course name
# ============================================================

def test_page_title_is_not_course_name():

    record = _extract(
        "Duration\n6 Months\n",
        source_title="Best Data Science Course in Kerala"
    )

    assert record["course_name"] != "Best Data Science Course in Kerala"


# ============================================================
# 3 & 4. SEO title / marketing headline vs actual course name
# ============================================================

def test_seo_headline_never_becomes_course_name_even_with_no_content_match():

    # No recognizable subject keyword anywhere in the body -- the
    # extractor must NOT fall back to the flashy title just to fill
    # the field.
    record = _extract(
        "Contact us for more information.\n",
        source_title="No.1 Best Training Institute in Kerala!!!"
    )

    assert record["course_name"] == NOT_AVAILABLE


def test_marketing_headline_with_real_subject_elsewhere_still_extracts_subject():

    page = """
No.1 Best Data Science Course in Calicut, Kerala and UAE

Curriculum
Python
Statistics
Data Science fundamentals
"""

    record = _extract(page)

    assert record["course_name"] == "Data Science"


# ============================================================
# 5 & 6. Duration pattern extraction / invalid duration text
# ============================================================

def test_duration_pattern_extraction_variants():

    for text, expected_contains in [
        ("Duration\n3 months\n", "3 months"),
        ("Duration\n6 Months\n", "6 Months"),
        ("Duration\n9 Months\n", "9 Months"),
        ("Duration\n1 year\n", "1 year"),
        ("Duration\n12 months\n", "12 months"),
        ("Duration\n24 weeks\n", "24 weeks"),
        ("Duration\n100 hours\n", "100 hours"),
    ]:

        record = _extract(f"Data Science\n\n{text}")

        assert record["duration"] == expected_contains, text


def test_invalid_duration_text_returns_not_available():

    for phrase in [
        "flexible schedule",
        "weekend classes",
        "flexible timings",
    ]:

        record = _extract(f"Data Science\n\nDuration\n{phrase}\n")

        assert record["duration"] == NOT_AVAILABLE, phrase


# ============================================================
# 7. Fee extraction
# ============================================================

def test_fee_extraction_variants():

    for text in [
        "Fees\n₹45,000\n",
        "Fees\n₹ 45,000\n",
        "Fees\nRs. 45,000\n",
        "Fees\nINR 45000\n",
    ]:

        record = _extract(f"Data Science\n\n{text}")

        assert record["fees"] != NOT_AVAILABLE, text
        assert "45" in record["fees"]


def test_unrelated_numbers_are_not_extracted_as_fees():

    record = _extract(
        "Data Science\n\nFees\nOver 3000 students have enrolled since 2015\n"
    )

    assert record["fees"] == NOT_AVAILABLE


# ============================================================
# 8. Eligibility extraction
# ============================================================

def test_eligibility_extraction_preserves_short_real_answer():

    record = _extract("Data Science\n\nEligibility\nAny graduate\n")

    assert record["eligibility"] == "Any graduate"


def test_eligibility_extraction_common_phrasings():

    for phrase in [
        "12th pass with 50% marks.",
        "Open to degree holders.",
        "No prior programming knowledge required.",
    ]:

        record = _extract(f"Data Science\n\nEligibility\n{phrase}\n")

        assert record["eligibility"] != NOT_AVAILABLE


# ============================================================
# 9. LMS vs live online distinction
# ============================================================

def test_lms_vs_live_online_distinction():

    lms_record = _extract("Data Science\n\nMode\nSelf-paced LMS access\n")
    live_record = _extract("Data Science\n\nMode\nLive online training sessions\n")

    assert lms_record["learning_mode"] == "LMS / Self-paced"
    assert live_record["learning_mode"] != "LMS / Self-paced"


# ============================================================
# 10. Location extraction
# ============================================================

def test_location_extraction_known_places():

    for place in ["Calicut", "Kochi", "Kozhikode", "Trivandrum", "Thiruvananthapuram"]:

        record = _extract(f"Data Science\n\nLocation\n{place}, Kerala\n")

        assert place in record["location"], place


def test_location_does_not_extract_arbitrary_marketing_place_names():

    record = _extract(
        "Data Science\n\nLocation\nJoin thousands of students worldwide\n"
    )

    assert record["location"] == NOT_AVAILABLE


# ============================================================
# 11. Long marketing paragraph should not become a field value
# ============================================================

def test_long_marketing_paragraph_does_not_become_certification_value():

    page = (
        "Data Science\n\nCertification\n"
        + "More than 3 lac companies use this platform to hire interns and "
        "employees and hence they will recognise a certificate from us when "
        "they see one on your resume. However please note that more than "
        "the certificate, companies hire candidates based on their skills "
        "and their performance in the interview so we hope you do a good "
        "job of learning the skill in the training and preparing well.\n"
    )

    record = _extract(page)

    assert record["certification"] != NOT_AVAILABLE
    assert len(record["certification"]) <= 200


# ============================================================
# 12. Missing field returns "Not available"
# ============================================================

def test_missing_fields_return_not_available():

    record = _extract("Data Science\n\nDuration\n6 Months\n")

    assert record["fees"] == NOT_AVAILABLE
    assert record["eligibility"] == NOT_AVAILABLE
    assert record["certification"] == NOT_AVAILABLE
    assert record["location"] == NOT_AVAILABLE
    assert record["learning_mode"] == NOT_AVAILABLE


# ============================================================
# 13. source_title remains separate
# ============================================================

def test_source_title_never_equals_institute_or_course_name():

    record = _extract(
        "Data Science\n\nDuration\n6 Months\n",
        source_url="https://codemehub.com/x",
        source_title="Best Data Science Course in Kerala – Scope, Skills & Jobs 2026"
    )

    assert record["source_title"] == "Best Data Science Course in Kerala – Scope, Skills & Jobs 2026"
    assert record["institute_name"] != record["source_title"]
    assert record["course_name"] != record["source_title"]


# ============================================================
# 14. Institute name remains separate from course name
# ============================================================

def test_institute_name_and_course_name_are_never_the_same_value():

    page = """
Data Science

Codeme Hub offers this program with dedicated mentors.

Duration
9 Months
"""

    record = _extract(page, source_url="https://codemehub.com/x")

    assert record["institute_name"] == "Codeme Hub"
    assert record["course_name"] == "Data Science"
    assert record["institute_name"] != record["course_name"]


# ============================================================
# 15. Existing Codeme deduplication still works
# ============================================================
# The full three-source merge scenario (this exact institute/course)
# is already covered end-to-end by
# tests/test_course_pipeline_integration.py, which passes unchanged
# after this step's changes -- see that file for the complete
# extraction -> dedup -> merge assertion.
