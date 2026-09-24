"""
Step 4 -- field extraction hardening tests.

Targets the specific false-positive patterns found by testing the real,
deployed pipeline against real search results: an FAQ block ("Ques.
What are the eligibility requirements?") miscategorized as an
eligibility section and then truncated to the single word "Ques."; a
genuine "Certification" heading whose actual content is an unrelated
marketing/stats blob; course_category picking a longer but unrelated
keyword over the course's own resolved subject; "Online Classes"/
"Offline Classes" matching the organization-name pattern; and a
compound, suffix-less domain label ("stthomas.ac.in") being
capitalized into an unreadable guess ("Stthomas").

Synthetic fixtures only -- no live web search, no embedding model, no
Groq call.
"""

from web_retrieval.chunker import section_chunk_text
from web_retrieval.text_cleaner import clean_text
from web_retrieval.retriever import detect_field_category
from web_retrieval.course_extractor import (
    NOT_AVAILABLE,
    _extract_first_sentence,
    _institute_name_from_domain,
    extract_course_record,
)


def _extract(page_text, source_url="https://example.com/course", source_title="", page_h1=None):

    cleaned = clean_text(page_text)
    chunks = section_chunk_text(cleaned, max_chunk_size=1000, overlap_lines=2)

    return extract_course_record(
        chunks,
        {"source_url": source_url, "source_title": source_title},
        page_context=cleaned[:500],
        page_h1=page_h1,
    )


# ============================================================
# A. detect_field_category
# ============================================================

def test_a_genuine_heading_matches():

    assert detect_field_category("Eligibility\nAny graduate") == "eligibility"


def test_a_faq_block_does_not_match():

    assert detect_field_category(
        "Ques. What are the eligibility requirements for Data Science courses?"
    ) is None


def test_a_mid_sentence_mention_does_not_match():

    assert detect_field_category(
        "Students are eligible to apply if they hold a bachelor's degree."
    ) is None


# ============================================================
# B. abbreviation handling
# ============================================================

def test_b_ques_abbreviation_not_a_sentence_boundary():

    result = _extract_first_sentence(
        "Ques. What are the eligibility requirements for Data Science courses?"
    )

    assert result != "Ques."
    assert result == "Ques. What are the eligibility requirements for Data Science courses?"


def test_b_ans_abbreviation_not_a_sentence_boundary():

    result = _extract_first_sentence(
        "Ans. The minimum eligibility criteria include passing 10+2."
    )

    assert result != "Ans."
    assert "minimum eligibility criteria" in result


# ============================================================
# C. certification
# ============================================================

def test_c_stats_badge_blob_is_not_certification():

    page = """
Data Science

Certification
10k+
Membership
20+
Industry Projects
High-Growth Career With Industry-Focused Data Science Training
"""

    record = _extract(page)

    assert record["certification"] == NOT_AVAILABLE


def test_c_genuine_certification_is_extracted():

    page = """
Data Science

Certification
Industry-recognized certificate provided upon completion
"""

    record = _extract(page)

    assert record["certification"] != NOT_AVAILABLE
    assert "certificate" in record["certification"].lower()


# ============================================================
# D. admission
# ============================================================

def test_d_marketing_paragraph_is_not_admission_information():

    page = """
Data Science

Admission
Join our amazing program today and unlock your true potential!
"""

    record = _extract(page)

    assert record["admission_information"] == NOT_AVAILABLE


def test_d_genuine_admission_instructions_are_extracted():

    page = """
Data Science

Admission
Apply online and complete the admission process within 3 working days.
"""

    record = _extract(page)

    assert record["admission_information"] != NOT_AVAILABLE
    assert "apply" in record["admission_information"].lower()


# ============================================================
# E. placement
# ============================================================

def test_e_marketing_paragraph_is_not_placement_information():

    page = """
Data Science

Placement
Our trainers are the best in the industry with years of experience.
"""

    record = _extract(page)

    assert record["placement_information"] == NOT_AVAILABLE


def test_e_genuine_placement_assistance_is_extracted():

    page = """
Data Science

Placement
Placement assistance is provided to all students after course completion.
"""

    record = _extract(page)

    assert record["placement_information"] != NOT_AVAILABLE
    assert "placement" in record["placement_information"].lower()


# ============================================================
# F. course category
# ============================================================

def test_f_related_discipline_in_prose_does_not_become_category():

    page = """
Data Science

Data Science integrates Artificial Intelligence, statistics and mathematics.
"""

    record = _extract(page)

    assert record["course_name"] == "Data Science"
    assert record["course_category"] != "Artificial Intelligence"
    assert record["course_category"] == "Data Science"


def test_f_explicit_category_heading_is_preserved():

    page = """
Data Science

Category
Artificial Intelligence & Data Science
"""

    record = _extract(page)

    assert record["course_name"] == "Data Science"
    assert record["course_category"] == "Artificial Intelligence & Data Science"


# ============================================================
# G. institute
# ============================================================

def test_g_online_classes_never_becomes_institute():

    record = _extract("Data Science\n\nMode\nOnline Classes\n")

    assert record["institute_name"] != "Online Classes"


def test_g_offline_classes_never_becomes_institute():

    record = _extract("Data Science\n\nMode\nOffline Classes available on weekends\n")

    assert record["institute_name"] != "Offline Classes"


def test_g_weekend_classes_never_becomes_institute():

    record = _extract(
        "Data Science\n\nWe offer Weekend Classes for working professionals.\n"
    )

    assert record["institute_name"] != "Weekend Classes"


# ============================================================
# H. domain fallback
# ============================================================

def test_h_compound_suffixless_domain_is_not_available():

    assert _institute_name_from_domain("https://stthomas.ac.in/data-science-sf/") is None


def test_h_stthomas_full_pipeline_is_not_available():

    page = "Department of Data Science offers an undergraduate program.\n"

    record = _extract(page, source_url="https://stthomas.ac.in/data-science-sf/")

    assert record["institute_name"] == NOT_AVAILABLE
    assert record["institute_name"] != "Stthomas"


# ============================================================
# I. Step 5 -- Codeme Hub concatenated Duration heading regression
# ============================================================
# codemehub.com's real page text (confirmed via live fetch) has its H1
# fused directly onto the very next heading with no separating space
# at all: "...Data Science Course in Calicut, Kerala &
# UAECourse Duration : 9 MonthsTraining by Industrial Experts...".
# Step 4's start-of-line-only detect_field_category tightening made
# this chunk invisible to field-aware retrieval entirely -- Duration
# came back "Not available" for every query against this page, not
# because of anything query-specific. Step 5 Part 1 adds a
# boundary-aware fallback: the label as a whole word anywhere in the
# first line, immediately followed by a genuine separator.

CODEME_CONCATENATED_TEXT = (
    "Empowering Careers with the Best Data Science Course in Calicut, "
    "Kerala & UAECourse Duration : 9 MonthsTraining by Industrial "
    "Experts24x7 LMS AccessFlexible TimingsOnline & Offline Live "
    "Training"
)


def test_i_concatenated_duration_heading_is_detected():

    assert detect_field_category(CODEME_CONCATENATED_TEXT) == "duration"


def test_i_concatenated_duration_survives_extraction():

    record = _extract(
        CODEME_CONCATENATED_TEXT,
        source_url="https://codemehub.com/data-science-course-in-calicut-kerala-and-uae/",
    )

    assert record["duration"] != NOT_AVAILABLE
    assert "9" in record["duration"]
    assert "month" in record["duration"].lower()


def test_i_faq_eligibility_false_positive_still_rejected():
    """
    Existing Step 4 protection, re-asserted here alongside the new
    boundary-aware rule: a field word with no separator immediately
    after it is still rejected, even mid-line.
    """

    assert detect_field_category(
        "Ques. What are the eligibility requirements for Data Science courses?"
    ) is None


def test_i_ml_models_false_positive_still_rejected():
    """
    Existing Step 4 protection: "MODELS" contains the substring "MODE"
    but is not a whole word match for the "mode" label, so it's still
    rejected even though the boundary-aware rule now looks beyond the
    start of the line.
    """

    assert detect_field_category(
        "The course covers deployment of ML MODELS in production."
    ) is None
