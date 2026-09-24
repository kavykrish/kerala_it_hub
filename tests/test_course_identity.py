"""
Step 3C -- course identity correctness tests.

The real-world validation after Step 3B found that course_name could
still come out wrong (a curriculum topic like "Artificial
Intelligence" instead of the page's actual subject "Data Science"),
and that institute_name could be misattributed on third-party
comparison/blog pages whose own domain looks innocuous. These tests
target that specific failure mode directly: a wrong value is strictly
worse than "Not available" (see course_extractor.py's module
docstring and this file's TEST D/E).

Synthetic fixtures only -- no live web search, no embedding model, no
Groq call.
"""

from web_retrieval.chunker import section_chunk_text
from web_retrieval.text_cleaner import clean_text
from web_retrieval.course_extractor import NOT_AVAILABLE, extract_course_record


def _extract(page_text, source_url="https://example.com/course", source_title=""):

    cleaned = clean_text(page_text)
    chunks = section_chunk_text(cleaned, max_chunk_size=1000, overlap_lines=2)

    return extract_course_record(
        chunks,
        {"source_url": source_url, "source_title": source_title},
        page_context=cleaned[:500]
    )


# ============================================================
# TEST A -- Data Science course with AI curriculum
# ============================================================

def test_a_curriculum_topic_does_not_override_primary_subject():

    page = """
Data Science

Learn Python, Machine Learning, Artificial Intelligence and Deep Learning.
"""

    record = _extract(page, source_title="Best Data Science Course in Kerala")

    assert record["source_title"] == "Best Data Science Course in Kerala"
    assert record["course_name"] == "Data Science"
    assert record["course_name"] != "Artificial Intelligence"


# ============================================================
# TEST B -- Explicit course title
# ============================================================

def test_b_explicit_course_name_heading():

    page = """
Course Name
Data Science
"""

    record = _extract(page)

    assert record["course_name"] == "Data Science"


# ============================================================
# TEST C -- Course title + curriculum
# ============================================================

def test_c_course_heading_plus_curriculum():

    page = """
Course:
Data Science

Curriculum:
Python
Machine Learning
Artificial Intelligence
"""

    record = _extract(page)

    assert record["course_name"] == "Data Science"


# ============================================================
# TEST D -- Marketing title only, nothing reliable elsewhere
# ============================================================

def test_d_title_alone_is_not_enough():

    page = "Contact us to learn more about our programs.\n"

    record = _extract(page, source_title="Best Data Science Course in Kerala")

    assert record["course_name"] == NOT_AVAILABLE


# ============================================================
# TEST E -- Third-party comparison page
# ============================================================

def test_e_listicle_title_suppresses_institute_name():

    page = """
Data Science

Alpha Institute offers a great program.
Beta Academy is also highly rated.
Gamma College provides excellent training.
"""

    record = _extract(
        page,
        source_url="https://someblog.com/roundup",
        source_title="10 Best Data Science Courses in Kerala"
    )

    assert record["institute_name"] == NOT_AVAILABLE


def test_e_multiple_institutes_in_content_suppresses_institute_name():

    page = """
Data Science

Alpha Institute offers a great program.
Beta Academy is also highly rated.
"""

    record = _extract(
        page,
        source_url="https://someblog.com/roundup",
        source_title="A roundup of good courses"
    )

    assert record["institute_name"] == NOT_AVAILABLE


def test_e_comparison_language_in_title_suppresses_institute_name():

    page = """
Data Science

Codeme Hub offers this program.
"""

    record = _extract(
        page,
        source_url="https://someblog.com/x",
        source_title="Comparison of the top data science institutes"
    )

    assert record["institute_name"] == NOT_AVAILABLE


# ============================================================
# TEST F -- Official institute page
# ============================================================

def test_f_official_institute_page_extracts_both_correctly():

    page = """
Codeme Hub

Data Science Course
"""

    record = _extract(page, source_url="https://codemehub.com/data-science")

    assert record["institute_name"] == "Codeme Hub"
    assert record["course_name"] == "Data Science"


# ============================================================
# TEST G -- Curriculum contains a stronger/longer keyword
# ============================================================

def test_g_longer_curriculum_keyword_does_not_win():

    page = """
Course:
Data Science

Curriculum:
Artificial Intelligence
Machine Learning
Deep Learning
Natural Language Processing
"""

    record = _extract(page)

    assert record["course_name"] == "Data Science"
    assert record["course_name"] != "Artificial Intelligence"
    assert record["course_name"] != "Natural Language Processing"


# ============================================================
# Additional: "No.1 Best ..." is a superlative claim about ONE
# institute, not a numbered listicle of several -- must not be
# suppressed the same way a genuine "10 Best ..." title is.
# ============================================================

def test_no_1_superlative_claim_is_not_treated_as_a_listicle():

    page = """
Data Science

Codeme Hub offers this program.
"""

    record = _extract(
        page,
        source_url="https://codemehub.com/x",
        source_title="No.1 Best Data Science Course in Calicut, Kerala and UAE"
    )

    assert record["institute_name"] == "Codeme Hub"


# ============================================================
# Additional: a single institute mention is NOT treated as
# a comparison page (make sure the fix isn't over-broad)
# ============================================================

def test_single_institute_mention_is_not_flagged_as_comparison():

    page = """
Data Science

Codeme Hub offers this program with dedicated mentors.

Duration
9 Months
"""

    record = _extract(page, source_url="https://codemehub.com/x")

    assert record["institute_name"] == "Codeme Hub"
