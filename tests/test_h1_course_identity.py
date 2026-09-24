"""
Step 3E -- structured H1/JSON-LD course identity tests.

The Step 3D forensic investigation found that two real pages
(collegedunia.com, blitzacademy.org) both have an unambiguous <h1>
naming their actual subject ("Data Science"), but the pipeline
discarded all HTML structure before extraction ever ran -- so
course_name fell through to a keyword match against ordinary prose
that happened to mention "Artificial Intelligence" as a related field.
These tests exercise extract_course_record's new page_h1/
json_ld_identity parameters directly (course_extractor.py doesn't do
any fetching itself, so no network/page_reader involvement is needed
here -- see tests/test_course_identity.py's _extract helper for the
page_context-only path these build on).
"""

from web_retrieval.chunker import section_chunk_text
from web_retrieval.text_cleaner import clean_text
from web_retrieval.course_extractor import NOT_AVAILABLE, extract_course_record


def _extract(
    page_text,
    source_url="https://example.com/course",
    source_title="",
    page_h1=None,
    page_h2=None,
    json_ld_identity=None
):

    cleaned = clean_text(page_text)
    chunks = section_chunk_text(cleaned, max_chunk_size=1000, overlap_lines=2)

    return extract_course_record(
        chunks,
        {"source_url": source_url, "source_title": source_title},
        page_context=cleaned[:500],
        page_h1=page_h1,
        page_h2=page_h2,
        json_ld_identity=json_ld_identity
    )


# ============================================================
# TEST A -- CollegeDunia-style H1
# ============================================================

def test_a_collegedunia_style_h1():

    page = """
Kerala, with its high literacy rate and tech-ready population, is open
to adopting new technologies like machine learning, cloud computing,
and artificial intelligence. Top institutions in Kerala offering Data
Science courses include IIT Palakkad and others.
"""

    record = _extract(
        page,
        source_url="https://collegedunia.com/courses/data-science/x",
        source_title="Data Science Courses in Kerala: Top College, Top Institutes, Fees ...",
        page_h1="Data Science Courses in Kerala: College, Top Institutes, Fees, Salary, Placements 2026"
    )

    assert record["course_name"] == "Data Science"
    assert record["course_name"] != "Artificial Intelligence"


# ============================================================
# TEST B -- Blitz-style H1
# ============================================================

def test_b_blitz_style_h1():

    page = """
Blitz Academy, Kerala's leading software training institute, provides
top-notch data science training. It integrates concepts from computer
engineering, mathematics, artificial intelligence, and statistics.
"""

    record = _extract(
        page,
        source_url="https://blitzacademy.org/coursedetail.php",
        source_title="Data Science Course in Kerala | Kochi",
        page_h1="Data Science with machine learning"
    )

    assert record["course_name"] == "Data Science with Machine Learning"
    assert record["course_name"] != "Artificial Intelligence"


# ============================================================
# TEST C -- H1 + AI prose
# ============================================================

def test_c_h1_beats_ai_mentioned_in_prose():

    page = """
Data Science integrates artificial intelligence and statistics to
uncover meaningful insights.
"""

    record = _extract(page, page_h1="Data Science")

    assert record["course_name"] == "Data Science"


# ============================================================
# TEST D -- H1 + AI curriculum
# ============================================================

def test_d_h1_beats_ai_in_curriculum_list():

    page = """
Curriculum
Artificial Intelligence
Deep Learning
NLP
"""

    record = _extract(page, page_h1="Data Science")

    assert record["course_name"] == "Data Science"


# ============================================================
# TEST E -- Marketing H1 only, nothing reliable elsewhere
# ============================================================

def test_e_marketing_h1_alone_is_not_available():

    page = "Contact us to learn more about our programs.\n"

    record = _extract(page, page_h1="Best Data Science Course in Kerala")

    assert record["course_name"] == NOT_AVAILABLE


# ============================================================
# TEST F -- Explicit Course Name overrides H1
# ============================================================

def test_f_explicit_course_name_overrides_h1():

    page = """
Course Name
Data Analytics
"""

    record = _extract(page, page_h1="Best Data Science Course in Kerala")

    assert record["course_name"] == "Data Analytics"


# ============================================================
# TEST G -- No H1, existing page_context fallback still works
# ============================================================

def test_g_no_h1_falls_back_to_page_context():

    page = """
Data Science

Duration
6 Months
"""

    record = _extract(page, page_h1=None)

    assert record["course_name"] == "Data Science"


# ============================================================
# TEST H -- H1 must not become institute_name automatically
# ============================================================

def test_h_h1_does_not_leak_into_institute_name():

    page = "Learn data science with hands-on projects.\n"

    record = _extract(
        page,
        source_url="https://example-institute.com/x",
        page_h1="Data Science Courses in Kerala: Top Institutes, Fees ..."
    )

    assert record["institute_name"] != record["course_name"]
    assert "Top Institutes" not in record["institute_name"]


# ============================================================
# TEST I -- Aggregator H1: course_name identified, institute suppressed
# ============================================================

def test_i_aggregator_h1_identifies_course_but_not_institute():

    page = """
Data Science

Alpha Institute offers a great program.
Beta Academy is also highly rated.
"""

    record = _extract(
        page,
        source_url="https://someblog.com/roundup",
        source_title="10 Best Data Science Courses in Kerala",
        page_h1="Data Science Courses in Kerala: Top Institutes, Fees ..."
    )

    assert record["course_name"] == "Data Science"
    assert record["institute_name"] == NOT_AVAILABLE


# ============================================================
# TEST J -- Existing Codeme regression
# ============================================================

def test_j_codeme_regression():

    page = """
Data Science

Codeme Hub offers this program with dedicated mentors.

Duration
9 Months
"""

    # Codeme's real page has no <h1> at all (confirmed directly against
    # the live page) -- this must still resolve correctly via the
    # existing page_context fallback, unaffected by page_h1 being None.
    record = _extract(
        page,
        source_url="https://codemehub.com/x",
        page_h1=None
    )

    assert record["institute_name"] == "Codeme Hub"
    assert record["course_name"] == "Data Science"


# ============================================================
# TEST K -- Official institute page with a clean H1 (Rogersoft-style)
# ============================================================

def test_k_official_institute_page_with_h1():

    page = """
Rogersoft provides hands-on data science training with real projects.
"""

    record = _extract(
        page,
        source_url="https://www.rogersoft.com/course/data-science-training",
        source_title="Best Data Science Course Training Institute in Kochi, Kerala",
        page_h1="Data Science Training"
    )

    assert record["course_name"] == "Data Science"
    assert record["institute_name"] == "Rogersoft"


# ============================================================
# JSON-LD identity path (parallels the H1 path, lower priority)
# ============================================================

def test_json_ld_identity_used_when_no_h1():

    page = "Contact us for more information.\n"

    record = _extract(
        page,
        page_h1=None,
        json_ld_identity="Data Science Courses In Kerala"
    )

    assert record["course_name"] == "Data Science"


def test_h1_takes_priority_over_json_ld():

    page = "Contact us for more information.\n"

    record = _extract(
        page,
        page_h1="Data Science with machine learning",
        json_ld_identity="Web Development Courses In Kerala"
    )

    assert record["course_name"] == "Data Science with Machine Learning"


# ============================================================
# Real-world regression: an FAQ-style H1 ("What is the best ...?")
# keeps the whole question as a prefix unless the hype-word check
# looks past just the heading's first word -- confirmed against
# synnefo.academy's actual <h1> during Step 3E real-query validation.
# ============================================================

def test_faq_style_h1_is_rejected():

    page = "Some blog content about various courses.\n"

    record = _extract(
        page,
        source_url="https://synnefo.academy/blog/x",
        page_h1="What is the best Data Science Course in Kerala for beginners in 2026?"
    )

    assert record["course_name"] == NOT_AVAILABLE
    assert record["course_name"] != "What is the best Data Science"


# ============================================================
# Real-world regression: a call-to-action H1 ("Become a Data Science
# Pro with...") must not keep "Become a" as if it were a legitimate
# qualifier like "Advanced" -- confirmed against futurixacademy.com's
# actual <h1> during Step 3E real-query validation.
# ============================================================

def test_cta_verb_h1_prefix_is_rejected():

    page = "Join Kerala's best data science course and get the most out of it.\n"

    record = _extract(
        page,
        source_url="https://futurixacademy.com/",
        page_h1="Become a Data Science Pro with Futurix's premier Data Science Course in Kerala."
    )

    assert record["course_name"] != "Become a Data Science"


# ============================================================
# Real-world regression: a page with no <h1> at all falls back to its
# first <h2> -- confirmed against rogersoft.com, which has zero <h1>
# tags but states its real course identity, "Data Science & Machine
# Learning with AI (Artificial Intelligence)", as its first <h2>.
# ============================================================

def test_h2_fallback_used_when_no_h1():

    page = "AI (Artificial Intelligence) & Data science is an interdisciplinary field.\n"

    record = _extract(
        page,
        source_url="https://www.rogersoft.com/course/data-science-training",
        page_h1=None,
        page_h2="Data Science & Machine Learning with AI (Artificial Intelligence)"
    )

    assert record["course_name"] == "Data Science & Machine Learning"
    assert record["course_name"] != "Artificial Intelligence"


def test_h2_not_used_when_h1_present():

    page = "Contact us for more information.\n"

    record = _extract(
        page,
        page_h1="Data Science",
        page_h2="Some Other Heading Mentioning Machine Learning"
    )

    assert record["course_name"] == "Data Science"


# ============================================================
# Real-world regression: a student testimonial ("...Thankyou Blitz
# Academy.") must not be miscounted as a SECOND, different
# organization from the page's other "Blitz Academy" mentions --
# confirmed against blitzacademy.org's actual page content, where this
# previously made a genuine single-institute page get wrongly flagged
# as a multi-institute comparison page (institute_name suppressed).
# ============================================================

def test_testimonial_mention_does_not_create_false_second_institute():

    page = """
Data Science

Blitz Academy provides top-notch data science training.

I am a student. Got a great job. Thankyou Blitz Academy.
"""

    record = _extract(page, source_url="https://blitzacademy.org/x")

    assert record["institute_name"] == "Blitz Academy"
