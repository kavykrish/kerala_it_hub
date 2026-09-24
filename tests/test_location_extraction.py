"""
Step 5B -- conservative location extraction/normalization tests.

Covers web_retrieval/course_extractor.py's location priority chain
(explicit Location/Venue/Address/Campus heading > JSON-LD structured
address > a Kerala place name next to a 6-digit PIN code anywhere on
the page -- never a bare city name in ordinary marketing prose), and
web_retrieval/kerala_locations.py's alias normalization
(Kochi/Cochin, Kozhikode/Calicut, ...). Pure unit/fixture tests -- no
live web search, no embedding model, no Groq call.
"""

from web_retrieval.chunker import section_chunk_text
from web_retrieval.text_cleaner import clean_text
from web_retrieval.course_extractor import NOT_AVAILABLE, extract_course_record
from web_retrieval.kerala_locations import canonical_location
from web_retrieval.query_intent import parse_query_intent
from web_retrieval.course_ranker import filter_and_rank_courses
from web_retrieval.course_extractor import empty_course_record


def _extract(page_text, **kwargs):

    cleaned = clean_text(page_text)
    chunks = section_chunk_text(cleaned, max_chunk_size=1000, overlap_lines=2)

    return extract_course_record(
        chunks,
        {"source_url": kwargs.pop("source_url", "https://example.com/course")},
        page_context=cleaned[:500],
        **kwargs
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
# Explicit heading extraction (Address/Campus/Location/Venue)
# ============================================================

def test_explicit_address_heading_with_pin_code():

    page = "Data Science\n\nAddress\nCodeme Hub, Ring Road, Kochi - 682001\n"

    record = _extract(page)

    assert record["location"] != NOT_AVAILABLE
    assert "kochi" in record["location"].lower()


def test_explicit_campus_heading_calicut():

    page = "Data Science\n\nCampus\nCalicut Campus\n"

    record = _extract(page)

    assert record["location"] != NOT_AVAILABLE
    assert "calicut" in record["location"].lower()


def test_explicit_location_heading_still_works():

    page = "Data Science\n\nLocation\nKochi, Kerala\n"

    record = _extract(page)

    assert record["location"] != NOT_AVAILABLE
    assert "kochi" in record["location"].lower()


# ============================================================
# Conservative behaviour -- no inference from marketing prose
# ============================================================

def test_bare_city_mention_in_marketing_text_is_not_available():

    page = (
        "Best Data Science Course in Kochi, Kerala with placement "
        "support and industry mentors.\n"
    )

    record = _extract(page)

    assert record["location"] == NOT_AVAILABLE


def test_bare_city_mention_in_h1_style_text_is_not_available():

    page = "Empowering careers with our Data Science course in Trivandrum.\n"

    record = _extract(page)

    assert record["location"] == NOT_AVAILABLE


# ============================================================
# Strong address pattern (PIN code, no heading required)
# ============================================================

def test_pin_code_adjacent_to_city_without_heading_is_extracted():

    page = (
        "Data Science\n\nAbout us\nWe are located near MG Road, "
        "Kochi 682016. Call us now for more details.\n"
    )

    record = _extract(page)

    assert record["location"] != NOT_AVAILABLE
    assert "kochi" in record["location"].lower()
    assert "682016" in record["location"]


def test_pin_code_before_city_name_is_extracted():

    page = "Data Science\n\nFooter\nOur office: 673001, Kozhikode, Kerala.\n"

    record = _extract(page)

    assert record["location"] != NOT_AVAILABLE
    assert "kozhikode" in record["location"].lower()


def test_no_pin_code_present_stays_not_available():

    page = "Data Science\n\nAbout us\nWe have trained thousands of students in Kochi.\n"

    record = _extract(page)

    assert record["location"] == NOT_AVAILABLE


# ============================================================
# JSON-LD structured address fallback
# ============================================================

def test_json_ld_location_used_when_no_heading():

    record = _extract(
        "Data Science\n\nSome unrelated marketing content.\n",
        json_ld_location="Kochi",
    )

    assert record["location"] != NOT_AVAILABLE
    assert "kochi" in record["location"].lower()


def test_json_ld_location_not_used_when_heading_already_found_something():

    page = "Data Science\n\nLocation\nCalicut\n"

    record = _extract(page, json_ld_location="Kochi")

    # Explicit heading wins over JSON-LD.
    assert "calicut" in record["location"].lower()


def test_json_ld_location_with_no_recognizable_place_stays_not_available():

    record = _extract(
        "Data Science\n\nSome unrelated marketing content.\n",
        json_ld_location="123 Main Street",
    )

    assert record["location"] == NOT_AVAILABLE


# ============================================================
# Codeme regression -- existing behaviour preserved
# ============================================================

def test_codeme_style_page_location_behaviour_preserved():
    """
    Codeme's real page (see Step 5 Part 1's regression fixture) has no
    Location/Address/Campus heading and no PIN code anywhere -- its
    location must still correctly stay "Not available" rather than
    guessing from the "Calicut, Kerala & UAE" mentioned in its own
    marketing headline.
    """

    page = (
        "Empowering Careers with the Best Data Science Course in "
        "Calicut, Kerala & UAECourse Duration : 9 MonthsTraining by "
        "Industrial Experts\n"
    )

    record = _extract(page, source_url="https://codemehub.com/x")

    assert record["duration"] != NOT_AVAILABLE  # Step 5 Part 1 fix intact
    assert record["location"] == NOT_AVAILABLE  # no heading, no PIN code


# ============================================================
# Alias normalization (Kochi/Cochin, etc.)
# ============================================================

def test_kochi_cochin_alias_match():

    assert canonical_location("Cochin") == canonical_location("Kochi")


def test_calicut_kozhikode_alias_match():

    assert canonical_location("Calicut") == canonical_location("Kozhikode")


def test_trivandrum_thiruvananthapuram_alias_match():

    assert canonical_location("Trivandrum") == canonical_location("Thiruvananthapuram")


def test_ernakulam_is_not_folded_into_kochi():

    assert canonical_location("Ernakulam") != canonical_location("Kochi")


def test_query_intent_normalizes_cochin_to_canonical_kochi():

    intent = parse_query_intent("data science courses in Cochin")

    assert intent["requested_location"] == canonical_location("Kochi")


# ============================================================
# End-to-end ranking -- "Data Science courses in Kochi"
# ============================================================

def test_kochi_query_ranks_cochin_labelled_record_first():
    """
    The exact real-world gap Step 5B reports: a course record whose
    own source page said "Cochin" must still rank above one with no
    location at all when the user asks for "Kochi".
    """

    cochin_labelled = make_record(
        institute_name="Alpha", course_name="Data Science", location="Cochin, Kerala"
    )
    unspecified = make_record(
        institute_name="Beta", course_name="Data Science"
    )
    different_city = make_record(
        institute_name="Gamma", course_name="Data Science", location="Trivandrum"
    )

    intent = parse_query_intent("Data Science courses in Kochi")

    ranked = filter_and_rank_courses(
        [different_city, unspecified, cochin_labelled],
        intent
    )

    assert ranked[0]["institute_name"] == "Alpha"
    assert ranked[1]["institute_name"] == "Beta"
    assert ranked[2]["institute_name"] == "Gamma"


def test_missing_location_never_hard_dropped():
    """
    Part 6's explicit requirement: a record with no structured
    location must remain eligible/present, never removed.
    """

    unspecified = make_record(institute_name="Beta", course_name="Data Science")

    ranked = filter_and_rank_courses(
        [unspecified],
        {"requested_location": "kochi"}
    )

    assert len(ranked) == 1
    assert ranked[0]["institute_name"] == "Beta"


def test_location_ranking_is_soft_even_for_named_institute_query():
    """
    Part 6: location matching stays a soft signal unless the query
    explicitly names an institute -- and even then, institute-name
    matching (Part 7, unchanged) takes priority over location, not the
    other way round.
    """

    codeme_wrong_location = make_record(
        institute_name="Codeme Hub", course_name="Data Science", location="Trivandrum"
    )
    other_right_location = make_record(
        institute_name="Luminar Technolab", course_name="Data Science", location="Kochi"
    )

    ranked = filter_and_rank_courses(
        [other_right_location, codeme_wrong_location],
        {"requested_location": "kochi"},
        query="tell me about the data science course at Codeme Hub",
    )

    # Named-institute match still wins over location match.
    assert ranked[0]["institute_name"] == "Codeme Hub"
