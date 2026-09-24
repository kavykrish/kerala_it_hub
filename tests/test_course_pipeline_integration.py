"""
Full-pipeline integration test: multiple synthetic source pages ->
chunking -> structured extraction -> deduplication/merging -> final
course records.

This is also the exact "IMPORTANT SAFETY CHECK" example from the Step
3 spec: three source pages for "Codeme Hub" / "Data Science" that each
only have PART of the course's information, which must end up as ONE
merged course record with every field filled in from whichever source
had it, and all three source URLs preserved.

No live web search, no embedding model, no Groq call -- pure
chunker + extractor + merger, exactly like tests/test_course_extractor.py.
"""

from web_retrieval.chunker import section_chunk_text
from web_retrieval.text_cleaner import clean_text
from web_retrieval.course_extractor import NOT_AVAILABLE, extract_course_record
from web_retrieval.course_merger import deduplicate_courses


def _extract(page_text, source_url, source_title):

    cleaned = clean_text(page_text)
    chunks = section_chunk_text(cleaned, max_chunk_size=1000, overlap_lines=2)

    return extract_course_record(
        chunks,
        {"source_url": source_url, "source_title": source_title},
        page_context=cleaned[:500]
    )


# Source 1: has Duration only.
SOURCE_1_PAGE = """
Data Science

Duration
9 Months
"""
SOURCE_1_URL = "https://codemehub.com/data-science-course"
SOURCE_1_TITLE = "Data Science Course | Codeme Hub"

# Source 2: a differently-branded/named version, has Fees + Eligibility.
# Institute name is only recoverable from the page's own content here
# (not the title, and not the URL's compound domain) -- this is what a
# real page's own "About"/footer blurb naming itself looks like.
SOURCE_2_PAGE = """
Data Science (Beginners)

Codeme Hub offers this program with dedicated industry mentors.

Fees
Rs. 45,000

Eligibility
Any graduate
"""
SOURCE_2_URL = "https://codemehub-techlearning.com/ds-beginners"
SOURCE_2_TITLE = "Data Science (Beginners) Course | Codeme Hub (Tech Learning)"

# Source 3: back to the plain name, has Mode + Location.
SOURCE_3_PAGE = """
Data Science

Mode
Online & Offline

Location
Calicut
"""
SOURCE_3_URL = "https://codemehub.com/data-science-calicut"
SOURCE_3_TITLE = "Data Science Course | Codeme Hub"


def test_full_pipeline_merges_three_partial_sources_into_one_course():

    record_1 = _extract(SOURCE_1_PAGE, SOURCE_1_URL, SOURCE_1_TITLE)
    record_2 = _extract(SOURCE_2_PAGE, SOURCE_2_URL, SOURCE_2_TITLE)
    record_3 = _extract(SOURCE_3_PAGE, SOURCE_3_URL, SOURCE_3_TITLE)

    # Sanity check on the extraction step itself before merging --
    # each source really is partial on its own.
    assert record_1["duration"] == "9 Months"
    assert record_1["fees"] == NOT_AVAILABLE

    assert record_2["fees"] != NOT_AVAILABLE
    assert "45,000" in record_2["fees"]
    assert record_2["eligibility"] == "Any graduate"
    assert record_2["duration"] == NOT_AVAILABLE

    assert record_3["learning_mode"] != NOT_AVAILABLE
    assert "Online" in record_3["learning_mode"]
    assert record_3["location"] == "Calicut"

    result = deduplicate_courses([record_1, record_2, record_3])

    # ---- ONE merged course, not three ----
    assert len(result["courses"]) == 1
    assert result["merge_report"]["original_count"] == 3
    assert result["merge_report"]["final_count"] == 1
    assert result["merge_report"]["merges_performed"] == 1

    merged = result["courses"][0]

    # ---- every field ends up filled from whichever source had it ----
    assert merged["institute_name"] == "Codeme Hub"
    assert "Data Science" in merged["course_name"]
    assert merged["duration"] == "9 Months"
    assert "45,000" in merged["fees"]
    assert merged["eligibility"] == "Any graduate"
    assert merged["learning_mode"] == "Online & Offline"
    assert merged["location"] == "Calicut"

    # ---- all three sources preserved, duplicates removed ----
    assert len(merged["source_urls"]) == 3
    assert set(merged["source_urls"]) == {SOURCE_1_URL, SOURCE_2_URL, SOURCE_3_URL}
    assert len(merged["sources"]) == 3

    # ---- human-readable merge report, for logging/debugging ----
    log_text = "\n".join(result["merge_report"]["log_lines"])
    assert "Original course records: 3" in log_text
    assert "After deduplication: 1" in log_text
    assert "MERGE:" in log_text
