"""
Test for the field-aware retrieval guarantee
(web_retrieval/retriever.py's retrieve_field_aware_chunks).

Uses the real chunker and the real local embedding model (fastembed
runs entirely on-device via ONNX -- no network call, no live web
search, no Groq call). This is the one test in this project that needs
the embedding model loaded, so it's kept separate from
test_course_extractor.py's otherwise-instant unit tests.
"""

import pytest

from web_retrieval.chunker import section_chunk_text
from web_retrieval.embedding_model import load_embedding_model
from web_retrieval.retriever import (
    detect_field_category,
    retrieve_field_aware_chunks,
    semantic_retrieve_chunks,
)


@pytest.fixture(scope="module")
def embedding_model():
    return load_embedding_model()


# A realistic course page with five distinct field sections, none of
# which share much vocabulary with the query below -- this is exactly
# the situation the audit found was losing fields under the old flat
# top_k=4 cut.
MULTI_FIELD_PAGE = """
Data Science and Machine Learning Course

Duration
6 Months, weekday batches

Fees
Rs. 85,000 payable in two installments

Eligibility
Open to graduates from any stream

Mode
Offline classes at the Kochi campus

Location
Kochi, Kerala

Some unrelated marketing paragraph about awards and rankings that has
nothing to do with any specific field, just general praise for the
institute and its trainers and alumni network, written at length so it
is long and keyword-dense compared to the short field sections above.
"""

QUERY = "data science course in kochi"


def test_flat_top_k_alone_can_drop_fields(embedding_model):
    """
    Establishes the bug the fix addresses: with the plain (unchanged)
    semantic_retrieve_chunks and a small top_k, at least one of the
    short field sections is dropped in favour of the long, generic
    marketing paragraph.
    """

    chunks = section_chunk_text(MULTI_FIELD_PAGE, max_chunk_size=1000, overlap_lines=2)

    results = semantic_retrieve_chunks(
        query=QUERY,
        chunks=chunks,
        model=embedding_model,
        top_k=3
    )

    surviving_categories = {
        detect_field_category(item[-1])
        for item in results
        if detect_field_category(item[-1])
    }

    # With only 3 slots and 5 distinct fields competing against a long
    # marketing paragraph, not every field can have survived.
    assert len(surviving_categories) < 5


def test_field_aware_retrieval_keeps_every_distinct_field(embedding_model):

    chunks = section_chunk_text(MULTI_FIELD_PAGE, max_chunk_size=1000, overlap_lines=2)

    results = retrieve_field_aware_chunks(
        query=QUERY,
        chunks=chunks,
        model=embedding_model,
        top_k_general=3
    )

    surviving_categories = {
        detect_field_category(item[-1])
        for item in results
        if detect_field_category(item[-1])
    }

    assert surviving_categories == {
        "duration",
        "fees",
        "eligibility",
        "learning_mode",
        "location",
    }


def test_field_aware_retrieval_returns_same_tuple_shape(embedding_model):

    chunks = section_chunk_text(MULTI_FIELD_PAGE, max_chunk_size=1000, overlap_lines=2)

    results = retrieve_field_aware_chunks(
        query=QUERY,
        chunks=chunks,
        model=embedding_model,
        top_k_general=3
    )

    for item in results:

        assert len(item) == 6

        final_score, semantic_score, topic_bonus, keyword_bonus, matched_topics, chunk = item

        assert isinstance(final_score, float)
        assert isinstance(chunk, str)


def test_field_aware_retrieval_no_duplicate_chunks(embedding_model):

    chunks = section_chunk_text(MULTI_FIELD_PAGE, max_chunk_size=1000, overlap_lines=2)

    results = retrieve_field_aware_chunks(
        query=QUERY,
        chunks=chunks,
        model=embedding_model,
        top_k_general=3
    )

    chunk_texts = [item[-1] for item in results]

    assert len(chunk_texts) == len(set(chunk_texts))


def test_field_aware_retrieval_handles_empty_chunks(embedding_model):

    assert retrieve_field_aware_chunks(QUERY, [], embedding_model) == []
    assert retrieve_field_aware_chunks("", ["some chunk"], embedding_model) == []
