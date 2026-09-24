import re

import numpy as np

from web_retrieval.embedding_model import create_embedding


# =========================================================
# 1. TOKENIZATION
# =========================================================

def tokenize(text: str):
    """
    Convert text into lowercase words.
    """

    if not text:
        return []

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# =========================================================
# 2. BASIC KEYWORD SCORE
# =========================================================

def calculate_score(query: str, chunk: str) -> int:
    """
    Calculate basic keyword overlap between
    query and chunk.
    """

    query_words = set(tokenize(query))
    chunk_words = set(tokenize(chunk))

    if not query_words or not chunk_words:
        return 0

    matching_words = query_words.intersection(
        chunk_words
    )

    return len(matching_words)


# =========================================================
# 3. OLD KEYWORD RETRIEVAL
# =========================================================

def retrieve_chunks(
    query: str,
    chunks: list,
    top_k: int = 3
):
    """
    Retrieve chunks using basic keyword matching.
    """

    if not query or not chunks:
        return []

    scored_chunks = []

    for chunk in chunks:

        score = calculate_score(
            query,
            chunk
        )

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        item
        for item in scored_chunks[:top_k]
        if item[0] > 0
    ]


# =========================================================
# 4. IMPORTANT IT TOPICS
# =========================================================

IMPORTANT_TOPICS = [

    "data science",

    "data analytics",

    "machine learning",

    "deep learning",

    "generative ai",

    "artificial intelligence",

    "power bi",

    "tableau",

    "structured query language",

    "sql",

    "python",

    "java",

    "cybersecurity",

    "cyber security",

    "cloud computing",

    "devops",

    "full stack",

    "web development",

    "database",

    "databases",

    "natural language processing",

    "nlp",

    "computer vision",

    "internet of things",

    "iot",

    "blockchain",

    "rag",

    "retrieval augmented generation",

    "tensorflow",

    "pytorch",

    "pandas",

    "numpy",

    "scikit learn",

    "scikit-learn"

]


# =========================================================
# 5. FIND TOPIC MATCHES
# =========================================================

def find_topic_matches(
    query: str,
    chunk: str
):
    """
    Find exact IT topic phrases that appear
    in both the query and the chunk.
    """

    if not query or not chunk:
        return []

    query_lower = query.lower()
    chunk_lower = chunk.lower()

    matched_topics = []

    for topic in IMPORTANT_TOPICS:

        if topic in query_lower and topic in chunk_lower:

            matched_topics.append(topic)

    return matched_topics


# =========================================================
# 6. EXACT TOPIC BONUS
# =========================================================

def calculate_topic_bonus(
    query: str,
    chunk: str
):
    """
    Give a strong bonus when an exact technical
    topic appears in both the question and chunk.

    Example:

    Query:
        Does the course teach SQL?

    Chunk:
        SQL commands
        DDL
        DML
        Joins

    SQL gets a strong bonus.
    """

    matched_topics = find_topic_matches(
        query,
        chunk
    )

    if not matched_topics:
        return 0.0

    # Strong bonus for exact topic match
    return 0.30 * len(matched_topics)


# =========================================================
# 6B. STRUCTURED FIELD BONUS
# =========================================================
# The chunker (see web_retrieval/chunker.py) already splits pages
# into their own small chunk whenever a heading like "Duration" or
# "Eligibility" starts a section. Those chunks are short and don't
# repeat the course/topic keywords from the query (e.g. a chunk that
# is just "Duration: 6 months" has almost nothing in common with a
# query like "python courses in trivandrum"), so plain semantic
# similarity ranks them low and they get crowded out of top_k by
# longer, more keyword-dense chunks -- which is why fields like
# Duration/Fees/Eligibility kept coming back blank even when the
# source page actually had them. Give these a bonus so they're
# reliably included whenever they exist for a page.

# Maps each recognized heading label to the structured Course field
# (see web_retrieval/course_extractor.py) it corresponds to. Several
# labels legitimately map to the same field (e.g. "fee"/"fees"/"course
# fee" all mean the record's "fees" field) -- this is also what lets
# retrieve_field_aware_chunks (below) guarantee at least one chunk per
# *distinct field*, not just per label spelling.
FIELD_LABEL_CATEGORIES = {
    "duration": "duration",
    "eligibility": "eligibility",
    "fees": "fees",
    "fee": "fees",
    "course fee": "fees",
    "admission": "admission_information",
    "certification": "certification",
    "certificate": "certification",
    "mode": "learning_mode",
    "batch": "learning_mode",
    "timing": "learning_mode",
    "schedule": "learning_mode",
    "location": "location",
    "venue": "location",
    "placement": "placement_information",
    "curriculum": "curriculum",
    "syllabus": "curriculum",
    "course name": "course_name",
    "program name": "course_name",
    "course title": "course_name",
    "category": "course_category",
    "course category": "course_category",
}

# Kept as a tuple for compatibility with anything iterating the raw
# label spellings rather than the category mapping above.
FIELD_LABELS = tuple(FIELD_LABEL_CATEGORIES.keys())


_FIELD_LABEL_PATTERNS = {
    label: re.compile(r"^" + re.escape(label) + r"\b", re.IGNORECASE)
    for label in FIELD_LABEL_CATEGORIES
}


def detect_field_category(chunk: str):
    """
    Which structured Course field (see web_retrieval/course_extractor.py
    -- duration, fees, eligibility, learning_mode, location,
    certification, curriculum, admission_information,
    placement_information) this chunk looks like it holds, based on the
    heading label the chunker split it on -- or None if it doesn't look
    like a labelled field section at all.

    Requires the label to be at the very START of the chunk's first
    line (mirroring course_extractor.strip_heading_prefix's own
    stricter check) -- as a standalone heading ("Duration\\n3 months")
    or inline ("Duration: 3 months"). A field word merely appearing
    somewhere WITHIN the chunk is not enough: an FAQ block like "Ques.
    What are the eligibility requirements?" mentions "eligibility" but
    isn't an eligibility section, and "...DEPLOYMENT OF ML MODELS"
    isn't a Mode section just because "MODELS" contains the substring
    "MODE" -- both were confirmed, real false positives against real
    pages before this was tightened from a substring-anywhere check to
    this start-of-line check.
    """

    if not chunk or not chunk.strip():
        return None

    first_line = chunk.strip().splitlines()[0].strip()

    for label, pattern in _FIELD_LABEL_PATTERNS.items():

        if pattern.match(first_line):
            return FIELD_LABEL_CATEGORIES[label]

    return None


def calculate_field_bonus(chunk: str) -> float:
    """
    Give a bonus to chunks that look like they hold a specific,
    structured course detail (duration, fees, eligibility, etc.),
    based on the heading the chunker split it on.
    """

    return 0.25 if detect_field_category(chunk) else 0.0


# =========================================================
# 7. KEYWORD BONUS
# =========================================================

def calculate_keyword_bonus(
    query: str,
    chunk: str
):
    """
    Give a smaller bonus for important individual
    technical words.
    """

    query_words = set(tokenize(query))
    chunk_words = set(tokenize(chunk))

    if not query_words or not chunk_words:
        return 0.0

    important_keywords = {
        "sql",
        "python",
        "java",
        "power",
        "bi",
        "tableau",
        "machine",
        "learning",
        "deep",
        "generative",
        "ai",
        "nlp",
        "rag",
        "cloud",
        "devops",
        "cybersecurity",
        "cyber",
        "data",
        "analytics",
        "science",
        "database",
        "pandas",
        "numpy",
        "tensorflow",
        "pytorch"
    }

    matching_keywords = (
        query_words
        .intersection(chunk_words)
        .intersection(important_keywords)
    )

    return len(matching_keywords) * 0.05


# =========================================================
# 8. HYBRID RETRIEVAL
# =========================================================

def semantic_retrieve_chunks(
    query: str,
    chunks: list,
    model,
    top_k: int = 3
):
    """
    Hybrid retrieval.

    Combines:

    1. Semantic similarity
    2. Exact technical-topic matching
    3. Keyword matching

    Final score:

        Semantic Score
        +
        Topic Bonus
        +
        Keyword Bonus
    """

    if not query or not chunks:
        return []

    # -----------------------------------------------------
    # Create query embedding
    # -----------------------------------------------------

    query_embedding = create_embedding(
        model,
        query
    )

    scored_chunks = []

    # -----------------------------------------------------
    # Process every chunk
    # -----------------------------------------------------

    for chunk in chunks:

        # -----------------------------------------------
        # Semantic similarity
        # -----------------------------------------------

        chunk_embedding = create_embedding(
            model,
            chunk
        )

        # Both embeddings are L2-normalized (see create_embedding),
        # so their dot product is already the cosine similarity.
        semantic_score = float(
            np.dot(query_embedding, chunk_embedding)
        )

        # -----------------------------------------------
        # Exact topic bonus
        # -----------------------------------------------

        topic_bonus = calculate_topic_bonus(
            query,
            chunk
        )

        # -----------------------------------------------
        # Keyword bonus
        # -----------------------------------------------

        keyword_bonus = calculate_keyword_bonus(
            query,
            chunk
        )

        # -----------------------------------------------
        # Structured field bonus
        # -----------------------------------------------

        field_bonus = calculate_field_bonus(
            chunk
        )

        # -----------------------------------------------
        # Final hybrid score
        # -----------------------------------------------

        final_score = (
            semantic_score
            + topic_bonus
            + keyword_bonus
            + field_bonus
        )

        # -----------------------------------------------
        # Find matched topics for debugging
        # -----------------------------------------------

        matched_topics = find_topic_matches(
            query,
            chunk
        )

        # Store all information
        scored_chunks.append(
            (
                final_score,
                semantic_score,
                topic_bonus,
                keyword_bonus,
                matched_topics,
                chunk
            )
        )

    # -----------------------------------------------------
    # Sort by final score
    # -----------------------------------------------------

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scored_chunks[:top_k]


# =========================================================
# 8B. FIELD-AWARE RETRIEVAL
# =========================================================
# semantic_retrieve_chunks's flat top_k cut (above) was silently
# dropping structured fields like fees/eligibility whenever a page had
# more distinct field sections than top_k -- a short, keyword-poor
# chunk like "Fees: Rs. 45,000" barely overlaps a general query like
# "data science courses in kochi" and loses the ranking to longer,
# keyword-dense chunks (e.g. a Curriculum section), even though the
# field bonus nudges its score up slightly. Nudging isn't a guarantee.
#
# This wraps semantic_retrieve_chunks (kept completely unchanged, so
# general relevance ranking still works exactly as before) and adds,
# on top, one guaranteed chunk for every DISTINCT field category this
# page actually has that didn't already make the general cut -- so a
# page with Duration + Fees + Eligibility + Mode + Location keeps all
# five instead of losing whichever ones scored lowest against the
# user's general question.

def retrieve_field_aware_chunks(
    query: str,
    chunks: list,
    model,
    top_k_general: int = 3
):
    """
    Field-aware retrieval for one source page's chunks.

    Returns the general top_k_general chunks by the existing hybrid
    semantic + keyword ranking, PLUS at least one chunk for every
    distinct structured field (see FIELD_LABEL_CATEGORIES) present on
    the page that the general ranking didn't already include.

    Returns the same (final_score, semantic_score, topic_bonus,
    keyword_bonus, matched_topics, chunk) tuple shape as
    semantic_retrieve_chunks, so existing callers can unpack results
    identically -- only the size/composition of the list changes.
    """

    if not query or not chunks:
        return []

    general_top = semantic_retrieve_chunks(
        query=query,
        chunks=chunks,
        model=model,
        top_k=top_k_general
    )

    included_chunk_texts = {
        item[-1]
        for item in general_top
    }

    query_embedding = create_embedding(
        model,
        query
    )

    seen_categories = set()
    field_results = []

    for chunk in chunks:

        if chunk in included_chunk_texts:
            continue

        category = detect_field_category(chunk)

        if not category or category in seen_categories:
            continue

        seen_categories.add(category)
        included_chunk_texts.add(chunk)

        chunk_embedding = create_embedding(
            model,
            chunk
        )

        semantic_score = (
            float(np.dot(query_embedding, chunk_embedding))
            if query_embedding is not None and chunk_embedding is not None
            else 0.0
        )

        topic_bonus = calculate_topic_bonus(query, chunk)
        keyword_bonus = calculate_keyword_bonus(query, chunk)
        field_bonus = calculate_field_bonus(chunk)

        final_score = (
            semantic_score
            + topic_bonus
            + keyword_bonus
            + field_bonus
        )

        matched_topics = find_topic_matches(query, chunk)

        field_results.append((
            final_score,
            semantic_score,
            topic_bonus,
            keyword_bonus,
            matched_topics,
            chunk
        ))

    return general_top + field_results


# =========================================================
# 9. DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 80)
    print("KERALA IT HUB - HYBRID RETRIEVER TEST")
    print("=" * 80)

    sample_chunks = [

        "Eligibility: Open to all graduates and diploma holders.",

        "The course teaches Python and Machine Learning.",

        "The course includes SQL and Power BI.",

        "The course includes Generative AI and RAG.",

        "The course includes Deep Learning using CNN and RNN."

    ]

    from web_retrieval.embedding_model import (
        load_embedding_model
    )

    model = load_embedding_model()

    test_queries = [

        "Does the course teach SQL?",

        "Does the course include Power BI?",

        "Does the course include Generative AI?",

        "Does the course include Deep Learning?"

    ]

    for query in test_queries:

        print("\n" + "=" * 80)
        print("QUESTION")
        print("=" * 80)

        print(query)

        results = semantic_retrieve_chunks(
            query=query,
            chunks=sample_chunks,
            model=model,
            top_k=3
        )

        for rank, (
            final_score,
            semantic_score,
            topic_bonus,
            keyword_bonus,
            matched_topics,
            chunk
        ) in enumerate(
            results,
            start=1
        ):

            print("\n" + "-" * 80)
            print(f"RESULT {rank}")
            print("-" * 80)

            print(
                "Final score    :",
                round(final_score, 4)
            )

            print(
                "Semantic score :",
                round(semantic_score, 4)
            )

            print(
                "Topic bonus    :",
                round(topic_bonus, 4)
            )

            print(
                "Keyword bonus  :",
                round(keyword_bonus, 4)
            )

            print(
                "Matched topics :",
                matched_topics
            )

            print("\nChunk:")
            print(chunk)

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVER TEST COMPLETE")
    print("=" * 80)