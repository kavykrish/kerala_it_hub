import re

from web_retrieval.embedding_model import create_embedding

from sklearn.metrics.pairwise import cosine_similarity


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

        semantic_score = cosine_similarity(
            [query_embedding],
            [chunk_embedding]
        )[0][0]

        semantic_score = float(
            semantic_score
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
        # Final hybrid score
        # -----------------------------------------------

        final_score = (
            semantic_score
            + topic_bonus
            + keyword_bonus
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