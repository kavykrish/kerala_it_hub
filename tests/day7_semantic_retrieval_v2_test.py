import sys
import os


# =========================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =========================================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from web_retrieval.page_reader import fetch_page
from web_retrieval.text_cleaner import clean_text
from web_retrieval.chunker import section_chunk_text
from web_retrieval.embedding_model import load_embedding_model
from web_retrieval.retriever import semantic_retrieve_chunks


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    url = "https://www.rogersoft.com/course/data-science-training"

    print("=" * 80)
    print("KERALA IT HUB - DAY 7 HYBRID RETRIEVAL TEST")
    print("=" * 80)


    # =====================================================
    # STEP 1: FETCH WEBPAGE
    # =====================================================

    print("\nSTEP 1: FETCHING WEBPAGE")
    print("-" * 80)

    raw_text = fetch_page(url)

    if not raw_text:
        print("❌ Failed to fetch webpage.")
        return

    print("✅ Webpage fetched successfully.")
    print("Raw characters:", len(raw_text))


    # =====================================================
    # STEP 2: CLEAN TEXT
    # =====================================================

    print("\nSTEP 2: CLEANING TEXT")
    print("-" * 80)

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        print("❌ Text cleaning failed.")
        return

    print("✅ Text cleaned successfully.")
    print("Cleaned characters:", len(cleaned_text))


    # =====================================================
    # STEP 3: CREATE CHUNKS
    # =====================================================

    print("\nSTEP 3: CREATING SECTIONS")
    print("-" * 80)

    chunks = section_chunk_text(
        cleaned_text,
        max_chunk_size=1000,
        overlap_lines=2
    )

    if not chunks:
        print("❌ No chunks created.")
        return

    print("✅ Section chunking successful.")
    print("Total chunks:", len(chunks))


    # =====================================================
    # STEP 4: LOAD EMBEDDING MODEL
    # =====================================================

    print("\nSTEP 4: LOADING EMBEDDING MODEL")
    print("-" * 80)

    model = load_embedding_model()

    if model is None:
        print("❌ Failed to load embedding model.")
        return

    print("✅ Embedding model loaded.")


    # =====================================================
    # STEP 5: TEST QUESTIONS
    # =====================================================

    test_queries = [

        "What is the eligibility for this course?",

        "How long is the course?",

        "Does the course teach machine learning?",

        "What Python topics are covered?",

        "Does the course include deep learning?",

        "Does the course teach SQL?",

        "Does the course include Power BI?",

        "Does the course include Generative AI?"

    ]


    # =====================================================
    # STEP 6: HYBRID RETRIEVAL
    # =====================================================

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 80)


    for query_number, query in enumerate(
        test_queries,
        start=1
    ):

        print("\n" + "=" * 80)
        print(f"QUERY {query_number}")
        print("=" * 80)

        print("\nQuestion:")
        print(query)


        # -------------------------------------------------
        # Retrieve top 3 chunks
        # -------------------------------------------------

        results = semantic_retrieve_chunks(
            query=query,
            chunks=chunks,
            model=model,
            top_k=3
        )


        if not results:
            print("\n❌ No relevant chunks found.")
            continue


        # -------------------------------------------------
        # Display retrieved chunks
        # -------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1
        ):

            (
                final_score,
                semantic_score,
                topic_bonus,
                keyword_bonus,
                matched_topics,
                chunk
            ) = result


            print("\n" + "-" * 80)
            print(f"RESULT {rank}")
            print("-" * 80)

            print(
                "Final score      :",
                round(final_score, 4)
            )

            print(
                "Semantic score   :",
                round(semantic_score, 4)
            )

            print(
                "Topic bonus      :",
                round(topic_bonus, 4)
            )

            print(
                "Keyword bonus    :",
                round(keyword_bonus, 4)
            )

            print(
                "Matched topics   :",
                matched_topics
            )

            print("\nRetrieved chunk:")
            print(chunk[:1000])


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 80)
    print("DAY 7 HYBRID RETRIEVAL TEST COMPLETE")
    print("=" * 80)

    print("""
Pipeline tested:

User Query
    ↓
Semantic Embedding
    +
Exact Topic Matching
    +
Keyword Matching
    ↓
Hybrid Score
    ↓
Top Relevant Chunks
""")


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()