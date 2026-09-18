import sys
import os

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

from web_retrieval.page_reader import fetch_page
from web_retrieval.text_cleaner import clean_text
from web_retrieval.chunker import chunk_text
from web_retrieval.embedding_model import load_embedding_model
from web_retrieval.retriever import semantic_retrieve_chunks


def main():

    url = "https://www.rogersoft.com/course/data-science-training"

    questions = [
        "Who can join this course?",
        "How long does the course take?",
        "What will I learn in this course?",
        "Does this course teach machine learning?"
    ]

    print("\n" + "=" * 75)
    print("KERALA IT HUB - DAY 7 SEMANTIC RETRIEVAL TEST")
    print("=" * 75)

    # --------------------------------------------------
    # STEP 1: FETCH WEBPAGE
    # --------------------------------------------------

    print("\nFetching webpage...")

    raw_text = fetch_page(url)

    if not raw_text:
        print("❌ Webpage fetch failed.")
        return

    print("✅ Webpage fetched.")
    print("Raw characters:", len(raw_text))

    # --------------------------------------------------
    # STEP 2: CLEAN
    # --------------------------------------------------

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        print("❌ Text cleaning failed.")
        return

    print("✅ Text cleaned.")
    print("Cleaned characters:", len(cleaned_text))

    # --------------------------------------------------
    # STEP 3: CHUNK
    # --------------------------------------------------

    chunks = chunk_text(
        cleaned_text,
        chunk_size=1000,
        overlap=150
    )

    if not chunks:
        print("❌ Chunking failed.")
        return

    print("✅ Chunks created.")
    print("Total chunks:", len(chunks))

    # --------------------------------------------------
    # STEP 4: LOAD EMBEDDING MODEL
    # --------------------------------------------------

    print("\nLoading embedding model...")

    model = load_embedding_model()

    # --------------------------------------------------
    # STEP 5: SEMANTIC RETRIEVAL
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("SEMANTIC RETRIEVAL")
    print("=" * 75)

    successful_queries = 0

    for question in questions:

        print("\n" + "-" * 75)
        print("QUESTION:")
        print(question)
        print("-" * 75)

        results = semantic_retrieve_chunks(
            question,
            chunks,
            model,
            top_k=3
        )

        if not results:

            print("❌ No results found.")
            continue

        successful_queries += 1

        for rank, (score, chunk) in enumerate(
            results,
            start=1
        ):

            print(f"\nRank {rank}")
            print(
                "Similarity Score:",
                round(score, 4)
            )

            print("\nRetrieved Chunk:")
            print(chunk[:1000])

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 7 SEMANTIC RETRIEVAL SUMMARY")
    print("=" * 75)

    print("Webpage fetched      : YES")
    print("Text cleaned         : YES")
    print("Chunks created       : YES")
    print("Embedding model      : YES")
    print("Semantic retrieval   : YES")
    print("Total chunks         :", len(chunks))
    print("Queries tested       :", len(questions))
    print("Successful queries   :", successful_queries)

    if successful_queries == len(questions):

        print("\n" + "=" * 75)
        print("DAY 7 SEMANTIC RETRIEVAL TEST: SUCCESS")
        print("=" * 75)

    else:

        print("\n" + "=" * 75)
        print("DAY 7 SEMANTIC RETRIEVAL TEST: REVIEW")
        print("=" * 75)


if __name__ == "__main__":
    main()