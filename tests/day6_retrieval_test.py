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
from web_retrieval.retriever import retrieve_chunks


def main():

    url = "https://www.rogersoft.com/course/data-science-training"

    questions = [
        "What is the eligibility for this course?",
        "What is the duration of the course?",
        "What topics are covered in the course?",
        "Does the course include machine learning?",
    ]

    print("\n" + "=" * 75)
    print("KERALA IT HUB - DAY 6 REAL RETRIEVAL TEST")
    print("=" * 75)

    # --------------------------------------------------
    # STEP 1: FETCH WEBPAGE
    # --------------------------------------------------

    print("\n" + "-" * 75)
    print("STEP 1: FETCH WEBPAGE")
    print("-" * 75)

    raw_text = fetch_page(url)

    if not raw_text:
        print("❌ Failed to fetch webpage.")
        return

    print("✅ Webpage fetched successfully.")
    print("Raw characters:", len(raw_text))

    # --------------------------------------------------
    # STEP 2: CLEAN TEXT
    # --------------------------------------------------

    print("\n" + "-" * 75)
    print("STEP 2: CLEAN TEXT")
    print("-" * 75)

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        print("❌ Cleaning failed.")
        return

    print("✅ Text cleaned successfully.")
    print("Cleaned characters:", len(cleaned_text))

    # --------------------------------------------------
    # STEP 3: CREATE CHUNKS
    # --------------------------------------------------

    print("\n" + "-" * 75)
    print("STEP 3: CREATE CHUNKS")
    print("-" * 75)

    chunks = chunk_text(
        cleaned_text,
        chunk_size=1000,
        overlap=150
    )

    if not chunks:
        print("❌ No chunks created.")
        return

    print("✅ Chunking successful.")
    print("Total chunks:", len(chunks))

    # --------------------------------------------------
    # STEP 4: RETRIEVE RELEVANT CHUNKS
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("STEP 4: RETRIEVE RELEVANT CHUNKS")
    print("=" * 75)

    for question in questions:

        print("\n" + "-" * 75)
        print("QUESTION:")
        print(question)
        print("-" * 75)

        results = retrieve_chunks(
            question,
            chunks,
            top_k=3
        )

        if not results:

            print("❌ No relevant chunks found.")
            continue

        for rank, (score, chunk) in enumerate(
            results,
            start=1
        ):

            print(f"\nRank {rank}")
            print("Relevance Score:", score)

            print("\nRetrieved Content:")
            print(chunk[:1200])

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 6 REAL RETRIEVAL TEST COMPLETE")
    print("=" * 75)

    print("Webpage fetched : YES")
    print("Text cleaned    : YES")
    print("Chunks created  : YES")
    print("Retrieval tested: YES")
    print("Total chunks    :", len(chunks))

    print("\n" + "=" * 75)
    print("DAY 6 RETRIEVAL PIPELINE: SUCCESS")
    print("=" * 75)


if __name__ == "__main__":
    main()