import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

from web_retrieval.page_reader import fetch_page
from web_retrieval.text_cleaner import clean_text
from web_retrieval.chunker import section_chunk_text


def main():

    url = "https://www.rogersoft.com/course/data-science-training"

    print("=" * 70)
    print("KERALA IT HUB - DAY 7 REAL PAGE SECTION CHUNKING TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: FETCH
    # ---------------------------------------------------------

    print("\nSTEP 1: FETCHING WEBPAGE")
    print("-" * 70)

    raw_text = fetch_page(url)

    if not raw_text:
        print("❌ Failed to fetch webpage.")
        return

    print("✅ Webpage fetched successfully.")
    print("Raw characters:", len(raw_text))

    # ---------------------------------------------------------
    # STEP 2: CLEAN
    # ---------------------------------------------------------

    print("\nSTEP 2: CLEANING TEXT")
    print("-" * 70)

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        print("❌ Cleaning failed.")
        return

    print("✅ Text cleaned successfully.")
    print("Cleaned characters:", len(cleaned_text))

    # ---------------------------------------------------------
    # STEP 3: SECTION CHUNKING
    # ---------------------------------------------------------

    print("\nSTEP 3: SECTION CHUNKING")
    print("-" * 70)

    chunks = section_chunk_text(cleaned_text)

    if not chunks:
        print("❌ No chunks created.")
        return

    print("✅ Section chunking successful.")
    print("Total sections:", len(chunks))

    # ---------------------------------------------------------
    # STEP 4: DISPLAY CHUNKS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SECTION DETAILS")
    print("=" * 70)

    for i, chunk in enumerate(chunks, start=1):

        print("\n" + "-" * 70)
        print(f"SECTION {i}")
        print("-" * 70)

        # Print first 1000 characters
        print(chunk[:1000])

        if len(chunk) > 1000:
            print("\n...[remaining content truncated]...")

        print("\nCharacters:", len(chunk))

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DAY 7 SECTION CHUNKING SUMMARY")
    print("=" * 70)

    print("Webpage fetched       : YES")
    print("Text cleaned          : YES")
    print("Sections created      :", len(chunks))

    print("\n" + "=" * 70)
    print("DAY 7 REAL PAGE SECTION CHUNKING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()