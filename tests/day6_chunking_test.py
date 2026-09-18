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


def main():

    url = "https://www.rogersoft.com/course/data-science-training"

    print("\n" + "=" * 75)
    print("KERALA IT HUB - DAY 6 REAL WEBPAGE CHUNKING TEST")
    print("=" * 75)

    print("\nURL:")
    print(url)

    # --------------------------------------------------
    # STEP 1: FETCH
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
    # STEP 2: CLEAN
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
    # STEP 3: CHUNK
    # --------------------------------------------------

    print("\n" + "-" * 75)
    print("STEP 3: CREATE TEXT CHUNKS")
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
    # STEP 4: DISPLAY CHUNKS
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("GENERATED CHUNKS")
    print("=" * 75)

    for i, chunk in enumerate(chunks, start=1):

        print("\n" + "-" * 75)
        print(f"CHUNK {i}")
        print("-" * 75)

        print(chunk)

        print("\nCharacters:", len(chunk))

    # --------------------------------------------------
    # STEP 5: SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 6 CHUNKING TEST SUMMARY")
    print("=" * 75)

    print("Webpage fetched       : YES")
    print("Text cleaned          : YES")
    print("Chunks created        : YES")
    print("Total chunks          :", len(chunks))

    print("\n" + "=" * 75)
    print("DAY 6 CHUNKING TEST: SUCCESS")
    print("=" * 75)


if __name__ == "__main__":
    main()