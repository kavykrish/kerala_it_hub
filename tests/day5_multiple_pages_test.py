import sys
import os

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

from web_retrieval.search_engine import search_web
from web_retrieval.page_reader import fetch_page
from web_retrieval.text_cleaner import (
    clean_text,
    is_useful_course_page
)


def main():

    query = "Data Science courses in Kochi Kerala"

    print("\n" + "=" * 75)
    print("KERALA IT HUB - DAY 5 MULTIPLE PAGE TEST")
    print("=" * 75)

    print("\nSearch Query:")
    print(query)

    # --------------------------------------------------
    # STEP 1: SEARCH WEB
    # --------------------------------------------------

    print("\n" + "-" * 75)
    print("STEP 1: SEARCHING WEB")
    print("-" * 75)

    results = search_web(query, max_results=5)

    if not results:
        print("\nNo search results found.")
        return

    print(f"\nFound {len(results)} search results.")

    # --------------------------------------------------
    # STEP 2: TEST EACH PAGE
    # --------------------------------------------------

    successful_pages = 0
    useful_pages = 0

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 75)
        print(f"RESULT {i}")
        print("=" * 75)

        title = result.get("title", "")
        url = result.get("url", "")

        print("Title:", title)
        print("URL:", url)

        if not url:
            print("❌ No URL available.")
            continue

        # --------------------------------------------------
        # FETCH PAGE
        # --------------------------------------------------

        print("\nFetching webpage...")

        raw_text = fetch_page(url)

        if not raw_text:

            print("❌ Page extraction failed.")
            continue

        successful_pages += 1

        print("✅ Page extracted successfully.")
        print("Raw characters:", len(raw_text))

        # --------------------------------------------------
        # CLEAN TEXT
        # --------------------------------------------------

        cleaned_text = clean_text(raw_text)

        if not cleaned_text:

            print("❌ Cleaning failed.")
            continue

        print("✅ Text cleaned successfully.")
        print("Cleaned characters:", len(cleaned_text))

        # --------------------------------------------------
        # QUALITY CHECK
        # --------------------------------------------------

        useful = is_useful_course_page(cleaned_text)

        if useful:

            useful_pages += 1

            print("✅ Useful course page: YES")

        else:

            print("❌ Useful course page: NO")

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 5 MULTIPLE PAGE TEST SUMMARY")
    print("=" * 75)

    print("Search results found       :", len(results))
    print("Pages successfully read    :", successful_pages)
    print("Useful course pages        :", useful_pages)

    print("\n" + "=" * 75)

    if successful_pages > 0 and useful_pages > 0:

        print("DAY 5 MULTIPLE PAGE TEST: SUCCESS")

    else:

        print("DAY 5 MULTIPLE PAGE TEST: NEEDS REVIEW")

    print("=" * 75)


if __name__ == "__main__":
    main()