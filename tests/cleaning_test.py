import sys
import os

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

from web_retrieval.page_reader import fetch_page
from web_retrieval.text_cleaner import (
    clean_text,
    is_useful_course_page
)


def main():

    # --------------------------------------------------
    # TEST WEBPAGE
    # --------------------------------------------------

    url = "https://www.rogersoft.com/course/data-science-training"

    print("\n" + "=" * 70)
    print("KERALA IT HUB - DAY 5 CLEANING TEST")
    print("=" * 70)

    print("\nWebpage URL:")
    print(url)

    # --------------------------------------------------
    # STEP 1: FETCH WEBPAGE
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 1: FETCHING WEBPAGE")
    print("-" * 70)

    raw_text = fetch_page(url)

    if not raw_text:

        print("\n❌ Failed to fetch webpage.")
        return

    print("\n✅ Webpage fetched successfully.")

    print("Raw characters:", len(raw_text))

    # --------------------------------------------------
    # STEP 2: CLEAN TEXT
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 2: CLEANING TEXT")
    print("-" * 70)

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:

        print("\n❌ Text cleaning failed.")
        return

    print("\n✅ Text cleaned successfully.")

    print("Cleaned characters:", len(cleaned_text))

    # --------------------------------------------------
    # STEP 3: COURSE PAGE QUALITY CHECK
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 3: COURSE PAGE QUALITY CHECK")
    print("-" * 70)

    useful_page = is_useful_course_page(cleaned_text)

    if useful_page:

        print("\n✅ Useful course page: YES")

    else:

        print("\n❌ Useful course page: NO")

    # --------------------------------------------------
    # STEP 4: DISPLAY CLEANED CONTENT
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CLEANED WEBPAGE CONTENT")
    print("=" * 70)

    print(cleaned_text[:5000])

    # --------------------------------------------------
    # STEP 5: SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("DAY 5 TEST SUMMARY")
    print("=" * 70)

    print("URL fetched successfully : YES")
    print("Text extracted           : YES")
    print("Text cleaned             : YES")

    if useful_page:
        print("Course page validation    : PASS")
    else:
        print("Course page validation    : FAIL")

    print("Raw characters           :", len(raw_text))
    print("Cleaned characters       :", len(cleaned_text))

    print("\n" + "=" * 70)

    if useful_page:

        print("DAY 5 CLEANING TEST: SUCCESS")

    else:

        print("DAY 5 CLEANING TEST: FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()