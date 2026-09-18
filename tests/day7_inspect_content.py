import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

from web_retrieval.page_reader import fetch_page
from web_retrieval.text_cleaner import clean_text


def main():

    url = "https://www.rogersoft.com/course/data-science-training"

    print("=" * 70)
    print("KERALA IT HUB - DAY 7 CONTENT INSPECTION")
    print("=" * 70)

    raw_text = fetch_page(url)

    if not raw_text:
        print("Failed to fetch webpage.")
        return

    cleaned_text = clean_text(raw_text)

    print("\nTotal characters:", len(cleaned_text))

    # Important course-related keywords
    keywords = [
        "Duration",
        "Eligibility",
        "Course Highlights",
        "Course Modules",
        "Machine Learning",
        "Deep Learning",
        "Generative AI",
        "Reviews",
        "Testimonials"
    ]

    print("\n" + "=" * 70)
    print("KEYWORD LOCATIONS")
    print("=" * 70)

    for keyword in keywords:

        position = cleaned_text.lower().find(
            keyword.lower()
        )

        if position != -1:
            print(f"{keyword:25} → character {position}")
        else:
            print(f"{keyword:25} → NOT FOUND")

    print("\n" + "=" * 70)
    print("CONTENT AROUND COURSE INFORMATION")
    print("=" * 70)

    # Display the first 8000 characters
    print(cleaned_text[:8000])


if __name__ == "__main__":
    main()