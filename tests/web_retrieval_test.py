import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from web_retrieval.search_engine import search_web
from web_retrieval.page_reader import fetch_page


def main():

    query = "Data Science courses in Kochi Kerala"

    print("\n" + "=" * 70)
    print("KERALA IT HUB - DAY 4 WEB RETRIEVAL TEST")
    print("=" * 70)

    print("\nSearch Query:")
    print(query)

    # --------------------------------------------------
    # STEP 1: Search the web
    # --------------------------------------------------

    print("\nSearching the web...")
    
    results = search_web(query, max_results=5)

    if not results:
        print("No search results found.")
        return

    print(f"\nFound {len(results)} search results.")

    # --------------------------------------------------
    # STEP 2: Display search results
    # --------------------------------------------------

    for i, result in enumerate(results, start=1):

        print("\n" + "-" * 70)
        print(f"Result {i}")
        print("-" * 70)

        print("Title:", result["title"])
        print("URL:", result["url"])
        print("Snippet:", result["snippet"][:300])

    # --------------------------------------------------
    # STEP 3: Try reading the search results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TESTING PAGE EXTRACTION")
    print("=" * 70)

    successful_page = False

    for i, result in enumerate(results, start=1):

        url = result["url"]

        print(f"\nTrying Result {i}...")
        print("URL:", url)

        page_text = fetch_page(url)

        if page_text:

            print("\nSUCCESS!")
            print("Page successfully extracted.")

            print("\nFirst 3000 characters:")
            print("-" * 70)
            print(page_text[:3000])

            print("\n" + "-" * 70)
            print("Total characters extracted:", len(page_text))

            successful_page = True
            break

        else:

            print("Could not extract this page.")
            print("Trying the next result...")

    # --------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------

    print("\n" + "=" * 70)

    if successful_page:
        print("DAY 4 WEB RETRIEVAL TEST: SUCCESS")
    else:
        print("DAY 4 WEB RETRIEVAL TEST: FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()