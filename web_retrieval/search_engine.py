from ddgs import DDGS


def search_web(query: str, max_results: int = 5):
    """
    Search the web and return relevant search results.

    Parameters:
        query: Search query entered by the user.
        max_results: Maximum number of results to return.

    Returns:
        A list of dictionaries containing title, URL and snippet.
    """

    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=max_results
            )

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })

    except Exception as e:
        print(f"Web search error: {e}")

    return results


if __name__ == "__main__":

    query = "Data Science courses in Kochi Kerala"

    print("\nSearching the web...")
    print(f"Query: {query}\n")

    results = search_web(query)

    if not results:
        print("No search results found.")

    else:
        for i, result in enumerate(results, start=1):

            print(f"Result {i}")
            print("-" * 60)
            print("Title:", result["title"])
            print("URL:", result["url"])
            print("Snippet:", result["snippet"])
            print()