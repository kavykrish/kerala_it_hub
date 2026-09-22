from ddgs import DDGS


# ddgs's "auto" backend tries up to 8 underlying engines in small
# batches (2 at a time), waiting up to 5s per batch -- including
# Wikipedia and Grokipedia, which are encyclopedia lookups, not
# general web search, and essentially never surface a course page.
# That alone can add 15-25+ seconds before a single page gets
# fetched. Restricting to general web-search engines cuts that
# overhead substantially while keeping ddgs/DuckDuckGo itself as the
# search library.
SEARCH_BACKENDS = "google,duckduckgo,brave,mojeek"


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
                max_results=max_results,
                backend=SEARCH_BACKENDS
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