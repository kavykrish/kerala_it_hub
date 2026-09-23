from urllib.parse import urlsplit, urlunsplit

from ddgs import DDGS


def normalize_url(url: str) -> str:
    """
    Reduce a URL to a form that's stable across search engines, so
    the same page returned by two different backends (with different
    tracking params, a trailing slash, or http vs https) is recognized
    as one duplicate instead of two separate results.
    """

    if not url:
        return ""

    parts = urlsplit(url)

    path = parts.path.rstrip("/")

    # Drop query string and fragment -- almost always just tracking
    # params (utm_source, etc.), never part of the page's real identity
    # for the kind of static course-listing pages we're fetching.
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        path,
        "",
        ""
    ))


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

    seen_urls = set()

    try:
        with DDGS() as ddgs:
            # Over-fetch a bit before deduping -- querying multiple
            # backends (see SEARCH_BACKENDS) commonly returns the same
            # popular page more than once, so asking for exactly
            # max_results raw results can leave fewer than max_results
            # *distinct* ones after dedup.
            search_results = ddgs.text(
                query,
                max_results=max_results * 2,
                backend=SEARCH_BACKENDS
            )

            for result in search_results:

                url = result.get("href", "")

                normalized = normalize_url(url)

                if not normalized or normalized in seen_urls:
                    continue

                seen_urls.add(normalized)

                results.append({
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": result.get("body", "")
                })

                if len(results) >= max_results:
                    break

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