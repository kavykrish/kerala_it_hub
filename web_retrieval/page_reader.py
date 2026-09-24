import html as html_module
import json
import re
import sys

import requests
import trafilatura


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    )
}


def _download_html(url: str):
    """
    The one and only network request per URL -- both fetch_page and
    fetch_page_structured (below) call this and then work from the
    same downloaded HTML, so getting structural signals (H1, JSON-LD)
    alongside the extracted text never costs a second fetch.
    """

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

        return response.text

    except requests.exceptions.RequestException as e:

        print(
            f"Error fetching webpage: {e}",
            file=sys.stderr
        )

        return None


def _extract_readable_text(downloaded_html):

    try:

        text = trafilatura.extract(
            downloaded_html,
            include_links=False,
            include_images=False
        )

        if not text:

            print(
                "Could not extract readable text from the page.",
                file=sys.stderr
            )

            return None

        return text

    except Exception as e:

        print(
            f"Error extracting webpage content: {e}",
            file=sys.stderr
        )

        return None


# ============================================================
# STRUCTURAL SIGNALS (H1 / JSON-LD)
# ============================================================
# trafilatura's plain-text extraction flattens heading hierarchy --
# an <h1> reads identically to any other paragraph by the time
# course extraction ever sees it. That flattening is exactly what let
# a related term mentioned in ordinary prose ("... integrates
# concepts from ... artificial intelligence ...") outrank the page's
# actual, unambiguous subject stated in its own <h1> (see the Step 3D
# investigation). These are lightweight regex extractors, not a full
# HTML parser -- deliberately avoids adding a new dependency
# (BeautifulSoup isn't in requirements.txt) for what's a narrow,
# well-defined task.

_H1_PATTERN = re.compile(r"<h1\b[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_H2_PATTERN = re.compile(r"<h2\b[^>]*>(.*?)</h2>", re.IGNORECASE | re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_JSON_LD_PATTERN = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL
)


def _clean_extracted_text(raw_html_fragment):

    text = _TAG_PATTERN.sub(" ", raw_html_fragment)
    text = html_module.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _extract_h1(downloaded_html):
    """
    The page's first <h1>, tags/entities stripped. None if there
    isn't one, or what's there is too short to be meaningful (e.g. an
    empty/icon-only heading) -- never invented.
    """

    match = _H1_PATTERN.search(downloaded_html)

    if not match:
        return None

    text = _clean_extracted_text(match.group(1))

    if not text or len(text) < 3:
        return None

    return text


def _extract_h2(downloaded_html):
    """
    The page's first <h2> -- used only as a fallback when a page has
    no <h1> at all. Confirmed against a real page during Step 3E
    testing (rogersoft.com has zero <h1> tags but its actual course
    identity, "Data Science & Machine Learning with AI (Artificial
    Intelligence)", is its first <h2>).
    """

    match = _H2_PATTERN.search(downloaded_html)

    if not match:
        return None

    text = _clean_extracted_text(match.group(1))

    if not text or len(text) < 3:
        return None

    return text


def _extract_json_ld_identity(downloaded_html):
    """
    Best-effort identity signal from the page's own JSON-LD structured
    data, if any. Checked in order of reliability:

    1. A WebPage's breadcrumb -- the last (most specific) breadcrumb
       item is usually already a clean subject name on its own
       ("Home" > "Courses" > "Data Science"), no further normalization
       needed.
    2. An Article/NewsArticle/BlogPosting/Course's headline or name --
       typically the same text as the page's own <h1>, so it goes
       through the same course_extractor normalization H1 does.

    None if the page has no JSON-LD, or nothing in it looks like a
    reliable identity signal -- this is supporting evidence only, used
    below H1 in course_extractor's priority order.
    """

    for match in _JSON_LD_PATTERN.finditer(downloaded_html):

        raw = match.group(1).strip()

        try:
            data = json.loads(raw)

        except (json.JSONDecodeError, ValueError):
            continue

        items = data if isinstance(data, list) else [data]

        for item in items:

            if not isinstance(item, dict):
                continue

            if item.get("@type") == "WebPage":

                breadcrumb = item.get("breadcrumb")

                if isinstance(breadcrumb, dict):

                    elements = breadcrumb.get("itemListElement", [])

                    names = [
                        el.get("name").strip()
                        for el in elements
                        if isinstance(el, dict) and isinstance(el.get("name"), str) and el.get("name").strip()
                    ]

                    if names:
                        return names[-1]

    for match in _JSON_LD_PATTERN.finditer(downloaded_html):

        raw = match.group(1).strip()

        try:
            data = json.loads(raw)

        except (json.JSONDecodeError, ValueError):
            continue

        items = data if isinstance(data, list) else [data]

        for item in items:

            if not isinstance(item, dict):
                continue

            if item.get("@type") in ("Article", "NewsArticle", "BlogPosting", "Course"):

                headline = item.get("headline") or item.get("name")

                if isinstance(headline, str) and headline.strip():
                    return headline.strip()

    return None


# ============================================================
# PUBLIC API
# ============================================================

def fetch_page(url: str):
    """
    Fetch and extract readable text from a course webpage. Unchanged
    behaviour/signature -- existing callers (mcp_server's
    fetch_course_page tool, tests) keep working exactly as before.
    """

    downloaded = _download_html(url)

    if downloaded is None:
        return None

    return _extract_readable_text(downloaded)


def fetch_page_structured(url: str):
    """
    Like fetch_page, but also returns the page's H1 (and, if it has no
    H1 at all, its first H2 as a fallback) plus any JSON-LD identity
    signal -- all extracted from the SAME downloaded HTML, no second
    network request. Used by mcp_server/course_search_server.py so
    course_extractor.py can use these as course-identity evidence
    stronger than a keyword match in unordered/reordered chunk text.

    Returns:
        {
            "text": <trafilatura-extracted text, or None>,
            "page_h1": <str or None>,
            "page_h2": <str or None -- only populated when page_h1 is
                None; see _extract_h2>,
            "json_ld_identity": <str or None>,
        }
    """

    downloaded = _download_html(url)

    if downloaded is None:
        return {
            "text": None,
            "page_h1": None,
            "page_h2": None,
            "json_ld_identity": None,
        }

    page_h1 = _extract_h1(downloaded)

    return {
        "text": _extract_readable_text(downloaded),
        "page_h1": page_h1,
        "page_h2": _extract_h2(downloaded) if page_h1 is None else None,
        "json_ld_identity": _extract_json_ld_identity(downloaded),
    }


if __name__ == "__main__":

    url = (
        "https://www.rogersoft.com/"
        "course/data-science-training"
    )

    print("\nFetching webpage...")
    print("URL:", url)
    print("\n" + "=" * 70)

    page_text = fetch_page(url)

    if page_text:

        print("Webpage successfully extracted!")
        print("=" * 70)
        print(page_text[:5000])

        print("\n" + "=" * 70)
        print(
            "Total characters extracted:",
            len(page_text)
        )

    else:

        print("Failed to extract webpage content.")
