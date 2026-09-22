import concurrent.futures
import re

from mcp.server import MCPServer

from web_retrieval.search_engine import search_web
from web_retrieval.page_reader import fetch_page
from web_retrieval.text_cleaner import (
    clean_text,
    is_useful_course_page
)
from web_retrieval.chunker import section_chunk_text
from web_retrieval.embedding_model import load_embedding_model
from web_retrieval.retriever import semantic_retrieve_chunks


# =========================================================
# CREATE MCP SERVER
# =========================================================

mcp = MCPServer(
    "Kerala IT Hub"
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

embedding_model = load_embedding_model()


# =========================================================
# CLEAN SEARCH QUERY
# =========================================================
# Users phrase questions conversationally ("I'm a beginner in
# Trivandrum, can you suggest good data science courses with
# placement support and tell me which is best?"), which is exactly
# what we want for the final answer, but a literal web search engine
# does much worse with a long, filler-heavy sentence than with a
# short, keyword-focused one. Strip conversational filler (leading
# AND trailing) and cap the length, while the full original question
# is still used for chunk ranking and for the LLM's answer.

# Not anchored to the start of the string -- filler commonly trails
# an initial clause too ("...in Trivandrum, can you suggest...").
FILLER_PATTERNS = (
    r"\bi\s*('?m| am)\s+a?\s*beginner\s*(in|at)?\b",
    r"\bi\s*('?m| am)\s+(looking|searching)\s+for\b",
    r"\bi\s+(want|need)\s+to\s+know\s*(about)?\b",
    r"\bi\s+(want|need)\b",
    r"\b(can|could|would)\s+you\s+(please\s+)?(suggest|tell|recommend|help)\w*\s*(me)?\b",
    r"\bplease\s+(suggest|tell|recommend|help)\w*\s*(me)?\b",
    r"\b(suggest|recommend)\s+me\b",
    r"\btell\s+me\s*(about)?\b",
    r"\blooking\s+for\b",
)

TRAILING_INSTRUCTION_TRIGGERS = (
    "compare",
    "comparison",
    "and tell",
    "and let me know",
    "and suggest",
    "suggest which",
    "which is better",
    "which one is better",
    "which is best",
    "which one is best",
    "tell me which",
    "explain",
)

# A search engine's precision drops on very long, sentence-like
# queries -- keep it focused on the core topic/location words.
MAX_SEARCH_QUERY_WORDS = 14


def clean_search_query(query: str) -> str:
    """
    Turn a conversationally-phrased question into a short,
    keyword-focused string to actually search the web for.
    """

    cleaned = query.strip()

    # ----------------------------------------------------
    # Strip conversational filler, wherever it appears (a
    # question can chain more than one filler phrase, e.g.
    # "I am a beginner in X, can you suggest...")
    # ----------------------------------------------------

    for pattern in FILLER_PATTERNS:

        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")

    if not cleaned:
        cleaned = query.strip()

    # ----------------------------------------------------
    # Strip trailing instructional clause
    # ----------------------------------------------------

    lowered = cleaned.lower()

    cut_index = len(cleaned)

    for trigger in TRAILING_INSTRUCTION_TRIGGERS:

        idx = lowered.find(trigger)

        if idx != -1:
            cut_index = min(cut_index, idx)

    cleaned = cleaned[:cut_index].strip(" ,.")

    if not cleaned:
        cleaned = query.strip()

    # ----------------------------------------------------
    # Tidy up a dangling leading/trailing preposition or
    # conjunction left over from either cleaning step above.
    # ----------------------------------------------------

    for _ in range(2):

        cleaned = re.sub(
            r"^(about|for|on|in|at)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE
        ).strip(" ,.")

        cleaned = re.sub(
            r"\s+(and|with|for|to|about|in|at|of|on)$",
            "",
            cleaned,
            flags=re.IGNORECASE
        ).strip(" ,.")

    if not cleaned:
        cleaned = query.strip()

    # ----------------------------------------------------
    # Cap length -- keep the first N words
    # ----------------------------------------------------

    words = cleaned.split()

    if len(words) > MAX_SEARCH_QUERY_WORDS:
        cleaned = " ".join(words[:MAX_SEARCH_QUERY_WORDS])

    return cleaned


# =========================================================
# TOOL 1: SEARCH KERALA IT COURSES
# =========================================================

@mcp.tool()
def search_kerala_courses(
    query: str,
    max_results: int = 5
):
    """
    Search for IT and technology courses in Kerala.

    Args:
        query: Course-related search question.
        max_results: Maximum number of search results.

    Returns:
        Search results containing title, URL and snippet.
    """

    if not query:
        return {
            "error": "Search query cannot be empty."
        }

    search_query = (
        f"{clean_search_query(query)} IT technology course Kerala"
    )

    results = search_web(
        search_query,
        max_results=max_results
    )

    return {
        "query": query,
        "results": results
    }


# =========================================================
# TOOL 2: FETCH COURSE WEBPAGE
# =========================================================

@mcp.tool()
def fetch_course_page(
    url: str
):
    """
    Fetch and extract readable text from a course webpage.

    Args:
        url: URL of the course webpage.

    Returns:
        Extracted webpage text.
    """

    if not url:
        return {
            "error": "URL cannot be empty."
        }

    page_text = fetch_page(url)

    if not page_text:
        return {
            "error": "Could not extract webpage content.",
            "url": url
        }

    return {
        "url": url,
        "content": page_text
    }


# =========================================================
# TOOL 3: RETRIEVE COURSE INFORMATION
# =========================================================

@mcp.tool()
def retrieve_course_information(
    query: str,
    max_results: int = 5,
    top_k: int = 3
):
    """
    Search Kerala IT courses and retrieve the most
    relevant information using the RAG retrieval pipeline.

    Pipeline:

        Web Search
            ↓
        Page Reader
            ↓
        Text Cleaning
            ↓
        Chunking
            ↓
        Hybrid Retrieval (per source page)

    Args:
        query: User's course-related question.
        max_results: Maximum number of webpages to search.
        top_k: Number of relevant chunks to keep PER SOURCE PAGE
            (not a global total) -- e.g. max_results=6, top_k=3 can
            return up to 18 chunks, three from each of up to six
            pages, so every fetched page gets a fair chance to
            contribute its own course details instead of the single
            most topically-relevant page crowding out the rest.

    Returns:
        Relevant course information with source URLs.
    """

    if not query:
        return {
            "error": "Query cannot be empty."
        }


    # =====================================================
    # STEP 1: WEB SEARCH
    # =====================================================

    search_query = (
        f"{clean_search_query(query)} IT technology course Kerala"
    )

    search_results = search_web(
        search_query,
        max_results=max_results
    )


    if not search_results:
        return {
            "query": query,
            "results": [],
            "message": "No search results found."
        }


    # =====================================================
    # STEP 2: FETCH WEBPAGES IN PARALLEL
    # =====================================================
    # Fetching is pure network wait time, and one page's content
    # doesn't depend on any other page's fetch -- doing this one at a
    # time (as it used to) meant up to `max_results` sequential
    # round-trips before any processing even started, which was the
    # single biggest contributor to comparison questions (max_results
    # up to 10) timing out on Render's slower/shared CPU.

    urls_to_fetch = [
        result.get("url", "")
        for result in search_results
        if result.get("url", "")
    ]

    page_texts = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:

        future_to_url = {
            executor.submit(fetch_page, url): url
            for url in urls_to_fetch
        }

        for future in concurrent.futures.as_completed(future_to_url):

            url = future_to_url[future]

            try:
                page_texts[url] = future.result()

            except Exception:
                page_texts[url] = None


    # =====================================================
    # STEP 3: CLEAN, FILTER AND CHUNK EACH FETCHED PAGE
    # =====================================================

    all_chunks = []

    source_pages = []


    for result in search_results:

        url = result.get(
            "url",
            ""
        )

        if not url:
            continue


        # -------------------------------------------------
        # ALREADY-FETCHED WEBPAGE TEXT
        # -------------------------------------------------

        page_text = page_texts.get(
            url
        )

        if not page_text:
            continue


        # -------------------------------------------------
        # CLEAN TEXT
        # -------------------------------------------------

        cleaned_text = clean_text(
            page_text
        )

        if not cleaned_text:
            continue


        # -------------------------------------------------
        # CHECK COURSE RELEVANCE
        # -------------------------------------------------

        if not is_useful_course_page(
            cleaned_text
        ):
            continue


        # -------------------------------------------------
        # CHUNK TEXT
        # -------------------------------------------------

        chunks = section_chunk_text(
            cleaned_text,
            max_chunk_size=1000,
            overlap_lines=2
        )

        if not chunks:
            continue


        # -------------------------------------------------
        # STORE CHUNKS WITH SOURCE INFORMATION
        # -------------------------------------------------

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "source_url": url,
                "source_title": result.get(
                    "title",
                    ""
                )
            })


        source_pages.append({
            "title": result.get(
                "title",
                ""
            ),
            "url": url
        })


    # =====================================================
    # CHECK PROCESSED CONTENT
    # =====================================================

    if not all_chunks:

        return {
            "query": query,
            "results": [],
            "message": (
                "No useful course information "
                "could be extracted."
            )
        }


    # =====================================================
    # STEP 4: RETRIEVE RELEVANT CHUNKS, PER SOURCE
    # =====================================================
    # Retrieving one global top_k across every page pooled together
    # let 1-2 heavily topic-matching pages dominate the whole result,
    # crowding out every other institute almost entirely -- and it
    # systematically dropped short, field-specific chunks (a chunk
    # that's just "Duration: 6 months" barely overlaps the query's
    # words at all) in favour of long, keyword-dense ones. Retrieving
    # top_k chunks from *each* source separately instead guarantees
    # every fetched, relevant page gets a fair chance to contribute
    # its own course details.

    chunks_by_source = {}

    for item in all_chunks:

        chunks_by_source.setdefault(
            item["source_url"],
            []
        ).append(item)


    final_results = []


    for source_url, source_chunks in chunks_by_source.items():

        chunk_texts = [
            item["text"]
            for item in source_chunks
        ]

        retrieved = semantic_retrieve_chunks(
            query=query,
            chunks=chunk_texts,
            model=embedding_model,
            top_k=top_k
        )

        for item in retrieved:

            (
                final_score,
                semantic_score,
                topic_bonus,
                keyword_bonus,
                matched_topics,
                chunk
            ) = item

            final_results.append({

                "content": chunk,

                "source_url": source_url,

                "source_title": source_chunks[0]["source_title"],

                "score": round(
                    float(final_score),
                    4
                ),

                "semantic_score": round(
                    float(semantic_score),
                    4
                ),

                "matched_topics": matched_topics

            })


    # =====================================================
    # STEP 5: RETURN RAG CONTEXT
    # =====================================================

    return {

        "query": query,

        "sources_checked": len(
            source_pages
        ),

        "chunks_created": len(
            all_chunks
        ),

        "retrieved_results": final_results

    }


# =========================================================
# RUN MCP SERVER
# =========================================================

if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )