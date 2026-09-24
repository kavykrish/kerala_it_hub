import concurrent.futures
import logging
import re

from mcp.server import MCPServer

from web_retrieval.search_engine import search_web
from web_retrieval.page_reader import fetch_page, fetch_page_structured
from web_retrieval.text_cleaner import (
    clean_text,
    is_useful_course_page
)
from web_retrieval.chunker import section_chunk_text
from web_retrieval.embedding_model import load_embedding_model
from web_retrieval.retriever import retrieve_field_aware_chunks
from web_retrieval.course_extractor import extract_course_record
from web_retrieval.course_merger import deduplicate_courses
from web_retrieval.query_intent import parse_query_intent
from web_retrieval.course_ranker import filter_and_rank_courses


logger = logging.getLogger(__name__)


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

    # Structural identity signals (H1, H2 fallback, JSON-LD) from the
    # SAME download fetch_page_structured already does for page_texts
    # above -- no second network request. See
    # web_retrieval/page_reader.py and course_extractor.py's
    # course_name priority order (explicit heading > H1 > H2 fallback
    # > JSON-LD > page_context).
    page_h1_by_source = {}
    page_h2_by_source = {}
    json_ld_identity_by_source = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:

        future_to_url = {
            executor.submit(fetch_page_structured, url): url
            for url in urls_to_fetch
        }

        for future in concurrent.futures.as_completed(future_to_url):

            url = future_to_url[future]

            try:
                result = future.result()
                page_texts[url] = result["text"]
                page_h1_by_source[url] = result["page_h1"]
                page_h2_by_source[url] = result["page_h2"]
                json_ld_identity_by_source[url] = result["json_ld_identity"]

            except Exception:
                page_texts[url] = None
                page_h1_by_source[url] = None
                page_h2_by_source[url] = None
                json_ld_identity_by_source[url] = None


    # =====================================================
    # STEP 3: CLEAN, FILTER AND CHUNK EACH FETCHED PAGE
    # =====================================================

    all_chunks = []

    source_pages = []

    # A bounded prefix of each page's own cleaned text, in ORIGINAL
    # reading order -- unlike the chunks below, which field-aware
    # retrieval (Step 4) re-ranks by relevance score and can return in
    # a different order, or omit entirely. course_extractor uses this
    # to find the page's own opening heading/subject line, which is a
    # far more reliable signal for the actual course name than a
    # keyword match anywhere in a reordered/retrieved chunk list (see
    # web_retrieval/course_extractor.py's page_context handling).
    PAGE_CONTEXT_CHAR_LIMIT = 500

    page_context_by_source = {}


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

        page_context_by_source[url] = cleaned_text[:PAGE_CONTEXT_CHAR_LIMIT]


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
    # crowding out every other institute almost entirely. Retrieving
    # top_k chunks from *each* source separately instead guarantees
    # every fetched, relevant page gets a fair chance to contribute
    # its own course details.
    #
    # Within one page, a flat top_k cut still systematically dropped
    # short, field-specific chunks (a chunk that's just "Duration: 6
    # months" barely overlaps the query's words at all) in favour of
    # long, keyword-dense ones -- so retrieve_field_aware_chunks adds a
    # guarantee on top of the existing ranking: every distinct
    # structured field the page actually has (duration, fees,
    # eligibility, mode, location, certification, curriculum,
    # admission, placement) gets at least one chunk through, in
    # addition to the general top_k by relevance to the question.

    chunks_by_source = {}

    for item in all_chunks:

        chunks_by_source.setdefault(
            item["source_url"],
            []
        ).append(item)


    final_results = []

    # One structured Course record per source page (see
    # web_retrieval/course_extractor.py) -- built from the SAME
    # field-aware retrieved chunks as final_results below, grouped by
    # source_url so every source produces exactly one record, never
    # one record per chunk. Deduplicated/merged in Step 4B; this list
    # itself stays one-record-per-page.
    course_records = []


    for source_url, source_chunks in chunks_by_source.items():

        chunk_texts = [
            item["text"]
            for item in source_chunks
        ]

        source_title = source_chunks[0]["source_title"]

        retrieved = retrieve_field_aware_chunks(
            query=query,
            chunks=chunk_texts,
            model=embedding_model,
            top_k_general=top_k
        )

        retrieved_chunk_texts = []

        for item in retrieved:

            (
                final_score,
                semantic_score,
                topic_bonus,
                keyword_bonus,
                matched_topics,
                chunk
            ) = item

            retrieved_chunk_texts.append(chunk)

            final_results.append({

                "content": chunk,

                "source_url": source_url,

                "source_title": source_title,

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

        course_records.append(
            extract_course_record(
                retrieved_chunk_texts,
                {
                    "source_url": source_url,
                    "source_title": source_title
                },
                page_context=page_context_by_source.get(source_url, ""),
                page_h1=page_h1_by_source.get(source_url),
                page_h2=page_h2_by_source.get(source_url),
                json_ld_identity=json_ld_identity_by_source.get(source_url)
            )
        )


    # =====================================================
    # STEP 4B: DEDUPLICATE / MERGE COURSE RECORDS
    # =====================================================
    # Different pages (the institute's own site, a directory listing,
    # a review blog, ...) describing the same real course used to
    # become entirely independent "SOURCE" blocks with no signal tying
    # them together -- see web_retrieval/course_merger.py for the
    # normalization + merge logic this fixes.

    dedup_result = deduplicate_courses(course_records)

    for line in dedup_result["merge_report"]["log_lines"]:
        logger.debug(line)


    # =====================================================
    # STEP 4C: QUERY-AWARE RANKING (Step 5)
    # =====================================================
    # Deterministic, no LLM call -- see web_retrieval/query_intent.py
    # and web_retrieval/course_ranker.py. Reorders the deduplicated
    # records so the ones that actually match what the user asked
    # about (named institute, topic, requested fields, level,
    # location) surface first; never drops a record, and never treats
    # a related-but-different course (Machine Learning, AI) as if it
    # WERE the requested topic (Data Science) -- see
    # course_ranker.py's module docstring.

    query_intent = parse_query_intent(query)

    ranked_courses = filter_and_rank_courses(
        dedup_result["courses"],
        query_intent,
        query=query
    )


    # =====================================================
    # STEP 5: RETURN RAG CONTEXT
    # =====================================================
    # retrieved_results is kept exactly as before (backward
    # compatible with anything already consuming it). course_records
    # is new: the deduplicated/merged/ranked structured records, for
    # backend/rag_generator.py to optionally use instead of raw
    # chunks. query_intent is new -- generate_rag_answer uses it to
    # tell the model what was specifically asked, without a second
    # LLM call. merge_report is debug/log-only, not meant for display.

    return {

        "query": query,

        "sources_checked": len(
            source_pages
        ),

        "chunks_created": len(
            all_chunks
        ),

        "retrieved_results": final_results,

        "course_records": ranked_courses,

        "query_intent": query_intent,

        "merge_report": dedup_result["merge_report"]

    }


# =========================================================
# RUN MCP SERVER
# =========================================================

if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )