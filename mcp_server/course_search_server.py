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
        f"{query} IT technology course Kerala"
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
    top_k: int = 5
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
        Hybrid Retrieval

    Args:
        query: User's course-related question.
        max_results: Maximum number of webpages to search.
        top_k: Number of relevant chunks to return.

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
        f"{query} IT technology course Kerala"
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
    # STEP 2: FETCH AND PROCESS WEBPAGES
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
        # FETCH WEBPAGE
        # -------------------------------------------------

        page_text = fetch_page(
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
    # STEP 3: RETRIEVE RELEVANT CHUNKS
    # =====================================================

    chunk_texts = [
        item["text"]
        for item in all_chunks
    ]


    retrieved = semantic_retrieve_chunks(
        query=query,
        chunks=chunk_texts,
        model=embedding_model,
        top_k=top_k
    )


    # =====================================================
    # STEP 4: MATCH RETRIEVED CHUNKS TO SOURCES
    # =====================================================

    final_results = []


    for item in retrieved:

        (
            final_score,
            semantic_score,
            topic_bonus,
            keyword_bonus,
            matched_topics,
            chunk
        ) = item


        source = None


        for original in all_chunks:

            if original["text"] == chunk:

                source = original

                break


        final_results.append({

            "content": chunk,

            "source_url": (
                source["source_url"]
                if source
                else ""
            ),

            "source_title": (
                source["source_title"]
                if source
                else ""
            ),

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