import asyncio
import contextlib
import json
import os
import sys

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mcp import (
    ClientSession,
    StdioServerParameters
)

from mcp.client.stdio import (
    stdio_client
)

from backend.rag_generator import (
    generate_rag_answer
)


# ============================================================
# OPTIONAL SHARED API KEY
# ============================================================
# If API_KEY is set in the environment, every /ask request must send a
# matching "X-API-Key" header. This is what stops a stranger who gets a
# copy of the APK from hammering your Groq quota and web-scraping budget.
# Leave API_KEY unset for local development.

API_KEY = os.getenv("API_KEY")


# ============================================================
# MCP SESSION LIFECYCLE
# ============================================================
# The MCP server (and the sentence-transformers embedding model it loads)
# used to be started fresh for every single /ask request, which meant
# re-loading the embedding model from scratch on every question. It is
# now started once when FastAPI boots and reused for the app's lifetime.
# stdio-based MCP sessions only support one in-flight call at a time, so
# concurrent requests are serialized through mcp_lock.

mcp_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.course_search_server"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            app.state.mcp_session = session

            yield

    app.state.mcp_session = None


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Kerala IT Hub API",
    description=(
        "AI-powered IT Course and Institute "
        "Navigator for Kerala"
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class HistoryTurn(BaseModel):
    question: str
    answer: str


class QuestionRequest(BaseModel):
    question: str
    # Recent conversation turns, oldest first, so follow-up questions
    # ("what about the fees for that one?") can be understood in
    # context. The client is expected to send only the last few turns
    # -- this is capped again server-side regardless (see ask_question).
    history: list[HistoryTurn] = []


# ============================================================
# API KEY CHECK
# ============================================================

def verify_api_key(x_api_key: str | None):

    if not API_KEY:
        return

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Kerala IT Hub API is running!",
        "status": "success"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# MCP RETRIEVAL
# ============================================================

COMPARISON_KEYWORDS = (
    "compare",
    "comparison",
    " vs ",
    " vs.",
    "versus",
    "difference between",
    "which is better",
    "which one is better",
    "various institutes",
    "different institutes",
    "list of institutes",
    "list of colleges",
)


def wants_comparison(question: str) -> bool:
    """
    Heuristic: does this question ask to compare multiple
    institutes/courses, rather than look up a single one?

    Comparison questions need more search results and more
    retrieved chunks to have enough distinct institutes to
    compare -- but that also takes longer, so we only pay
    that cost when the question actually calls for it.
    """

    lowered = f" {question.lower()} "

    return any(
        keyword in lowered
        for keyword in COMPARISON_KEYWORDS
    )


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================
# In a chat thread, a short question like "what about the fees for
# that one?" makes no sense as a standalone web search -- it has no
# topic of its own, only a reference back to the previous turn. Merge
# in the previous question's topic before searching/ranking, while
# the LLM still sees the real, current question plus the full
# conversation history to answer naturally.

FOLLOWUP_REFERENCE_WORDS = (
    "it",
    "that",
    "this",
    "those",
    "these",
    "same",
    "again",
    "more",
    "further",
    "previous",
    "earlier",
    "above",
    "first one",
    "second one",
    "third one",
    "that one",
    "the first",
    "the second",
    "the third",
)


def looks_like_followup(question: str, history: list) -> bool:

    if not history:
        return False

    lowered = f" {question.lower()} "

    has_reference = any(
        f" {word} " in lowered
        for word in FOLLOWUP_REFERENCE_WORDS
    )

    # A genuine follow-up almost always has a reference word; this is
    # only a safety net for terse fragments like "fees?" or
    # "duration?" that don't. Kept low so a legitimate short-but-
    # standalone question ("data science courses in kochi") isn't
    # mistaken for one.
    is_short = len(question.split()) <= 3

    return has_reference or is_short


async def retrieve_from_mcp(
    question: str,
    history: list | None = None
):
    """
    Call the already-running MCP session to retrieve
    course information for the given question.
    """

    session = app.state.mcp_session

    if session is None:
        return None

    history = history or []

    # A short/referential follow-up has no search topic of its own --
    # merge in the previous question so search and chunk ranking stay
    # anchored to the real subject.
    if looks_like_followup(question, history):
        search_question = f"{history[-1]['question']} {question}"
    else:
        search_question = question

    # max_results controls how many pages get fetched (breadth);
    # top_k is now chunks kept PER PAGE, not a global total (see
    # retrieve_course_information's docstring) -- so it stays small
    # even for comparisons, where max_results does the heavy lifting.
    if wants_comparison(search_question):
        max_results, top_k = 10, 4
    else:
        max_results, top_k = 7, 4

    async with mcp_lock:

        result = await session.call_tool(
            "retrieve_course_information",
            arguments={
                "query": search_question,
                "max_results": max_results,
                "top_k": top_k
            }
        )

    for content in result.content:

        if hasattr(content, "text"):

            try:
                return json.loads(content.text)

            except json.JSONDecodeError:
                continue

    return None


# ============================================================
# REMOVE DUPLICATE SOURCES
# ============================================================

def remove_duplicate_sources(
    sources: list
):
    """
    Remove duplicate source URLs.
    """

    unique_sources = []
    seen_urls = set()

    for source in sources:

        url = source.get(
            "url",
            ""
        )

        title = source.get(
            "title",
            ""
        )

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        unique_sources.append({
            "title": title,
            "url": url
        })

    return unique_sources


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
async def ask_question(
    request: QuestionRequest,
    x_api_key: str | None = Header(default=None)
):
    """
    Main API endpoint.

    Flow:

    User Question
          ↓
       FastAPI
          ↓
        MCP
          ↓
     Web Search
          ↓
     Page Reading
          ↓
       Chunking
          ↓
      Retrieval
          ↓
        Groq
          ↓
     Final Answer
    """

    verify_api_key(x_api_key)

    question = request.question.strip()

    # Keep only the last few turns -- an unbounded history would
    # blow up the prompt size and latency for a long-running chat.
    history = [
        {"question": turn.question, "answer": turn.answer}
        for turn in request.history[-3:]
    ]

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:

        return JSONResponse(
            content={
                "status": "error",
                "message": (
                    "Question cannot be empty."
                )
            },
            media_type=(
                "application/json; charset=utf-8"
            )
        )

    try:

        # ----------------------------------------------------
        # STEP 1: Retrieve information through MCP
        # ----------------------------------------------------

        retrieved_data = await retrieve_from_mcp(
            question,
            history
        )

        if not retrieved_data:

            return JSONResponse(
                content={
                    "status": "error",
                    "message": (
                        "Could not retrieve "
                        "course information."
                    )
                },
                media_type=(
                    "application/json; charset=utf-8"
                )
            )

        # ----------------------------------------------------
        # STEP 2: Get retrieved RAG results
        # ----------------------------------------------------

        retrieved_results = (
            retrieved_data.get(
                "retrieved_results",
                []
            )
        )

        # ----------------------------------------------------
        # STEP 3: Handle no relevant results
        # ----------------------------------------------------

        if not retrieved_results:

            return JSONResponse(
                content={
                    "status": "success",
                    "question": question,
                    "answer": (
                        "I could not find relevant "
                        "course information for this "
                        "question."
                    ),
                    "sources": []
                },
                media_type=(
                    "application/json; charset=utf-8"
                )
            )

        # ----------------------------------------------------
        # STEP 4: Generate answer using Groq
        # ----------------------------------------------------

        answer_result = generate_rag_answer(
            query=question,
            retrieved_results=retrieved_results,
            history=history
        )

        # ----------------------------------------------------
        # STEP 5: Clean source list
        # ----------------------------------------------------

        sources = remove_duplicate_sources(
            answer_result.get(
                "sources",
                []
            )
        )

        # ----------------------------------------------------
        # STEP 6: Construct final API response
        # ----------------------------------------------------

        response_data = {
            "status": "success",

            "question": question,

            "answer": answer_result.get(
                "answer",
                ""
            ),

            "sources": sources,

            "retrieval": {
                "sources_checked": (
                    retrieved_data.get(
                        "sources_checked",
                        0
                    )
                ),

                "chunks_created": (
                    retrieved_data.get(
                        "chunks_created",
                        0
                    )
                ),

                "chunks_retrieved": len(
                    retrieved_results
                )
            }
        }

        # ----------------------------------------------------
        # STEP 7: Return UTF-8 JSON response
        # ----------------------------------------------------

        return JSONResponse(
            content=response_data,
            media_type=(
                "application/json; charset=utf-8"
            )
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        return JSONResponse(
            content={
                "status": "error",
                "message": (
                    "An error occurred while "
                    "processing the question."
                ),
                "details": str(e)
            },
            media_type=(
                "application/json; charset=utf-8"
            ),
            status_code=500
        )


# ============================================================
# LOCAL DEV ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )
