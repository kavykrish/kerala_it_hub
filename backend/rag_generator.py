import os
import time

from dotenv import load_dotenv
from groq import APIStatusError, Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv(
    "GROQ_API_KEY"
)

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please check your .env file."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)


MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# RETRY ON TRANSIENT RATE LIMITS
# ============================================================
# Groq's own client auto-retries 429s and 5xxs, but not this specific
# 413 "tokens per minute" rate_limit_exceeded error -- it's treated as
# a permanent "your request is too large" client error. In practice
# it's transient: the account's rolling per-minute token window clears
# itself within well under a minute, so the identical request usually
# succeeds shortly after. Retry it ourselves instead of surfacing a
# failure that would very likely have worked on the next try.

GROQ_RATE_LIMIT_RETRY_WAIT_SECONDS = 20
GROQ_RATE_LIMIT_MAX_RETRIES = 3


def call_groq_with_retry(**kwargs):

    last_error = None

    for attempt in range(GROQ_RATE_LIMIT_MAX_RETRIES + 1):

        try:
            return client.chat.completions.create(**kwargs)

        except APIStatusError as e:

            is_token_rate_limit = (
                e.status_code == 413
                and "rate_limit_exceeded" in str(e)
            )

            if (
                not is_token_rate_limit
                or attempt == GROQ_RATE_LIMIT_MAX_RETRIES
            ):
                raise

            last_error = e

            time.sleep(
                GROQ_RATE_LIMIT_RETRY_WAIT_SECONDS
            )

    raise last_error


# ============================================================
# RAG ANSWER GENERATOR
# ============================================================

def generate_rag_answer(
    query: str,
    retrieved_results: list,
    history: list | None = None
):
    """
    Generate an answer using retrieved RAG context.

    The LLM answers only from the retrieved
    information and avoids unsupported claims.

    history: recent {"question", "answer"} turns, oldest first, so
    follow-up questions ("what about the fees for that one?") can be
    understood and answered in context instead of as a fresh,
    unrelated question.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query:

        return {
            "answer": "Please provide a question.",
            "sources": []
        }


    # --------------------------------------------------------
    # Validate retrieved results
    # --------------------------------------------------------

    if not retrieved_results:

        return {
            "answer": (
                "I could not find relevant course "
                "information for this question."
            ),
            "sources": []
        }


    # ========================================================
    # PREPARE RETRIEVED CONTEXT
    # ========================================================
    # Capped to a fixed character budget -- Groq's free tier for this
    # model enforces an 8000 tokens-per-minute limit, and widening
    # retrieval depth (more chunks per page, more pages for
    # comparisons) had been pushing some requests to ~13,500 tokens,
    # failing with a 413 "Request too large" error. Two passes keep
    # breadth over depth under that cap: every distinct source gets
    # its single most relevant chunk first (so a comparison still
    # covers as many institutes as possible), and only once every
    # source has that does a second pass add each source's remaining
    # chunks for extra detail, while budget allows.

    CONTEXT_CHAR_BUDGET = 15000

    context_parts = []

    sources = []

    seen_sources = set()

    included_chunks = set()

    source_index = 1

    context_chars = 0


    # --------------------------------------------------------
    # Pass 1: one chunk per distinct source (breadth)
    # --------------------------------------------------------

    for result in retrieved_results:

        if context_chars >= CONTEXT_CHAR_BUDGET:
            break

        content = result.get(
            "content",
            ""
        )

        source_title = result.get(
            "source_title",
            ""
        )

        source_url = result.get(
            "source_url",
            ""
        )

        if not content:
            continue

        source_key = source_url.strip()

        if (
            not source_key
            or source_key in seen_sources
        ):
            continue

        part = f"""
SOURCE {source_index}

Title:
{source_title}

URL:
{source_url}

Content:
{content}
"""

        context_parts.append(
            part
        )

        context_chars += len(
            part
        )

        seen_sources.add(
            source_key
        )

        included_chunks.add(
            (source_key, content)
        )

        sources.append({
            "title": source_title,
            "url": source_url
        })

        source_index += 1


    # --------------------------------------------------------
    # Pass 2: remaining chunks for sources already included
    # (extra detail, only while budget allows)
    # --------------------------------------------------------

    for result in retrieved_results:

        if context_chars >= CONTEXT_CHAR_BUDGET:
            break

        content = result.get(
            "content",
            ""
        )

        source_title = result.get(
            "source_title",
            ""
        )

        source_url = result.get(
            "source_url",
            ""
        )

        if not content:
            continue

        source_key = source_url.strip()

        if source_key not in seen_sources:
            continue

        if (source_key, content) in included_chunks:
            continue

        part = f"""
ADDITIONAL CONTENT

Title:
{source_title}

URL:
{source_url}

Content:
{content}
"""

        context_parts.append(
            part
        )

        context_chars += len(
            part
        )

        included_chunks.add(
            (source_key, content)
        )


    # --------------------------------------------------------
    # Combine all retrieved content
    # --------------------------------------------------------

    context = "\n".join(
        context_parts
    )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are Kerala IT Hub, an AI-powered IT course
and institute information assistant.

Your job is to answer questions about IT and
technology courses in Kerala.

Use ONLY the information provided in the
retrieved context.

IMPORTANT RULES:

1. Do not invent course names, fees, durations,
   eligibility requirements, locations, admission
   details, curriculum details, or other course
   information.

2. Treat the retrieved information as evidence
   from the listed sources, not as complete
   information about every IT course in Kerala.

3. Never use words such as:
   "all", "only", "every", "none", "best",
   "most", or "the complete list"
   unless the retrieved information explicitly
   supports that claim.

4. If the question asks for a list of courses,
   present the courses found in the retrieved
   results.

   Clearly indicate that the list is based on
   the retrieved sources and may not represent
   every course available in Kerala.

5. If a course or institute was not found in the
   retrieved results, do not claim that it
   does not exist.

6. If the retrieved information does not contain
   enough evidence to answer the question,
   clearly say that the information was not found
   in the retrieved sources.

7. Do not combine eligibility, fees, duration,
   curriculum, or other details from different
   institutes and present them as if they belong
   to the same institute.

8. When information comes from multiple institutes,
   clearly associate each detail with the correct
   institute or course.

9. Do not treat marketing claims such as
   "best", "No.1", "guaranteed job", "100%
   placement", or similar promotional statements
   as verified facts.

10. Preserve uncertainty.

    If the source says that fees, duration,
    eligibility, or other details depend on
    the batch, program, or conditions, do not
    provide a fixed value.

11. Prefer specific information over general
    assumptions.

12. Keep the answer concise and easy to understand.

13. When listing courses, institutes, or programs,
    mention the associated institute or provider
    whenever that information is available.

14. Do not claim that the retrieved list is
    complete unless the source explicitly states
    that it is complete.

15. Do not infer information that is not directly
    supported by the retrieved content.

16. At the end of the answer, provide a
    "Sources" section containing the relevant
    source titles and URLs.

    List sources as plain numbered lines
    ("1. Title - URL"), one per line. Never format
    the Sources section as a markdown table (no "|"
    characters), and never mix source citations into
    the comparison table from rule 18 as if they were
    another institute row.

    List each distinct source URL only ONCE in this
    section, even if you drew on it for several rows,
    bullet points, or parts of the answer.

    Never write a URL, or a numbered citation like
    "1. Title - URL", anywhere else in the answer.
    The body of the answer (before "Sources") should
    read as plain prose, a table, or bullet points --
    never a list of links. If you want to credit a
    specific claim to a source while writing the body,
    name the institute or publication in words instead
    (e.g. "according to Coursera") rather than pasting
    its URL inline.

17. When presenting information from a source,
    do not change its meaning or create facts
    that are not present in the retrieved content.

18. Whenever the retrieved information covers two
    or more distinct institutes or courses, present
    them as a markdown table so they can be compared
    side by side, instead of separate paragraphs.

    Use a header row and include whichever of these
    columns the retrieved content actually supports:
    Institute, Course, Duration, Fees, Eligibility,
    Mode (Online/Offline/Hybrid), Location,
    Certification.

    Leave a cell blank (write "-") rather than
    guessing if that detail was not present in the
    retrieved content for that row.

    Never put one column's value into a different
    column -- e.g. if you only know the LOCATION but
    not the duration, write "-" in the Duration cell.
    Do not fill it with the location, the institute
    name, or any other value that belongs to a
    different column. A cell's value must specifically
    answer that column's own question, or be "-".

19. Be as detailed and specific as the retrieved
    content allows. Prefer including a concrete
    detail (an exact duration, fee, eligibility
    requirement, or admission step) over a vague
    summary, as long as that detail is explicitly
    present in the retrieved content.

19b. When comparing two career paths, fields, or
    course TYPES rather than specific institutes
    (e.g. "data science vs data analytics", where
    rule 18's table doesn't apply), structure the
    answer with a bold subheading for each side
    followed by its own bullet points, instead of
    interleaving both sides' points into one mixed
    list. For example:

    **Data Science covers:**
    - point
    - point

    **Data Analytics covers:**
    - point
    - point

    Keep every bullet under the side it actually
    belongs to -- do not mix a Data Science point
    into the Data Analytics section or vice versa.

20. If a previous conversation is provided, use it
    only to understand what the current question
    refers to (e.g. "it", "that one", "the second
    institute"). Still answer strictly from the
    retrieved course information provided for THIS
    question, not from memory of the previous answer.

    Do not repeat the previous answer's full content
    -- focus on what the current question actually
    asks, and refer back briefly ("as mentioned for
    X above") only where useful.
"""


    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    history_section = ""

    if history:

        history_lines = []

        for turn in history:

            history_lines.append(
                f"User: {turn.get('question', '')}"
            )

            history_lines.append(
                f"Assistant: {turn.get('answer', '')}"
            )

        history_section = (
            "Previous conversation (for context only -- "
            "the current question is answered strictly "
            "from the retrieved course information below, "
            "not from this history):\n\n"
            + "\n".join(history_lines)
            + "\n\n\n"
        )


    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""
{history_section}Question:

{query}


Retrieved course information:

{context}


Using only the retrieved course information,
answer the question accurately.

Remember:

- Do not invent information.
- Do not assume missing information.
- Clearly distinguish between different institutes.
- If the results are incomplete, say that they
  are based on the retrieved sources.
- Do not claim that a course does not exist simply
  because it was not found in the retrieved results.
- If two or more institutes or courses appear in the
  retrieved information, present them as a markdown
  comparison table (see rule 18).
- Include specific details (durations, fees,
  eligibility, etc.) wherever the retrieved content
  actually provides them (see rule 19).
"""


    # ========================================================
    # GROQ API CALL
    # ========================================================

    response = call_groq_with_retry(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0,

        # Without max_tokens, Groq reserves capacity for a very large
        # default completion length against the account's
        # tokens-per-minute limit -- on top of the actual prompt
        # tokens -- which was the real cause of 413 "Request too
        # large" errors even after capping context size.
        max_tokens=2048,

        # openai/gpt-oss-20b is a reasoning model: its internal
        # "thinking" tokens are billed as part of completion_tokens,
        # separate from the visible answer. At the default reasoning
        # effort, a harder comparison task was consuming the *entire*
        # max_tokens budget on reasoning alone, leaving nothing for
        # the actual answer (finish_reason: length, empty content).
        # "low" is enough for this task (a grounded lookup/comparison
        # over already-retrieved text, not novel problem-solving) and
        # confirmed fixing it: reasoning dropped to ~100-600 tokens,
        # finish_reason back to "stop", with a complete answer.
        reasoning_effort="low"
    )


    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    answer = (
        response
        .choices[0]
        .message
        .content
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "answer": answer,
        "sources": sources
    }